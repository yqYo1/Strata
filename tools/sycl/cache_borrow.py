#!/usr/bin/env python3
"""Main-owned positive cache-loan admission; no clean speed or quality claim."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
import query_profile as q

gen, assets, guards, supervision = q.gen, q.assets, q.guards, q.supervision
P, B, A = q.P, q.B, q.A
ROOT = B / 'cache-borrow-check-v1'
RESULT = A / 'cache-borrow-current.json'
PROMPT_LENGTHS = (8194, 257)


def prompts(config, environment):
    # Retain the recipe and hashes, not thousands of reproducible token IDs.
    script = '''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,str(Path(sys.argv[1])/'tools'))
from strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate
p=Path(sys.argv[2]);v=json.loads((p/'vocab.json').read_text());tokens=[None]*len(v)
for text,index in v.items():tokens[index]=text
t=Tokenizer(tokens,(p/'merges.txt').read_text().split('\\n'),json.loads((p/'token_type.json').read_text()))
template=ChatTemplate(p/'chat_template.jinja')
marker='CACHE_LOAN_BODY_MARKER'
rendered=template.render([dict(role='user',content=marker)],enable_thinking=True)
assert rendered.count(marker)==1
prefix,_,suffix=rendered.partition(marker)
head=t.encode(prefix,parse_special=True);tail=t.encode(suffix,parse_special=True)
texts=['First review: explain queue dependencies and memory reuse in detail. ',
       'Second review: explain why an expert cache must be refilled before decode. ']
result=[]
for length,text in zip((8194,257),texts):
 body=t.encode(text,parse_special=False);n=length-len(head)-len(tail)
 assert n>0 and body
 result.append(head+(body*((n+len(body)-1)//len(body)))[:n]+tail)
print(json.dumps(result))
'''
    values = json.loads(subprocess.check_output([
        str(Path(environment['PATH'].split(':')[0])/'python'), '-c', script,
        str(P), config['tokenizer']], env=environment, cwd=P, timeout=30, text=True))
    guards.require(tuple(map(len, values)) == PROMPT_LENGTHS and all(
        type(x) is int and 0 <= x < 248320 for row in values for x in row), 'invalid cache-loan prompts')
    return values


def loan_evidence(log):
    pattern = r'^strata loan audit: phase=layout chunk=(\d+) first=(\d+) slots=(\d+) bytes=(\d+) need=(\d+) lent=(\d+) mask_match=1$'
    layouts = [dict(zip(('chunk','first','slots','bytes','need','lent'), map(int, row)))
               for row in re.findall(pattern, log, re.M)]
    tables = [dict(phase=p,entries=int(n),free_bytes=int(f),total_bytes=int(t)) for p,n,f,t in re.findall(
        r'^strata loan audit: phase=(lend|refill) table_entries=(\d+) table_match=1 free_bytes=(\d+) total_bytes=(\d+)$',log,re.M)]
    payloads = [dict(zip(('rows','bytes','lent'),map(int,row))) for row in re.findall(
        r'^strata loan audit: phase=payload rows=(\d+) bytes=(\d+) lent=(\d+) same_slots=1 payload_match=1$',log,re.M)]
    guards.require(len(layouts)==len(payloads)==2 and len(tables)==4 and
        [x['phase'] for x in tables]==['lend','refill','lend','refill'] and
        log.count('strata loan audit:')==8,'missing/ambiguous loan checks')
    for layout,payload in zip(layouts,payloads):
        guards.require(layout['slots']==1280 and 128 <= layout['first'] < 1280 and
            layout['bytes'] >= layout['need'] > 0 and
            layout['lent']==1280-layout['first']==payload['lent'] and
            payload['rows']==1280 and payload['bytes']>layout['bytes'], 'loan/payload conservation failed')
    guards.require(256 < layouts[0]['chunk'] <= 8192 and layouts[1]['chunk']==256 and
        layouts[1]['first']>layouts[0]['first'] and layouts[1]['bytes']<layouts[0]['bytes'],
        'distinct rounded chunks and tail offsets were not exercised')
    guards.require(all(x['entries']==48*512 and 0 <= x['free_bytes'] <= x['total_bytes'] for x in tables),
        'residency/VRAM observation invalid')
    phases = re.findall(r'^strata mtp: prompt expert phase=(\S+) free_bytes=\d+ total_bytes=\d+$',log,re.M)
    guards.require(phases==['before-detach','after-detach','after-prompt-before-restore','after-restore'],
        'configured MTP detach/restore was not exercised exactly once')
    return dict(layouts=layouts,tables=tables,payloads=payloads,mtp_phases=phases)


def prepare():
    full, count = q.current_build()
    base = gen.committed(assets.RESULT)
    guards.require(base['passed'] and base['ownership_closed'] and not base['active'] and
        not base['fresh_faults'] and not ROOT.exists() and not RESULT.exists(), 'closed baseline/fresh root required')
    revision = gen.git('rev-parse','HEAD').decode().strip()
    pins = gen.benchmark_pins(revision)
    settings = gen.committed(gen.SETTINGS)
    config = copy.deepcopy(base['config']); flags = config['args']
    for flag,value in (('--expert-cache','1280'),('--pool-workers','5'),('--pcie-frac','0'),
                       ('--prefill','8192'),('--expert-profile',str(ROOT/'expert-profile.bin'))):
        flags[flags.index(flag)+1] = value
    guards.require('--no-prefill-borrow' in flags and '--no-spec-split' in flags and
        flags[flags.index('--spec')+1]=='4','expected G1/T4 default route required')
    flags.remove('--no-prefill-borrow')
    config.update(log=str(ROOT/'engine.stderr'),env={})
    old_root = base['environment']['STRATA_MEASUREMENT_ROOT']
    env = {k:v.replace(old_root,str(ROOT)) for k,v in base['environment'].items()}
    env.update(STRATA_SYCL_QUERY_PROFILE='0',STRATA_SYCL_CACHE_LOAN_AUDIT='1',
        STRATA_SYCL_VERIFY_ARENA_COPY='0',STRATA_SYCL_VERIFY_PCIE_EARLY='0',
        STRATA_SYCL_VERIFY_PCIE_EARLY_AUDIT='0',UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',STRATA_TRACE='1')
    env.update({k:v for k,v in settings['environment'].items() if k.startswith('UR_LOG_') and k!='UR_LOG_TRACING'})
    for name in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission',
                 'STRATA_SYCL_PREFILL_SAME_QUEUE'):
        guards.require(env[name]==settings['environment'][name],'qualified safety setting differs')
    q.runtime_assets_unchanged(dict(mode='cache-borrow-check',runtime_asset_delta=base['runtime_asset_delta']))
    model = assets.model_pins(config,base['model_assets'])
    values = prompts(config,env)
    ROOT.mkdir(mode=0o700)
    for name in ('cache','tmp'): (ROOT/name).mkdir()
    shutil.copyfile(P/'data/expert-profile.bin',ROOT/'expert-profile.bin')
    guards.require(gen.common.digest(ROOT/'expert-profile.bin')==base['profile_sha256'],'profile changed')
    (ROOT/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    record = dict(mode='cache-borrow-check',source_commit=revision,build_source_commit=full['source_commit'],
        binary=full['binary'],source_build_record=dict(path=str(q.BUILD_RECORD),sha256=gen.common.digest(q.BUILD_RECORD)),
        original_build_inputs_recovered=count,component_result_sha256=gen.common.digest(q.COMPONENT),
        runtime_asset_delta=base['runtime_asset_delta'],root=str(ROOT),environment=env,config=config,
        command=[full['binary']['path'],'--serve',*flags],prompt_lengths=PROMPT_LENGTHS,
        prompt_sha256=[assets.fingerprint(row) for row in values],model_assets=model,
        machine_identity=assets.machine_identity(),
        tokenizer_assets={str(Path(config['tokenizer'])/name):gen.common.digest(Path(config['tokenizer'])/name)
            for name in ('vocab.json','merges.txt','token_type.json','chat_template.jinja')},
        source_count=len(pins),source_aggregate_sha256=assets.fingerprint(pins),
        config_sha256=gen.common.digest(ROOT/'config.json'),profile_sha256=base['profile_sha256'],
        settings_sha256=gen.common.digest(gen.SETTINGS),complete=False,passed=False,active=False,
        model_quality_qualified=False,model_performance_measured=False,model_route_executed=False,small_requests=[],
        budgets=dict(wall=900,cpu=54000,rss_cap=96<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20),
        gate='One process;1280 actual slots; two thinking-template requests8194/257 with16 generated tokens each. Positive tail loans at distinct chunks/offsets; GPU residency equals host after each lend/refill; full resident payload including unlent prefix equals immutable RAM; same slots restored; configured MTP detach/restore once; G1/T4 reached; no protocol/runtime/fresh-driver fault; source/model/runtime/boot unchanged. No retry.',
        limits='Diagnostic admission only. Readback audit adds work; no clean speed, task quality, hardware saturation, full physical262144/session/RESTORE or70tok/s qualification.',
        consumer='Prerequisite for matched owned/borrowed capacity and32K+ long-generation comparisons. Retire successful raw output after compact evidence is committed.')
    (ROOT/'plan.json').write_text(json.dumps(record,indent=2)+'\n')
    RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print('Prepared positive cache-loan check; commit the generated plan before run.')


def run():
    record = gen.committed(RESULT)
    guards.require(record==json.loads((ROOT/'plan.json').read_text()) and not record['complete'] and
        not (ROOT/'record.json').exists(),'fresh frozen plan required; no retry')
    full,count = q.current_build()
    guards.require(full['binary']==record['binary'] and count==record['original_build_inputs_recovered'] and
        gen.common.digest(q.BUILD_RECORD)==record['source_build_record']['sha256'] and
        gen.common.digest(q.COMPONENT)==record['component_result_sha256'],'build/component evidence changed')
    pins = gen.benchmark_pins(record['source_commit'])
    guards.require(len(pins)==record['source_count'] and assets.fingerprint(pins)==record['source_aggregate_sha256'],
        'controller inputs changed')
    for path, expected in [(gen.SETTINGS,record['settings_sha256']),(ROOT/'config.json',record['config_sha256']),
            (ROOT/'expert-profile.bin',record['profile_sha256']),*record['tokenizer_assets'].items()]:
        guards.require(gen.common.digest(path)==expected,'input changed: '+str(path))
    values = prompts(record['config'],record['environment'])
    guards.require([assets.fingerprint(row) for row in values]==record['prompt_sha256'],'prompt recipe changed')
    q.runtime_assets_unchanged(record)
    guards.require(assets.model_pins(record['config'],record['model_assets'])==record['model_assets'] and
        assets.machine_identity()==record['machine_identity'],'model/platform differs')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    journal = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True)
    cursor = next(x.split(': ',1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
    owner = None
    def persist():
        record['active'] = bool(owner and owner.active is not None)
        (ROOT/'record.json').write_text(json.dumps(dict(record,owned_commands=owner.commands if owner else []),indent=2)+'\n')
    def interact(io,entry):
        while not io.line().startswith('READY '): pass
        for prompt in values:
            io.send(('GEN 16 '+','.join(map(str,prompt))+'\n').encode()); tokens=[]
            while True:
                line=io.line();guards.require(not line.startswith('ERR'),line)
                if line.startswith('T '): tokens.append(int(line.split()[1]))
                if line.startswith('DONE '):
                    fields=line.split();guards.require(int(fields[1])==len(tokens)==16 and
                        fields[5] in ('length','stop','eos') and all(0<=x<248320 for x in tokens),'incomplete request')
                    record['small_requests'].append(dict(done=line,input_tokens=len(prompt),generated=len(tokens),
                        output_sha256=assets.fingerprint(tokens)));break
        io.send(b'QUIT\n');io.proc.stdin.close();io.exit(60)
    owner=supervision.Owner(ROOT/'owned',persist)
    try:
        entry,_,err=owner.run('check',record['command'],record['environment'],P,interaction=interact,**record['budgets'])
        record.update(complete=supervision.closed(entry),exit_code=entry.get('exit_code'))
        guards.require(record['complete'] and entry['exit_code']==0,'owned command failed')
        log=err.read_text(errors='replace');assets.verify_pool(log,5)
        guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|cleanup failed',log),
            'runtime error in loan diagnostic')
        headers=re.findall(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*$',log,re.M)
        guards.require([tuple(map(int,row)) for row in headers]==[(n,0,n,16) for n in PROMPT_LENGTHS] and
            all(f'captured the {n}-token window' in log for n in (1,4)), 'request/capture evidence differs')
        record['loan_checks']=loan_evidence(log)
        q.current_build();q.runtime_assets_unchanged(record)
        guards.require(gen.benchmark_pins(record['source_commit'])==pins and
            assets.model_pins(record['config'],record['model_assets'])==record['model_assets'] and
            assets.machine_identity()==record['machine_identity'],'inputs changed during run')
        record.update(passed=True,model_route_executed=True)
    except BaseException as error:
        record['error']=str(error)
        path=ROOT/'owned/check.stderr'
        if path.exists():record['error_excerpt']=path.read_text(errors='replace')[-4096:]
    finally:
        record['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        after=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True)
        record['fresh_faults']=[x for x in after.splitlines() if re.search(
            r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]
        record['boot_unchanged']=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        try: record['driver_after']=guards.driver_guard()
        except Exception as error: record.update(driver_guard_error=str(error),passed=False)
        record['passed']=bool(record['passed'] and record['ownership_closed'] and record['boot_unchanged'] and not record['fresh_faults'])
        persist();RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record.get(k) for k in ('complete','passed','ownership_closed','error')}))
    return 0 if record['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        return prepare() if args.action=='prepare' else run()


if __name__=='__main__':sys.exit(main())
