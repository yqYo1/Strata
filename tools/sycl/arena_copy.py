#!/usr/bin/env python3
"""Main-owned bounded positive Arena-copy model check; no speed/quality claim."""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
import cpu_capacity as assets
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision
from build_engine import blobs

P, B, A = gen.P, gen.B, gen.A
ROOT = B / 'decode-arena-copy-check-v1'
RESULT = A / 'decode-arena-copy-current.json'
BUILD = A / 'native-gdn-engine-build.json'
COMPONENT = A / 'verifier-copy-safety-current.json'
BUILD_RECORD = B / 'main-native-gdn-engine/record.json'
COPY_KEYS = ('submitted_blobs','submitted_bytes','completed_blobs','completed_bytes',
             'host_view_groups','gpu_miss_groups','cpu_unique_misses','cpu_entries')


def current_build():
    compact, component = gen.committed(BUILD), gen.committed(COMPONENT)
    full = json.loads(BUILD_RECORD.read_text())
    guards.require(full['passed'] and full['complete'] and full['ownership_closed'] and
        not full['active'] and all(full[k] == compact[k] for k in
        ('source_commit','binary','fixtures','compiler','configure_argv')), 'closed committed build required')
    guards.require(component['passed'] and component['complete'] and component['ownership_closed'] and
        not component['active'] and not component['fresh_faults'] and component['binary']==full['binary'] and
        component['fixtures']==full['fixtures'] and component['source_commit']==full['source_commit'] and
        len(component['checks'])==3 and 'cases=17 native_variable_stride=PASS' in component['checks'][0]['cases'][0]
        and len(component['checks'][1]['cases'])==10 and len(component['checks'][2]['cases'])==8,
        'matching passed native provider/submission/verifier component gates required')
    count = 0
    for source in [full['project'], *full['dependencies']]:
        recovered = {n:hashlib.sha256(data).hexdigest() for n,data in
            blobs(Path(source['root']),source['revision'],source['source_sha256'])}
        guards.require(recovered==source['source_sha256'], 'original build inputs not recoverable')
        for name, expected in recovered.items():
            guards.require(gen.common.digest(Path(source['root'])/name)==expected, 'native build input changed: '+name)
        count += len(recovered)
    for item in [full['binary'],*full['fixtures'].values()]:
        guards.require(gen.common.digest(item['path'])==item['sha256'], 'compiled artifact changed')
    return full, count


def prepare():
    full, count = current_build()
    base = gen.committed(assets.RESULT)
    guards.require(base['passed'] and base['complete'] and base['ownership_closed'] and
        not base['active'] and not base['fresh_faults'] and base['comparison_summary']['decision']=='reduced_capacity_slower',
        'closed capacity discriminator required')
    guards.require(not ROOT.exists() and not RESULT.exists(), 'inspect existing plan/owner; no automatic retry')
    revision = gen.git('rev-parse','HEAD').decode().strip()
    pins = gen.benchmark_pins(revision)
    settings = gen.committed(gen.SETTINGS)
    config = copy.deepcopy(base['config'])
    flags = config['args']
    flags[flags.index('--pool-workers')+1] = '5'
    flags[flags.index('--pcie-frac')+1] = '0.25'
    flags[flags.index('--expert-profile')+1] = str(ROOT/'expert-profile.bin')
    config.update(log=str(ROOT/'engine.stderr'),env={})
    prompts = assets.check_prompts(config,base['environment'])
    model = assets.model_pins(config,full=True)
    ROOT.mkdir(mode=0o700)
    for name in ('cache','tmp'): (ROOT/name).mkdir()
    old_root = base['environment']['STRATA_MEASUREMENT_ROOT']
    env = {k:v.replace(old_root,str(ROOT)) for k,v in base['environment'].items()}
    env.update(STRATA_SYCL_VERIFY_ARENA_COPY='1',UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',STRATA_TRACE='1')
    env.update({k:v for k,v in settings['environment'].items() if k.startswith('UR_LOG_') and k!='UR_LOG_TRACING'})
    for name in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission',
                 'STRATA_SYCL_PREFILL_SAME_QUEUE'):
        guards.require(env[name]==settings['environment'][name], 'qualified safety setting changed')
    shutil.copyfile(P/'data/expert-profile.bin', ROOT/'expert-profile.bin')
    guards.require(gen.common.digest(ROOT/'expert-profile.bin')==base['profile_sha256'], 'profile changed')
    (ROOT/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    record = dict(mode='arena-copy-check',source_commit=revision,build_source_commit=full['source_commit'],
        binary_origin_source_commit=full.get('binary_origin_source_commit',full['source_commit']),
        binary=full['binary'],source_build_record=dict(path=str(BUILD_RECORD),sha256=gen.common.digest(BUILD_RECORD)),
        original_build_inputs_recovered=count,component_result_sha256=gen.common.digest(COMPONENT),
        runtime_asset_delta=base['runtime_asset_delta'],root=str(ROOT),environment=env,config=config,
        command=[full['binary']['path'],'--serve',*flags],check_prompts=prompts,model_assets=model,
        model_assets_fingerprint=assets.fingerprint(model),machine_identity=assets.machine_identity(),
        tokenizer_assets={str(Path(config['tokenizer'])/name):gen.common.digest(Path(config['tokenizer'])/name)
            for name in ('vocab.json','merges.txt','token_type.json','chat_template.jinja')},
        source_count=len(pins),source_aggregate_sha256=assets.fingerprint(pins),
        config_sha256=gen.common.digest(ROOT/'config.json'),profile_sha256=base['profile_sha256'],
        settings_sha256=gen.common.digest(gen.SETTINGS),complete=False,passed=False,active=False,
        model_quality_qualified=False,model_performance_measured=False,small_requests=[],
        budgets=dict(wall=900,cpu=54000,rss_cap=96<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20),
        gate='One owned native process, two thinking-template questions,1..16 generated tokens each, pcie-frac0.25/pool5, nonzero immutable-view copy groups in both requests, exact submitted/completed blob+byte conservation and positive remaining CPU jobs. Require no protocol/runtime/fresh-driver fault and unchanged source/model/tokenizer/runtime/boot. No automatic retry.',
        limits='Tiny safety/actual-route check only. No task quality, speed comparison, physical PCIe bytes, saturation, physical262144/session/RESTORE or70tok/s qualification.',
        consumer='Admit a separate committed32K+ strong/weak thinking generation comparison; retire normal full traces after compact route/closure evidence is committed.')
    gen.runtime_assets_unchanged(record)
    (ROOT/'plan.json').write_text(json.dumps(record,indent=2)+'\n')
    RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print('Prepared bounded positive Arena-copy model check; commit the exact generated plan before run.')


def run():
    record = gen.committed(RESULT)
    guards.require(record==json.loads((ROOT/'plan.json').read_text()) and not record['complete'] and
        not (ROOT/'record.json').exists(), 'fresh frozen plan required; no restart')
    full,count = current_build()
    guards.require(full['binary']==record['binary'] and count==record['original_build_inputs_recovered'] and
        gen.common.digest(BUILD_RECORD)==record['source_build_record']['sha256'] and
        gen.common.digest(COMPONENT)==record['component_result_sha256'], 'build/component proof changed')
    pins = gen.benchmark_pins(record['source_commit'])
    guards.require(len(pins)==record['source_count'] and assets.fingerprint(pins)==record['source_aggregate_sha256'],
        'controller/benchmark inputs changed')
    for path, expected in [(gen.SETTINGS,record['settings_sha256']),(ROOT/'config.json',record['config_sha256']),
            (ROOT/'expert-profile.bin',record['profile_sha256']),*record['tokenizer_assets'].items()]:
        guards.require(gen.common.digest(path)==expected,'input changed: '+str(path))
    gen.runtime_assets_unchanged(record)
    guards.require(assets.model_pins(record['config'],record['model_assets'])==record['model_assets'] and
        assets.machine_identity()==record['machine_identity'], 'frozen model/platform differs')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    before = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True)
    cursor = next(x.split(': ',1)[1] for x in before.splitlines() if x.startswith('-- cursor: '))
    owner = None
    def persist():
        record['active'] = bool(owner and owner.active is not None)
        (ROOT/'record.json').write_text(json.dumps(dict(record,owned_commands=owner.commands if owner else []),indent=2)+'\n')
    def interact(io,entry):
        while not io.line().startswith('READY '): pass
        for prompt in record['check_prompts']:
            io.send(('GEN 16 '+','.join(map(str,prompt))+'\n').encode())
            tokens=[]
            while True:
                line=io.line();guards.require(not line.startswith('ERR'),line)
                if line.startswith('T '): tokens.append(int(line.split()[1]))
                if line.startswith('DONE '):
                    fields=line.split();guards.require(int(fields[1])==len(tokens) and 0<len(tokens)<=16 and
                        fields[5] in ('length','stop','eos') and all(0<=x<248320 for x in tokens),'incomplete request')
                    record['small_requests'].append(dict(done=line,generated=len(tokens),output_sha256=assets.fingerprint(tokens)));break
        io.send(b'QUIT\n');io.proc.stdin.close();io.exit(60)
    owner=supervision.Owner(ROOT/'owned',persist)
    try:
        entry,_,err=owner.run('check',record['command'],record['environment'],P,interaction=interact,**record['budgets'])
        record.update(complete=bool(supervision.closed(entry)),exit_code=entry.get('exit_code'))
        guards.require(record['complete'] and entry['exit_code']==0,'owned command failed')
        log=err.read_text(errors='replace');assets.verify_pool(log,5)
        pattern=r'^strata serve: SYCL_EXPERT_COPY '+r' '.join(name+r'=(\d+)' for name in COPY_KEYS)+r'$'
        rows=[dict(zip(COPY_KEYS,map(int,row))) for row in re.findall(pattern,log,re.M)]
        record['copy_route_requests']=rows
        guards.require(len(rows)==len(record['small_requests'])==2 and
            'ordinary-Arena copy enabled mode=0' in log,'actual enabled route/requests missing')
        for row in rows:
            guards.require(0<row['submitted_blobs']==row['completed_blobs']==row['host_view_groups']==row['gpu_miss_groups']
                and 0<row['submitted_bytes']==row['completed_bytes'] and row['cpu_unique_misses']>0 and row['cpu_entries']>0,
                'copy/CPU route conservation failed')
        current_build()
        guards.require(gen.benchmark_pins(record['source_commit'])==pins,'controller inputs changed during run')
        gen.runtime_assets_unchanged(record)
        guards.require(assets.machine_identity()==record['machine_identity'] and
            assets.model_pins(record['config'],record['model_assets'])==record['model_assets'],'model/platform changed')
        record.update(passed=True,actual_host_copy_route_confirmed=True)
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
        try:record['driver_after']=guards.driver_guard()
        except Exception as error:record.update(driver_guard_error=str(error),passed=False)
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
