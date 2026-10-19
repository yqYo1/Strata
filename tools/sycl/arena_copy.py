#!/usr/bin/env python3
"""Main-owned long-generation Arena-copy check and six-process clean screen."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

sys.dont_write_bytecode = True
import cpu_capacity as assets
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision
import resource_engine as resources
from build_engine import blobs

P, B, A = gen.P, gen.B, gen.A
RESULT = A / 'decode-arena-generation-current.json'
QUALIFIED = A / 'decode-arena-copy-current.json'
BUILD = A / 'native-gdn-engine-build.json'
COMPONENT = A / 'verifier-copy-safety-current.json'
BUILD_RECORD = B / 'main-native-gdn-engine/record.json'
ORDER = (0, 1, 1, 0, 0, 1)
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
        len(component['checks'])==4 and 'cases=17 native_variable_stride=PASS' in component['checks'][0]['cases'][0]
        and len(component['checks'][1]['cases'])==10 and len(component['checks'][2]['cases'])==8
        and len(component['checks'][3]['cases'])==16 and component.get('grouped_copy_projection_contract_passed') is True,
        'matching passed provider/submission/verifier/grouped-output component gates required')
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


def comparison_plan():
    return dict(version=1, order=list(ORDER), pairs=[[0,1],[3,2],[4,5]],
        workload='regular-code', minimum_input=32768, minimum_generated=1024,
        thinking=True, workers=5, pcie_frac=0.25, paired_ratio_threshold=1.02,
        stopping='One validated long candidate check, then six fresh clean processes only. Stop on any execution/input/fault/eligibility failure; no automatic retries or extra samples.',
        decision='Compare candidate/base decode tok/s in each pair. Signal faster only if all three ratios>1 and geometric mean>1.02; slower only if all three<1 and geomean<1/1.02; otherwise inconclusive. No candidate adoption from this strong-MTP screen.',
        limits='Decode throughput screen, not whole task quality, PCIe wire traffic or hardware saturation. Generated tokens and acceptance can differ; exact generated text is not required. Weak-MTP, agent-session, long-prefill nonregression and physical262144/session/RESTORE qualification remain separate.')


def copy_evidence(log, sample, arm):
    header = r'^strata serve: prompt (\d+) tokens = \d+ reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*\n'
    footer = r'strata serve: SYCL_EXPERT_COPY '+r' '.join(k+r'=(\d+)' for k in COPY_KEYS)+r'$'
    matches = [r for r in re.findall(header+footer,log,re.M)
        if int(r[0])==sample['prompt_tokens'] and int(r[1])==sample['read_tokens']
        and int(r[2])==sample['usage_generated_tokens']]
    guards.require(len(matches)==1, 'ambiguous or absent measured-request copy footer')
    row=dict(zip(COPY_KEYS,map(int,matches[0][3:])))
    guards.require(row['cpu_unique_misses']>0 and row['cpu_entries']>0, 'remaining CPU expert work missing')
    if arm:
        guards.require(0<row['submitted_blobs']==row['completed_blobs']==row['host_view_groups']==row['gpu_miss_groups']
            and 0<row['submitted_bytes']==row['completed_bytes'] and 'ordinary-Arena copy enabled mode=0' in log,
            'positive copy route/conservation failed')
    else:
        guards.require(all(row[k]==0 for k in COPY_KEYS[:6]) and 'ordinary-Arena copy enabled mode=0' not in log,
            'control unexpectedly enabled nonresident expert copies')
    row['logical_copy_bytes_per_emitted_token']=row['completed_bytes']/sample['usage_generated_tokens']
    return row


def screen(rows):
    guards.require(len(rows)==6 and [x['arm'] for x in rows]==list(ORDER) and
        all(x['passed'] and x['ownership_closed'] and x['request']['controller_eligible']
            and gen.sample_eligible(x['request']) for x in rows),
        'six eligible closed processes required')
    guards.require(len({(x['source_commit'],x['binary_sha256']) for x in rows})==1,
        'screen source/binary identities differ')
    guards.require(len({(x['request']['request_sha256'],x['request']['read_tokens']) for x in rows})==1,
        'screen prompts/input work differ')
    ratios=[rows[b]['request']['decode_tps']/rows[a]['request']['decode_tps'] for a,b in comparison_plan()['pairs']]
    geo=math.exp(sum(map(math.log,ratios))/3)
    decision=('candidate_faster_screen' if all(r>1 for r in ratios) and geo>1.02 else
              'candidate_slower_screen' if all(r<1 for r in ratios) and geo<1/1.02 else 'inconclusive')
    return dict(paired_ratios=ratios,geomean_ratio=geo,decision=decision,adoption_qualified=False,
        limits=comparison_plan()['limits'])


def prepare(stage, cohort):
    check=stage=='check'
    full,count=current_build()
    positive=gen.committed(QUALIFIED)
    guards.require(positive['passed'] and positive['complete'] and positive['ownership_closed'] and
        positive['actual_host_copy_route_confirmed'] and positive['binary']==full['binary'] and
        not positive['fresh_faults'], 'matching positive tiny model route required')
    if check:
        guards.require(not RESULT.exists(), 'inspect existing cohort; never overwrite or retry')
        source=gen.git('rev-parse','HEAD').decode().strip()
        seed_ref=dict(commit=source,path=str(assets.RESULT.relative_to(P)),sha256=gen.common.digest(assets.RESULT))
        base=gen.committed(assets.RESULT)
        guards.require(base['passed'] and base['complete'] and base['ownership_closed'] and not base['active'], 'closed seed required')
        completed=[]; qualification=None
    else:
        previous=gen.committed(RESULT)
        guards.require(previous['passed'] and previous['complete'] and previous['ownership_closed'] and
            not previous['active'] and not previous['fresh_faults'] and previous['plan']==comparison_plan() and
            ((cohort==0 and previous['stage']=='check') or (cohort>0 and previous['stage']=='clean' and previous['cohort']==cohort-1)),
            'close matching check/preceding clean process first')
        source=previous['source_commit'];seed_ref=previous['seed_reference']
        data=gen.git('show',seed_ref['commit']+':'+seed_ref['path'])
        guards.require(hashlib.sha256(data).hexdigest()==seed_ref['sha256'], 'seed identity changed')
        base=json.loads(data);completed=previous['processes'];qualification=previous['check_evidence']
        guards.require(qualification['passed'] and len(completed)==cohort, 'missing long qualification/cohort history')
        guards.require(full['binary']==previous['binary'] and count==previous['original_native_inputs_recovered'] and
            gen.common.digest(BUILD_RECORD)==previous['source_build_record']['sha256'] and
            gen.common.digest(COMPONENT)==previous['component_result_sha256'] and
            gen.common.digest(QUALIFIED)==previous['tiny_model_qualification']['sha256'] and
            gen.common.digest(gen.SETTINGS)==previous['settings_sha256'], 'frozen cohort qualification/build changed')
    arm=1 if check else ORDER[cohort]
    root=B/('decode-arena-generation-v1-check' if check else f'decode-arena-generation-v1-{cohort}')
    guards.require(not root.exists(), 'owned root exists; no automatic retry')
    pins=gen.benchmark_pins(source);settings=gen.committed(gen.SETTINGS)
    config=copy.deepcopy(base['config']);flags=config['args']
    for name,value in (('--pool-workers','5'),('--pcie-frac','0.25'),('--expert-profile',str(root/'expert-profile.bin'))):
        flags[flags.index(name)+1]=value
    config.update(log=str(root/'engine.stderr'),env={})
    old_root=base['environment']['STRATA_MEASUREMENT_ROOT']
    env={k:v.replace(old_root,str(root)) for k,v in base['environment'].items()}
    guards.require(not any(k=='STRATA_TRACE' or k.startswith('UR_LOG_') or k=='UR_ENABLE_LAYERS' for k in env), 'seed has observers')
    env['STRATA_SYCL_VERIFY_ARENA_COPY']=str(arm)
    for name in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','NEOReadDebugKeys','EnableDirectSubmission','STRATA_SYCL_PREFILL_SAME_QUEUE'):
        guards.require(env[name]==settings['environment'][name], 'qualified safety setting changed')
    if check:
        env.update(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',STRATA_TRACE='1')
        env.update({k:v for k,v in settings['environment'].items() if k.startswith('UR_LOG_') and k!='UR_LOG_TRACING'})
    model=assets.model_pins(config,None if check else previous['model_assets'],full=check)
    machine=assets.machine_identity()
    if not check:guards.require(machine==previous['machine_identity'], 'cohort machine/runtime changed')
    root.mkdir(mode=0o700)
    for name in ('cache','tmp'):(root/name).mkdir()
    shutil.copyfile(P/'data/expert-profile.bin',root/'expert-profile.bin')
    guards.require(gen.common.digest(root/'expert-profile.bin')==base['profile_sha256'], 'profile changed')
    (root/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    command=['/bin/sh',str(P/'bench.sh'),'--config',str(root/'config.json'),'--port',str(port),
        '--long-generation-only','--think','--runs','1','--workload-id',gen.WORKLOAD_ID,
        '--fail-fast-engine-error','--ready-timeout','600','--out',str(root/'bench'),
        '--long-generation-workload','regular-code']
    record=dict(mode='arena-generation-check' if check else 'arena-generation-screen',stage=stage,cohort=cohort,arm=arm,
        source_commit=source,build_source_commit=full['source_commit'],binary=full['binary'],
        binary_origin_source_commit=full['binary_origin_source_commit'],seed_reference=seed_ref,
        source_build_record=dict(path=str(BUILD_RECORD),sha256=gen.common.digest(BUILD_RECORD)),
        original_native_inputs_recovered=count,component_result_sha256=gen.common.digest(COMPONENT),
        tiny_model_qualification=dict(commit=gen.git('rev-parse','HEAD').decode().strip(),path=str(QUALIFIED.relative_to(P)),sha256=gen.common.digest(QUALIFIED)),
        runtime_asset_delta=base['runtime_asset_delta'],root=str(root),environment=env,config=config,command=command,
        model_assets=model,model_assets_fingerprint=assets.fingerprint(model),machine_identity=machine,
        tokenizer_assets={str(Path(config['tokenizer'])/n):gen.common.digest(Path(config['tokenizer'])/n)
            for n in ('vocab.json','merges.txt','token_type.json','chat_template.jinja')},
        profile_sha256=base['profile_sha256'],config_sha256=gen.common.digest(root/'config.json'),
        settings_sha256=gen.common.digest(gen.SETTINGS),source_count=len(pins),source_aggregate_sha256=assets.fingerprint(pins),
        plan=comparison_plan(),processes=completed,check_evidence=qualification,
        complete=False,passed=False,active=False,model_quality_qualified=False,adoption_qualified=False,
        clean_speed_measured=False,budgets=dict(wall=1800,cpu=54000,rss_cap=96<<30,text_cap=64<<20,file_cap=64<<20,total_cap=64<<20),
        consumer='Determine whether25% ordinary-Arena copy sharing warrants weak-MTP and agent-session validation; retire normal traces after useful samples/counters are committed.')
    gen.runtime_assets_unchanged(record)
    (root/'plan.json').write_text(json.dumps(record,indent=2)+'\n');RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print('Prepared '+stage+'; commit the exact generated plan before run.')


def run():
    record=gen.committed(RESULT);root=Path(record['root'])
    guards.require(not record['complete'] and record==json.loads((root/'plan.json').read_text()) and
        not (root/'record.json').exists(), 'fresh frozen plan required; no automatic restart')
    full,count=current_build()
    guards.require(full['binary']==record['binary'] and count==record['original_native_inputs_recovered'] and
        gen.common.digest(BUILD_RECORD)==record['source_build_record']['sha256'] and
        gen.common.digest(COMPONENT)==record['component_result_sha256'], 'build/component proof changed')
    pins=gen.benchmark_pins(record['source_commit'])
    guards.require(len(pins)==record['source_count'] and assets.fingerprint(pins)==record['source_aggregate_sha256'], 'benchmark inputs changed')
    for path,expected in [(gen.SETTINGS,record['settings_sha256']),(root/'config.json',record['config_sha256']),
            (root/'expert-profile.bin',record['profile_sha256']),*record['tokenizer_assets'].items()]:
        guards.require(gen.common.digest(path)==expected,'input changed: '+str(path))
    gen.runtime_assets_unchanged(record)
    guards.require(assets.model_pins(record['config'],record['model_assets'])==record['model_assets'] and
        assets.machine_identity()==record['machine_identity'], 'model/platform changed')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True)
    cursor=next(x.split(': ',1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '));owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        (root/'record.json').write_text(json.dumps(dict(record,owned_commands=owner.commands if owner else []),indent=2)+'\n')
    owner=supervision.Owner(root/'owned',persist)
    try:
        entry,_,_=owner.run('bench',record['command'],record['environment'],P,**record['budgets'])
        record.update(complete=bool(supervision.closed(entry)),exit_code=entry.get('exit_code'))
        guards.require(record['complete'] and entry['exit_code']==0,'owned benchmark failed')
        path,=(root/'bench').glob('*/results.json');suite=json.loads(path.read_text())['results']['long_generation']
        rows=gen.collect_samples(suite);record['requests']=rows;gen.require_complete_generation(suite,rows,('regular-code',))
        guards.require(rows[0]['acceptance_rate']>=0.60,'strong-MTP regime not confirmed')
        journal,requests=gen.journal_evidence(root);resources.require_journal_binding(rows,requests)
        record['warmups']=gen.warmup_evidence(root,journal)
        native=json.loads((root/'engine-exit.json').read_text())
        guards.require(native['exit_code']==0 and native['boot_id']==boot and native['binary']==record['binary'] and
            native['args']==['--serve',*record['config']['args']] and 'resource_collector' not in native,'native identity/closure differs')
        log=(root/'engine.stderr').read_text(errors='replace');assets.verify_pool(log,5)
        record['copy_evidence']=copy_evidence(log,rows[0],record['arm'])
        guards.require('strata decode cpu: ' not in log and 'strata decode timing: ' not in log,'unexpected observers')
        current_build();gen.runtime_assets_unchanged(record)
        guards.require(gen.benchmark_pins(record['source_commit'])==pins and assets.machine_identity()==record['machine_identity'] and
            assets.model_pins(record['config'],record['model_assets'],full=record['stage']=='clean' and record['cohort']==5)==record['model_assets'],
            'model/controller/platform changed during run')
        record.update(passed=True,native_identity=native['native'],clean_speed_measured=record['stage']=='clean')
    except BaseException as error:
        record['error']=str(error)
        try:
            recovered_journal,recovered_rows=gen.journal_evidence(root)
            record['request_journal']=recovered_journal
            if 'requests' not in record:
                record['requests']=[dict(row,controller_eligible=gen.sample_eligible(row)) for row in recovered_rows]
        except Exception as journal_error:
            record['journal_recovery_error']=str(journal_error)
        for path in (root/'engine.stderr',root/'owned/bench.stderr'):
            if path.exists():record['error_excerpt']=path.read_text(errors='replace')[-4096:]
    finally:
        record['ownership_closed']=owner.active is None and all(supervision.closed(c) for c in owner.commands)
        after=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True)
        record['fresh_faults']=[x for x in after.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',x)]
        record['boot_unchanged']=boot==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        try:record['driver_after']=guards.driver_guard()
        except Exception as error:record.update(driver_guard_error=str(error),passed=False)
        record['passed']=bool(record['passed'] and record['ownership_closed'] and record['boot_unchanged'] and not record['fresh_faults'])
        sample=dict(arm=record['arm'],passed=record['passed'],ownership_closed=record['ownership_closed'],
            source_commit=record['source_commit'],binary_sha256=record['binary']['sha256'],
            request=(record.get('requests') or [{}])[0],copy_evidence=record.get('copy_evidence'),fresh_faults=record['fresh_faults'])
        if record['stage']=='check':record['check_evidence']=sample
        else:
            record['processes'].append(sample)
            if record['passed'] and record['cohort']==5:
                try:record['screen_result']=screen(record['processes'])
                except Exception as error:record.update(screen_error=str(error),passed=False)
        persist();RESULT.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record.get(k) for k in ('stage','cohort','arm','complete','passed','ownership_closed','error')}))
    return 0 if record['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('stage',nargs='?',choices=('check','clean'),default='check');parser.add_argument('cohort',nargs='?',type=int,choices=range(6),default=0)
    args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        return prepare(args.stage,args.cohort) if args.action=='prepare' else run()


if __name__=='__main__':sys.exit(main())
