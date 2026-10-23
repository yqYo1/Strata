#!/usr/bin/env python3
"""Main-owned parser/first-use replay gates; repository LICENSE applies.

Only deterministic synthetic fixtures run here. Actual captured pairs require
their own committed provenance, host-check and full geometry qualification.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode=True
import qsa_order_replay as replay
import build_engine as build
import generation_engine as gen
import measurement_guards as guards
import measurement_owner as supervision
import ur_validation as validation
from measure_down_scatter import postchecks

P,B,A=gen.P,gen.B,gen.A
SCOPE='qsa_actual_mask_query_permutation_replay_audit_20261023'
TEST='tests/cpu/qsa_order_input.py'


def paths(mode):
    return replay.ROOT/(mode+'-v1'),A/('qsa-order-'+mode+'-current.json')


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def native():
    record=gen.committed(replay.RESULT)
    guards.require(record['passed'] and record['complete'] and record['ownership_closed'] and
        not record['active'] and record['stage']=='closed','closed replay build required')
    replay.diagnostic_admission()
    guards.require(replay.inputs(record['source_commit'])==record['inputs'],'original replay build inputs changed')
    for p,h in record['archives'].items():guards.require(build.digest(p)==h,'production archive changed')
    guards.require(build.digest(record['binary']['path'])==record['binary']['sha256'],'replay binary changed')
    return record


def current(record):
    built=native()
    guards.require(built['binary']==record['binary'] and built['source_commit']==record['native_source_commit'],
        'native replay lineage changed')
    guards.require(gen.benchmark_pins(record['controller_source_commit'])==record['controller_inputs'],
        'qualification controller changed')
    raw=gen.git('show',record['controller_source_commit']+':'+TEST)
    guards.require(raw==(P/TEST).read_bytes() and hashlib.sha256(raw).hexdigest()==record['host_test_sha256'],
        'host fixture source changed')
    if record['mode']=='smoke':
        host=gen.committed(paths('host')[1])
        guards.require(host['passed'] and host['complete'] and host['ownership_closed'] and not host['active'] and
            host['binary']==record['binary'] and len(host['samples'])==14,'closed parser tests required')
        scope=gen.committed(A/'research-index.json')[SCOPE]
        guards.require(scope.get('synthetic_smoke_static_eligible') is True and scope['synthetic_review']==record['review'],
            'synthetic-only smoke source review required')
        for item in record['review'].values():guards.require(build.digest(item['path'])==item['sha256'],'review changed')
        guards.require(replay.runtime_lineage()==record['runtime_lineage'],'runtime lineage changed')


def prepare(mode):
    root,result=paths(mode);guards.require(not root.exists() and not result.exists(),'fresh single-use qualification only')
    built=native();revision=gen.git('rev-parse','HEAD').decode().strip();root.mkdir(mode=0o700)
    for name in ('cache','tmp'):(root/name).mkdir()
    env=dict(built['environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'))
    record=dict(stage='prepared',mode=mode,controller_source_commit=revision,
        controller_inputs=gen.benchmark_pins(revision),host_test_sha256=build.digest(P/TEST),
        native_source_commit=built['source_commit'],binary=built['binary'],environment=env,
        active=False,ownership_closed=True,passed=False,complete=False,GPU_executed=False,
        model_adoption_qualified=False,actual_mask_replay_qualified=False,
        budgets=dict(wall=120,cpu=240,rss_cap=4<<30,text_cap=24<<20,file_cap=128<<20,total_cap=192<<20))
    if mode=='host':
        record['command']=['/usr/bin/python3',str(P/TEST),built['binary']['path'],built['binary']['sha256'],str(root/'fixture')]
        record['retention']='Keep14case statuses and source/recipe. Generated synthetic paired files have no next raw-byte consumer;retire after result commit.'
    else:
        scope=gen.committed(A/'research-index.json')[SCOPE]
        record.update(review=scope['synthetic_review'],runtime_lineage=replay.runtime_lineage(),command=[built['binary']['path'],'--smoke'])
        probe=record['runtime_lineage']['native_validation_probe']
        env.update(LD_PRELOAD=probe['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
            UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',STRATA_TRACE='1',
            UR_LOG_TRACING='level:info;flush:info;output:stderr',UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
            UR_LOG_LOADER='level:warning;flush:warning;output:stderr',UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
        record['scope']='No capture files:committed smoke() synthesizes97cells,three rows,widths0/33/17,cap33. Exact unpermuted output/perturbation/guards/immutability;no clean timing/model/capture qualification.'
        record['retention']='Keep compact checks. Successful full tracing is disposable after checked source/runtime/native exit and no fresh fault.'
    current(record);write(root/'plan.json',record);write(result,record)


def run(mode):
    root,result=paths(mode);record=gen.committed(result)
    guards.require(record==json.loads((root/'plan.json').read_text()) and not (root/'record.json').exists(),'fresh committed plan')
    current(record);boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    journal=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(x.split(': ',1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '));owner=None
    def persist():write(root/'record.json',dict(record,active=bool(owner and owner.active),commands=owner.commands if owner else []))
    owner=supervision.Owner(root/'owned',persist);persist()
    try:
        entry,out,err=owner.run(mode,record['command'],record['environment'],P,**record['budgets'])
        record['GPU_executed']=mode=='smoke'
        guards.require(supervision.closed(entry) and entry['exit_code']==0,'qualification did not close normally')
        text=out.read_text();trace=err.read_text()
        if mode=='host':
            tests=json.loads(text);guards.require(tests['passed'] and tests['GPU_executed'] is False and len(tests['samples'])==14 and
                all(s['passed'] for s in tests['samples']) and not trace,'parser tests incomplete')
            record.update(samples=tests['samples'],generated_fixture=dict(bytes=tests['generated_bytes'],files=tests['files']))
        else:
            require_case=r'^case layer=0 first=94 cells=97 queries=3 cap=33 cpu_order_ns=\d+ output_bits=PASS perturbation=PASS guards=PASS immutable=PASS$'
            guards.require(len(re.findall(require_case,text,re.M))==1 and
                text.count('fixture=PASS model=UNRUN physical_counters=UNMEASURED cpu_order=EXCLUDED')==1,'exact synthetic checks missing')
            record['activation']=validation.preload_evidence(trace,entry['pid'])
            body=trace.split(f'strata UR validation: pid={entry["pid"]} phase=before-main',1)[1]
            guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|Strata SYCL cleanup:',body),'runtime error')
            record['checks']=dict(original_row_output_bits=True,V_perturbation=True,guards=True,immutable=True,mixed_widths=[0,33,17])
        current(record);record.update(passed=True,complete=True)
    except BaseException as error:record.update(error=repr(error),passed=False,complete=False)
    finally:
        record.update(stage='closed',active=False,ownership_closed=owner.active is None and all(supervision.closed(c) for c in owner.commands))
        postchecks(record,boot,cursor);persist();write(result,record)
    print(json.dumps({k:record.get(k) for k in ('passed','complete','ownership_closed','fresh_faults','error')}),flush=True)
    return 0 if record['passed'] else 1


if __name__=='__main__':
    guards.require(len(sys.argv)==2 and sys.argv[1] in ('prepare-host','host','prepare-smoke','smoke'),'prepare-host|host|prepare-smoke|smoke')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        sys.exit(prepare(sys.argv[1][8:]) if sys.argv[1].startswith('prepare-') else run(sys.argv[1]))
