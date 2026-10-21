#!/usr/bin/env python3
"""Main-only, single-use device gates for prefill canonical restoration."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
import prefill_recovery as base
import cpu_census_engine as faults
from check_verifier_retirement import DELTA

gen,assets,guards,owner_lib=base.gen,base.assets,base.guards,base.supervision
q=base.history.q
P,B,A=gen.P,gen.B,gen.A
ROOT=B/'canonical-restore-chunk-v1'
RESULT=A/'canonical-restore-current.json'

def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def current(record,full_models=False):
    full,count=q.current_build()
    guards.require(full['source_commit']==record['native_source'] and
        full['binary']==record['binary'] and count==record['native_inputs_recovered'] and
        gen.common.digest(q.BUILD_RECORD)==record['build_sha256'] and
        gen.common.digest(q.COMPONENT)==record['component_sha256'] and
        assets.fingerprint(gen.benchmark_pins(record['source_commit']))==record['controller_fingerprint'] and
        gen.common.digest(gen.SETTINGS)==record['settings_sha256'] and
        assets.machine_identity()==record['machine'],'canonical restore input/build drift')
    setup=base.setup_reader.pilot()
    guards.require(assets.model_pins(setup['config'],record['models'],full=full_models)==record['models'],
        'canonical restore model assets changed')
    # Check the admitted assets directly; this is not a cache-borrow run and
    # must not be relabelled to pass a different launcher's mode allowlist.
    for name,item in gen.committed(gen.SETTINGS)['runtime_elfs'].items():
        if name in DELTA:guards.require(DELTA[name][0]==item['sha256'],'original runtime baseline changed')
        expected=DELTA[name][1]if name in DELTA else item['sha256']
        guards.require(gen.common.digest(name)==expected,'canonical runtime changed: '+name)
    return full

def prepare():
    guards.require(not ROOT.exists() and not RESULT.exists(),'fresh single-use canonical gate required')
    full,count=q.current_build();host=full['canonical_restore_host_contract']
    guards.require(host['passed'] and host['ownership_closed'] and host['cases']==8 and
        host['fixture_sha256']==full['fixtures']['canonical_restore_copy_host_test']['sha256'],
        'same-build canonical host contract required')
    setup=base.setup_reader.pilot();config=setup['config'];flags=config['args']
    source=gen.git('rev-parse','HEAD').decode().strip()
    parent=gen.committed(A/'prefill-recovery-current.json')
    record=dict(stage='prepared',source_commit=source,native_source=full['source_commit'],binary=full['binary'],
        native_inputs_recovered=count,build_sha256=gen.common.digest(q.BUILD_RECORD),
        component_sha256=gen.common.digest(q.COMPONENT),settings_sha256=gen.common.digest(gen.SETTINGS),
        controller_fingerprint=assets.fingerprint(gen.benchmark_pins(source)),machine=assets.machine_identity(),
        models=assets.model_pins(config,parent['models'],full=True),pack=flags[flags.index('--pack')+1],
        plan=dict(order=['small','tail','canonical-off','canonical-on'],
            scope='Two explicit whole/chunk helper cases at2MiB with full UR tracing,then64MiB+17 with validation. Then separate actual canonical loader/owner children with immutable flag0/1 and audit1. No model inference.',
            correctness='Exact helper payload/outer guards and1/2 actual copies. Actual canonical fixture checks captured payload,full per-row device/source reference,metadata,48 layer views,PLE bindings,publication and full restoration. Require1538035200B canonical,32284672B bank,82821120B persistent; audit actual1/23 copies. Loader descriptors are shared with the reference,not an independent metadata oracle.',
            stopping='Stop on first failed source,model,runtime,payload,route,owner,fault or budget check; preserve extracted output and never retry this root. Commit closed results before any model trial.',
            budgets=dict(wall_s=600,cpu_s=2400,rss_bytes=24<<30,text_bytes=64<<20,total_bytes=128<<20),
            limits='No performance/adoption gate. Low-memory helper and full-weight owner fixture do not reproduce near-full model VRAM,expert-cache handoff,sustained decode or physical262144/session/RESTORE. Those require separate prospective plans with current1256 decode layout fixed.',
            retention='Keep per-case source/route/inventory/payload/guard/owner/fault outcome. Retire normal full traces/cache/receipts after verified committed closure.'),result=None)
    current(record);ROOT.mkdir(mode=0o700);write(ROOT/'frozen.json',record);write(RESULT,record)

def validate(stdout,stderr,mode):
    if mode in ('small','tail'):
        n=2097152 if mode=='small' else 67108881
        expected=[f'canonical restore copy device: chunk64={arm} bytes={n} copies={2 if arm and mode=="tail" else 1} payload=1 guards=1'for arm in (0,1)]
        expected.append('canonical restore copy device: cases=2 passed=1 owners_retired=1')
        guards.require(stdout.splitlines()==expected,'helper payload/guard/copy-count output differs')
        return dict(cases=2,bytes=n,copies=[1,1 if mode=='small'else 2],payload_and_guards_match=True)
    match=re.fullmatch(r'COMPONENT_ONLY canonical=1538035200 bank=32284672 persistent=82821120 rows=(\d+); no model/runtime-quality claim\n',stdout)
    guards.require(match is not None and int(match[1])>48,'canonical fixture inventory/payload checks incomplete')
    expected=23 if mode=='canonical-on'else 1;chunk=67108864 if expected==23 else 0
    lines=[s for s in stderr.splitlines()if s.startswith('strata canonical restore:')]
    guards.require(lines==[f'strata canonical restore: phase=bank_full bytes=1538035200 copies={expected} chunk={chunk}'],
        'actual canonical restore dispatch/publication differs')
    return dict(canonical_bytes=1538035200,bank_bytes=32284672,persistent_bytes=82821120,
        rows=int(match[1]),copies=expected,chunk=chunk,fixture_payload_metadata_PLE_checks=True)

def execute():
    record=gen.committed(RESULT);full=current(record)
    guards.require(record['stage']=='prepared' and record['result']is None,'closed canonical gate; no retry')
    guards.require(record==json.loads((ROOT/'frozen.json').read_text()),'frozen canonical plan changed')
    root=ROOT/'device';guards.require(not root.exists(),'existing device attempt; inspect owner');root.mkdir()
    for name in ('tmp','cache'):(root/name).mkdir()
    boot,cursor=faults.start_fault_scope();row=dict(passed=False,complete=False,checks=[]);owner=None
    def persist():
        row['active']=bool(owner and owner.active is not None)
        write(root/'record.json',dict(row,commands=owner.commands if owner else []))
    owner=owner_lib.Owner(root/'owned',persist)
    try:
        for mode in record['plan']['order']:
            small=mode=='small';helper=mode in ('small','tail')
            fixture=full['fixtures']['sycl_canonical_restore_copy_test'if helper else'sycl_canonical_dense_bank_test']
            env=dict(gen.committed(gen.SETTINGS)['environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
                STRATA_SYCL_PREFILL_CANONICAL_CHUNK64='1'if mode=='canonical-on'else'0',
                STRATA_SYCL_PREFILL_CANONICAL_RESTORE_AUDIT='1')
            if not small:
                env={k:v for k,v in env.items()if not k.startswith('UR_LOG_')}
                env['UR_ENABLE_LAYERS']='UR_LAYER_PARAMETER_VALIDATION'
                for key in ('UR_LOG_VALIDATION','UR_LOG_LOADER','UR_LOG_LEVEL_ZERO'):env[key]='level:warning;flush:warning;output:stderr'
            command=[fixture['path'],*(['--small']if small else[])]if helper else[fixture['path'],record['pack'],*[n for n in record['models']if n.endswith('.gguf')]]
            item=dict(mode=mode,passed=False,fixture_sha256=fixture['sha256']);row['checks'].append(item)
            e,out,err=owner.run(mode,command,env,P,wall=600,cpu=2400,rss_cap=24<<30,text_cap=64<<20,file_cap=64<<20,total_cap=128<<20)
            text=out.read_text();diagnostic=err.read_text()
            item.update(output_excerpt=text[-2048:],exit_code=e.get('exit_code'),ownership_closed=owner_lib.closed(e),command=command)
            if not owner_lib.closed(e) or e.get('exit_code')!=0 or re.search(base.generation.FAILURE,diagnostic):
                item['stderr_excerpt']=diagnostic[-16384:]
            guards.require(owner_lib.closed(e) and e['exit_code']==0 and
                not re.search(base.generation.FAILURE,diagnostic),'canonical device child failed')
            item.update(validate(text,diagnostic,mode));item.update(passed=True,ownership_closed=True)
            if small:
                item['UR_trace_calls']=diagnostic.count('---> ur');guards.require(item['UR_trace_calls']>0,'missing small UR trace activation')
            current(record);persist()
        guards.require(row['checks'][2]['rows']==row['checks'][3]['rows'],'canonical owner inventory changed across arms')
        current(record,full_models=True);row.update(passed=True,complete=True)
    except BaseException as error:row['error']=str(error)
    finally:
        row['ownership_closed']=owner.active is None and all(owner_lib.closed(c)for c in owner.commands)
        row.update(fault_probe_successful=False,fresh_faults=None,boot_unchanged=False)
        try:row.update(faults.fault_evidence(cursor,boot))
        except Exception as error:row['fault_probe_error']=str(error)
        row['passed']=bool(row['passed'] and row['ownership_closed'] and row['fault_probe_successful'] and row['fresh_faults']==[] and row['boot_unchanged']);persist()
        record.update(stage='closed',result=row);write(RESULT,record)
    print(json.dumps({k:row.get(k)for k in ('passed','complete','ownership_closed','error')}));return 0 if row['passed']else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','device'));args=parser.parse_args()
    with (B/'owned-v0141-measurement.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);guards.no_model_owners([]);guards.driver_guard()
        return prepare()if args.action=='prepare'else execute()
if __name__=='__main__':sys.exit(main())
