#!/usr/bin/env python3
"""Bounded same-build streamed-Q8 and unchanged resident-oracle checks; Main."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode=True
import check_full_context as gate

P,B,RESULT=gate.P,gate.B,gate.RESULT
ROOT=B/'budget-qsa-component-v1'
KEY='budget_qsa_components'
ORDER=('host','smoke','component','resident','attention-smoke','attention-component')
INPUTS=sorted(set([*gate.INPUTS,'tools/sycl/check_streamed_qsa.py']))
require=gate.native.guards.require
digest=gate.native.build.digest
write=gate.write


def inputs(revision):
    pins={}
    for name,raw in gate.native.build.blobs(P,revision,INPUTS):
        require(raw==(P/name).read_bytes(),'commit exact component controller: '+name)
        pins[name]=hashlib.sha256(raw).hexdigest()
    return pins


def compact(record):return {k:v for k,v in record.items() if k not in ('environments','commands')}


def prepare(container):
    require(KEY not in container and not ROOT.exists(),'fresh component gate only')
    engine=gate.engine()
    require(engine['targets']==list(gate.native.TARGETS),
        'matching streamed same-build suite required')
    probe=gate.validation.preload_qualification()
    revision=gate.native.build.git(P,'rev-parse','HEAD').decode().strip()
    pins=inputs(revision)
    for row in engine['fixtures'].values():require(digest(row['path'])==row['sha256'],'fixture changed')
    ROOT.mkdir(mode=0o700);envs={};cases={}
    for mode in ORDER:
        root=ROOT/mode;root.mkdir();(root/'tmp').mkdir();(root/'cache').mkdir()
        env=dict(engine['environment'],TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
            CUDA_MODULE_LOADING='EAGER',UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION')
        require(not env.get('LD_PRELOAD'),'unexpected preload')
        env.update(LD_PRELOAD=probe['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'])
        for name in ('UR_LOG_VALIDATION','UR_LOG_LOADER','UR_LOG_LEVEL_ZERO'):
            env[name]='level:warning;flush:warning;output:stderr'
        if mode in ('smoke','attention-smoke'):
            env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_TRACING='level:info;flush:info;output:stderr')
        else:env.pop('UR_LOG_TRACING',None)
        target=('prefill_expert_bank_plan_test' if mode=='host' else 'sycl_qsa_bf16_attention_test' if mode.startswith('attention-') else 'sycl_native16_qsa_test' if mode=='resident' else 'sycl_native16_qsa_streamed_test')
        fixture=engine['fixtures'][target]
        if mode=='host':
            env.pop('LD_PRELOAD');env.pop('STRATA_UR_PROBE_EXPECTED_LOADER')
        argument=mode.removeprefix('attention-')
        cases[mode]=dict(command=[fixture['path']]+([] if mode in ('host','resident') else ['--'+argument]),
            fixture=fixture,environment_sha256=hashlib.sha256(json.dumps(env,sort_keys=True).encode()).hexdigest())
        envs[mode]=env
    record=dict(stage='prepared',source_commit=revision,input_sha256=pins,
        native_source_commit=engine['source_commit'],binary=engine['binary'],cases=cases,environments=envs,
        preload=probe,outcomes={},active=False,passed=False,complete=False,ownership_closed=True,
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),driver_before=gate.native.guards.driver_guard(),
        budgets=dict(wall_s=180,cpu_s=360,rss_bytes=6<<30,text_bytes=16<<20,file_bytes=16<<20,total_bytes=512<<20),
        criteria='First CPU budget fixture must keep the short prefix,permit full262143/262144 host tails only after removing optional prefix,and preserve invalid/overflow/headroom refusals. Commit each case before the next. Tiny full-API-traced smoke first:one case,14 pre-enqueue refusals,three append chunks including checked host unwind and host->stage refill/rewind. Then parameter-validated4case suite:partial/missing/invalid/nonidentity pages,8192 row ceiling,last3cells ending262144. All stage/host codes/scales exactly match separately qualified resident append;resident slots,guards,inputs/maps and drains pass. Then unchanged resident numerical oracle passes,followed by traced two-case and parameter-validated ten-case on-demand BF16 attention fixtures,including262144 cells. Same-child activation,normal owned exits,unchanged native/controller/runtime/boot/driver,no fresh fault. First failure stops;no retry or speed/model claim.',
        retention='Retain compact individual case/activation/guard/fault/source decisions. Retire successful traces,JIT and duplicate receipts after closure;original selected13 assets remain untouched. New engine+same-build fixtures are the only unselected generation for imminent capacity-model validation.',
        model_adoption_qualified=False,physical262144_model_qualified=False,decode_performance_qualified=False,
        ordinary_runtime_teardown_qualified=False)
    write(ROOT/'record.json',record);container[KEY]=compact(record);write(RESULT,container)
    print('Prepared host,streamed,resident and on-demand BF16 gates;commit before execution.',flush=True)


def admission(record,container):
    require(RESULT.read_bytes()==gate.native.build.git(P,'show','HEAD:'+str(RESULT.relative_to(P))) and
        container[KEY]==compact(record) and inputs(record['source_commit'])==record['input_sha256'],
        'commit exact component plan first')
    engine=gate.engine()
    require(engine['source_commit']==record['native_source_commit'] and engine['binary']==record['binary'] and
        gate.validation.preload_qualification()==record['preload'],'native/runtime identity changed')
    for mode,case in record['cases'].items():
        require(digest(case['fixture']['path'])==case['fixture']['sha256'] and
            hashlib.sha256(json.dumps(record['environments'][mode],sort_keys=True).encode()).hexdigest()==
                case['environment_sha256'],'fixture/effective environment changed')


def run(container,mode):
    record=json.loads((ROOT/'record.json').read_text());admission(record,container)
    order=ORDER;index=order.index(mode)
    require(mode not in record['outcomes'] and all(record['outcomes'].get(p,{}).get('passed') for p in order[:index])
        and record['stage']==('prepared' if index==0 else order[index-1]+'_closed'),'no retry or skipped gate')
    require(record['driver_before']==gate.native.guards.driver_guard() and
        record['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'driver/boot differs')
    text=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor=next(s.split(': ',1)[1] for s in text.splitlines() if s.startswith('-- cursor: '))
    original=json.loads((ROOT/'record.json').read_text());owner=None
    def persist():
        record['active']=bool(owner and owner.active is not None)
        write(ROOT/'record.json',dict(record,commands=owner.commands if owner else []))
    owner=gate.native.supervision.Owner(ROOT/mode/'owned',persist)
    row=dict(passed=False,complete=False,ownership_closed=False);record['outcomes'][mode]=row
    record['stage']=mode+'_running';persist()
    try:
        c=record['cases'][mode];b=record['budgets']
        entry,out,err=owner.run(mode,c['command'],record['environments'][mode],ROOT/mode,
            wall=b['wall_s'],cpu=b['cpu_s'],rss_cap=b['rss_bytes'],text_cap=b['text_bytes'],
            file_cap=b['file_bytes'],total_cap=b['total_bytes'])
        stdout,stderr=out.read_text(errors='replace'),err.read_text(errors='replace')
        row.update(exit_code=entry.get('exit_code'),stdout_sha256=digest(out),stderr_sha256=digest(err))
        require(gate.native.supervision.closed(entry) and entry['exit_code']==0,'fixture exit/ownership failed')
        row['GPU_work_submitted']=mode!='host'
        if mode!='host':
            row['activation']=gate.validation.preload_evidence(stderr,entry['pid'])
        boundary=re.search(r'^strata UR validation: .*phase=before-main[^\n]*\n',stderr,re.M)
        if mode!='host':
            require(boundary is not None and row['activation']['loader']==record['preload']['loader']['path'],
                'actual fixture first-init/provider proof missing')
        body=stderr[boundary.end():] if boundary else stderr
        require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain|cleanup failed',body),
            'unexpected post-canary runtime failure')
        if mode=='host':
            require('expert_bank_prefix_budget short_kept=PASS full262143_host_tail=PASS full262144_host_tail=PASS exact=PASS invalid=PASS' in stdout.splitlines() and
                stdout.splitlines()[-1]=='expert_bank_plan PASS GPU=UNRUN model=UNRUN physical262144_RESTORE=UNQUALIFIED' and not stderr, 'CPU budget oracle failed')
            row['prefix_budget_oracle']=True
        elif mode.startswith('attention-'):
            lines=stdout.splitlines();case_lines=[s for s in lines if s.startswith('case cells=')]
            require(len(case_lines)==(2 if mode=='attention-smoke' else 10) and
                all(s.endswith('guards=PASS inputs=PASS status=PASS') and 'oracle=BF16_operands_FP32_online' in s for s in case_lines) and
                lines[-1]=='fixture_status=PASS model_quality=UNRUN model_PP=UNRUN physical262144_model=UNRUN faults=EXTERNAL_REQUIRED', 'BF16 attention numerical/guard fixture failed')
            row['cases']=[dict(t.split('=',1) for t in s.split()[1:]) for s in case_lines]
            if mode=='attention-component':require(row['cases'][-1]['cells']=='262144','full BF16 attention cell extent missing')
            if mode=='attention-smoke':require('urEnqueueKernelLaunch' in body,'no actual BF16 attention kernel trace')
        elif mode=='resident':
            require(stdout.strip()=='QSA16 local component screen PASS; attention/model/physical262144/RESTORE/performance unqualified',
                'resident independent numerical oracle failed')
            row['resident_numerical_oracle']=True
        else:
            lines=stdout.splitlines();case_lines=lines[:-1]
            require(len(case_lines)==(1 if mode=='smoke' else 4) and
                lines[-1]=='QSA16 streamed publication PASS model=UNRUN physical262144_model_RESTORE_decode_performance=UNQUALIFIED' and
                all(s.startswith('QSA16_streamed_case ') and s.endswith(
                    'stage_host_reference=PASS resident_map=PASS guards=PASS inputs=PASS checked_drain=PASS') for s in case_lines),
                'streamed fixture case framing/guards failed')
            row['cases']=[dict(token.split('=',1) for token in s.split()[1:]) for s in case_lines]
            require(row['cases'][0]['rejects']=='14' and row['cases'][0]['chunks']=='3' and
                row['cases'][0]['unwind']=='1','missing pre-enqueue/checked-unwind/refill checks')
            if mode=='component':require(row['cases'][-1]['cells']=='262144' and
                row['cases'][-1]['pos0']=='262141' and row['cases'][-1]['tokens']=='3','final physical cell missing')
            if mode=='smoke':require('urEnqueueKernelLaunch' in body,'no actual GPU kernel trace')
        admission(original,container);row['passed']=True
    except BaseException as error:row['error']=repr(error)
    finally:
        row['ownership_closed']=owner.active is None and all(gate.native.supervision.closed(c) for c in owner.commands)
        try:
            text=subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
            row['fresh_faults']=[s for s in text.splitlines() if re.search(
                r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',s)]
            row['driver_after']=gate.native.guards.driver_guard()
            row['boot_unchanged']=record['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        except BaseException as error:row['final_probe_error']=repr(error)
        row['passed']=bool(row['passed'] and row['ownership_closed'] and row.get('fresh_faults')==[] and
            row.get('boot_unchanged') and row.get('driver_after')==record['driver_before'] and 'final_probe_error' not in row)
        row['complete']=row['passed'];record['ownership_closed']=row['ownership_closed']
        record['stage']=mode+'_closed' if row['passed'] else 'failed'
        record['passed']=record['complete']=bool(mode==ORDER[-1] and row['passed'])
        persist();container[KEY]=compact(record);write(RESULT,container)
    print(json.dumps({k:row.get(k) for k in ('passed','complete','exit_code','ownership_closed','fresh_faults','error','final_probe_error')}),flush=True)
    return 0 if row['passed'] else 1


if __name__=='__main__':
    require(sys.argv[1:] in ([mode] for mode in ('prepare',*ORDER)),'prepare|'+ '|'.join(ORDER))
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gate.native.guards.no_model_owners([])
        container=json.loads(RESULT.read_text())
        if sys.argv[1]=='prepare':prepare(container)
        else:sys.exit(run(container,sys.argv[1]))
