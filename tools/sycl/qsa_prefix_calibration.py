#!/usr/bin/env python3
"""Owned native14 QSA component calibration; repository LICENSE applies.

Reuse the original compiled fixture, not current C++ source. This isolates the
ordered denominator versus subgroup reduction with identical cached Q/vec4 KV.
It measures a software path and requested work, never physical saturation.
"""
import copy
import fcntl
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import coherent_model_basis as basis
import measurement_owner as owner
import qsa_cached_subgroup as diagnostics
import ur_validation as validation

P, B, RESULT = basis.P, basis.B, basis.RESULT
ROOT, KEY = B/'qsa-prefix-calibration-v1', 'qsa_prefix_calibration_v1'
THIS = 'tools/sycl/qsa_prefix_calibration.py'
TARGET = 'sycl_qsa_bf16_prefix_cached_subgroup_bench'
NUMERIC = 'sycl_qsa_bf16_prefix_cached_subgroup_test'
PYTHON = basis.reference.model.PYTHON
require, sha, signature = basis.require, basis.sha, basis.signature
CASES = ((78453,8192,33,'gather'), (78453,8192,2051,'gather'),
         (78453,8192,2051,'clustered'))


def save(row):
    basis.reference.core.write(ROOT/'record.json', row)
    values = json.loads(RESULT.read_text()); values[KEY] = basis.compact(row)
    basis.reference.core.write(RESULT, values)


def parse(text, process):
    samples = []
    for line in text.splitlines():
        if not line.startswith('sample '): continue
        row = dict(re.findall(r'(\w+)=([^\s]+)', line))
        require(len(row) == 14 and row['status'] == row['exact_bits'] == row['guards'] == 'PASS',
                'incomplete or failed timed output')
        for key in ('process','cells','queries','cap','round','order','variant','compared'):
            row[key] = int(row[key])
        row['milliseconds'] = float(row['milliseconds'])
        require(row['process'] == process and row['variant'] in (0,1)
            and row['implementation'] == ('ordered_exp','cached_subgroup')[row['variant']]
            and row['compared'] == row['queries']*24*256
            and math.isfinite(row['milliseconds']) and row['milliseconds'] > 0,
            'actual timed arm/extent invalid')
        samples.append(row)
    require(len(samples) == 36 and text.count(' samples=12 full_bits=PASS guards=PASS inputs=PASS status=PASS') == 3
        and 'fixture_status=PASS model_quality=UNRUN model_PP=UNRUN physical262144_model=UNRUN' in text,
        'all three cases/36 timings/immutable inputs required')
    points = []
    for case in CASES:
        rows = [s for s in samples if tuple(s[k] for k in ('cells','queries','cap','pattern')) == case]
        expected = [(r,o,(process+r+o)%2) for r in range(6) for o in range(2)]
        require([(s['round'],s['order'],s['variant']) for s in rows] == expected,
            'case missing, duplicated or pair order changed')
        floor = .5 if case[2] == 33 else 10.
        require(all(s['milliseconds'] >= floor for s in rows), 'timing batch below declared floor')
        medians = [statistics.median(s['milliseconds'] for s in rows if s['variant']==arm)
                   for arm in (0,1)]
        # Issued matrix operations include padded heads and the final32-cell
        # tile. Requested read upper bound excludes invalid cells/page holes;
        # caches, coalescing and packing make this unlike physical bus traffic.
        tiles = (case[2]+31)//32
        points.append(dict(case=list(case), samples=rows, medians_ms=medians,
            candidate_over_reference=medians[1]/medians[0], floor_ms=floor,
            issued_matrix_FLOPs=case[1]*2*tiles*2*16*32*256*2,
            requested_KV_read_upper_bytes=case[1]*case[2]*2*2*256*2,
            interval='admission + prefix refresh + attention + checked drain',
            physical_bandwidth_or_saturation=False))
    require(sum(len(p['samples']) for p in points)==len(samples), 'undeclared timed case')
    return points


def decide(processes):
    require(len(processes)==3 and [p['index'] for p in processes]==[0,1,2]
        and all(p['passed'] and p['ownership_closed'] and not p['fresh_faults'] for p in processes),
        'all three independent processes required')
    shapes = []
    for i, case in enumerate(CASES):
        ratios = [p['points'][i]['candidate_over_reference'] for p in processes]
        shapes.append(dict(case=list(case), process_time_ratios=ratios,
            geometric_mean_time_ratio=math.exp(statistics.mean(map(math.log,ratios))),
            repeated_gain=all(r<1 for r in ratios)))
    return dict(shapes=shapes, eligible_for_new_128K_plan=all(p['repeated_gain'] for p in shapes[1:]),
        unit='denominator reduction only; cached Q, vec4 K/V, tiles, storage, fixture and environment fixed',
        clean_model_speedup=False, hardware_saturation_known=False, inference_adopted=False,
        older_32K_64K_failed_model_gates_unchanged=True,
        limits='Synthetic component only. Retain small-cap losses separately. No extrapolated model PP, numerical/task/agent/TG/full-context/RESTORE or decode preservation qualification.')


def current(row):
    require(validation.committed(RESULT,(KEY,)) == basis.compact(row)
        and all(sha(P/p)==h for p,h in row['sources'].items()), 'commit/freeze complete probe inputs')
    native, proof, prior = basis.basis(row['original_native_proof'])
    require(row['binary']==native['binaries'][TARGET] and row['numeric']==native['binaries'][NUMERIC]
        and row['native_validation_probe']==native['native_validation_probe']
        and row['runtime_manifest']==prior['runtime_manifest'] and row['boot']==prior['boot']
        and row['driver']==prior['driver'] and row['environment_template']==prior['environment_template'],
        'original compiler/fixture/runtime basis changed')
    require(owner.artifact_bytes(ROOT)<=row['whole_bytes'], 'campaign output budget exceeded')
    return prior


def environment(row, root, trace=False):
    old = row['environment_template']['STRATA_MEASUREMENT_ROOT']
    env = {k:v.replace(old,str(root)) for k,v in row['environment_template'].items()}
    env.pop('STRATA_TRACE',None)
    env.update(STRATA_TEST_MAIN_GATES='0', LD_PRELOAD=row['native_validation_probe']['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=row['native_validation_probe']['loader']['path'],
        UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if trace else 'UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'))
    if trace: env['UR_LOG_TRACING']='level:info;flush:info;output:stderr'
    else: env.pop('UR_LOG_TRACING',None)
    require(all(env[k]==v for k,v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',
        NEOReadDebugKeys='1',EnableDirectSubmission='0',STRATA_SYCL_PREFILL_SAME_QUEUE='1',ZES_ENABLE_SYSMAN='0').items()),
        'qualified immutable safety settings changed')
    return env


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh plan only; inspect existing owner')
    native, proof, prior = basis.basis()
    revision=validation.git('rev-parse','HEAD').decode().strip()
    sources=basis.reference.core.pins(revision)
    sources.update({p:sha(P/p) for p in basis.POLICIES})
    require(THIS in sources and all(validation.git('show',revision+':'+p)==(P/p).read_bytes()
        for p in sources), 'commit all exact controller/helper/policy inputs first')
    ROOT.mkdir(mode=0o700)
    save(dict(stage='prepared',source_commit=revision,sources=sources,
        original_native_proof=proof,native_source_commit=basis.original.SOURCE,
        binary=native['binaries'][TARGET],numeric=native['binaries'][NUMERIC],
        native_validation_probe=native['native_validation_probe'],runtime_manifest=prior['runtime_manifest'],
        environment_template=copy.deepcopy(prior['environment_template']),boot=prior['boot'],driver=prior['driver'],
        stages=[],processes=[],active=False,passed=False,complete=False,ownership_closed=True,whole_bytes=256<<20,
        plan=dict(question='Calibrate the dominant native14 QSA prefix and its independent reduction alternative.',
            unit='Only ordered-exp ascending sum versus existing cached-subgroup sum; same cached Q,vec4 KV,score stride33,32-key tiles,16 padded heads,in-order queue,allocations,pack and readback checks. No bank/window/chunk/inference source/decode change.',
            order='Host refusal controls; traced numerical and benchmark smoke; full11-case numeric including physical262144 component; then three fresh component processes0,1,2. Three fixed cases78453live/8192queries:cap33gather,cap2051gather,cap2051clustered. Six alternating pairs per case; all108timings retained. No active researcher/reviewer during clean timings.',
            gate='Every actual arm/order/count/output/reference/guard/input invariant,quiet validation,normal0,closed/reaped owner,unchanged source/ELF/provider/boot/driver/no fresh fault. Per-invocation floor0.5ms cap33,10ms cap2051. Host clock-call median<1000ns and p99<5000ns before timing; interval includes checked production admission,pack and drain. Checks/uploads/readbacks outside timing.',
            performance='For each cap2051 pattern all3 process candidate/reference median ratios<1,without minimum gain; retain cap33 independently. This only allows drafting a NEW128Kcomponent-specific model plan,not launch/adoption. Old failed32/64Kcomparison stays failed.',
            stopping='First refusal,short timing batch,fault,error or budget stop closes failed without retry,hash refresh or altered criteria. No performance-based early stop.',
            retention='Keep source/config/ELF identities,108samples,three process ratios,correctness/fault/closure and requested-work bounds; retire successful raw after commit. Reuse same12-ELF generation; no new build.')))


def execute(mode):
    row=json.loads((ROOT/'record.json').read_text());current(row)
    expected=('host','smoke','numeric','0','1','2')
    require(row['stage']!='failed' and len(row['stages'])<len(expected)
        and mode==expected[len(row['stages'])], 'ordered fresh stage only; inspect current handle')
    if mode.isdigit():
        for path in Path('/proc').iterdir():
            if not path.name.isdigit(): continue
            try:
                args=(path/'cmdline').read_bytes().split(b'\0')
                require(not (b'--cd' in args and any(str(B/'research').encode() in a
                    or str(B/'review').encode() in a for a in args)),
                    'close owned research/reviewer processes before clean timing')
            except (FileNotFoundError,PermissionError,ProcessLookupError): pass
    root=ROOT/mode;root.mkdir();(root/'tmp').mkdir();(root/'cache').mkdir()
    scope=None
    def persist():
        basis.reference.core.write(root/'execution.json',dict(active=bool(scope and scope.active),
            commands=scope.commands if scope else []))
    scope=owner.Owner(root/'owned',persist)
    cursor=subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],
        text=True,timeout=15).strip().split('-- cursor: ')[-1]
    require(cursor and '\n' not in cursor,'kernel cursor required')
    result=dict(mode=mode,passed=False,complete=False,ownership_closed=False)
    row.update(active=True,ownership_closed=False);save(row)
    try:
        if mode=='host':
            argv=[PYTHON,'-B','-m','unittest','tools.test_qsa_prefix_calibration']
            jobs=[('controls',argv,False)]
        else:
            if mode=='smoke':jobs=[('numerical',[row['numeric']['path'],'--smoke'],True),('benchmark',[row['binary']['path'],'--smoke'],True)]
            elif mode=='numeric':jobs=[('numerical',[row['numeric']['path'],'--component'],False)]
            else:jobs=[('benchmark',[row['binary']['path'],'--component',mode],False)]
        if mode.isdigit():
            ticks=[]
            for _ in range(10000):
                a=time.clock_gettime_ns(time.CLOCK_MONOTONIC);b=time.clock_gettime_ns(time.CLOCK_MONOTONIC);ticks.append(b-a)
            result['host_clock_call_ns']=dict(median=statistics.median(ticks),p99=sorted(ticks)[9899],
                scope='Python wrapper upper proxy for CLOCK_MONOTONIC; not device event resolution')
            require(result['host_clock_call_ns']['median']<1000 and result['host_clock_call_ns']['p99']<5000,'timer overhead screen failed')
        checks=[]
        for name,argv,trace in jobs:
            env=environment(row,root,trace)
            if mode=='host':env.pop('LD_PRELOAD');env.pop('STRATA_UR_PROBE_EXPECTED_LOADER')
            entry,out,err=scope.run(name,argv,env,P,wall=360,cpu=900,rss_cap=4<<30,
                text_cap=64<<20,file_cap=64<<20,total_cap=96<<20)
            require(owner.closed(entry) and entry['exit_code']==0,'normal checked process closure required')
            text,errors=out.read_text(),err.read_text()
            if mode=='host':require('Ran 7 tests' in errors and errors.rstrip().endswith('OK'),'all host controls required')
            else:
                evidence=validation.preload_evidence(errors,entry['pid']);diagnostics.diagnostics(errors,entry['pid'])
                require(evidence['loader']==env['STRATA_UR_PROBE_EXPECTED_LOADER']
                    and (not (root/'validation.log').exists() or (root/'validation.log').stat().st_size==0), 'pre-main validation differs')
                require('fixture_status=PASS' in text and (not trace or 'urEnqueue' in errors),'actual complete GPU fixture required')
                if mode=='numeric':require(sum(s.startswith('case cells=') for s in text.splitlines())==11
                    and 'case cells=262144 live_cells=262144 ' in text,'full independent numerical cases missing')
                if mode.isdigit():result.update(index=int(mode),points=parse(text,int(mode)))
            checks.append(dict(name=name,pid=entry['pid'],exit_code=0,ownership_closed=True,
                validation=evidence if mode!='host' else None))
        result.update(passed=True,host_probe_accepted=True,checks=checks,GPU_executed=mode!='host')
    except BaseException as error:result.update(error=repr(error))
    finally:
        validation.finish(result,scope,cursor,row['boot'])
        result['complete']=bool(result['ownership_closed'] and 'error' not in result)
        result['passed']=bool(result['passed'] and result['complete'] and result['ownership_closed']
            and basis.original.coherent.build.guards.driver_guard()==row['driver'])
        row['stages'].append(result)
        if mode.isdigit():row['processes'].append(result)
        row.update(stage='failed' if not result['passed'] else ('closed' if mode=='2' else 'in_progress'),
            active=bool(scope.active),ownership_closed=result['ownership_closed'],complete=mode=='2' and result['passed'],
            passed=mode=='2' and result['passed'])
        if row['passed']:row['decision']=decide(row['processes'])
        save(row)
    print(json.dumps({k:result.get(k) for k in ('mode','passed','ownership_closed','fresh_faults','error')}),flush=True)
    require(result['passed'],result.get('error','component stage failed'))


if __name__=='__main__':
    require(sys.executable==PYTHON,'use pinned model Python')
    require(len(sys.argv)==2 and sys.argv[1] in ('prepare','host','smoke','numeric','0','1','2'),'explicit phase required')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);basis.original.coherent.build.guards.no_model_owners([])
        prepare() if sys.argv[1]=='prepare' else execute(sys.argv[1])
