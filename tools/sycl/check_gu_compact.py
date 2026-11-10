#!/usr/bin/env python3
"""Qualify and measure compact GU fusion using exact current native libraries.

Main-authored; repository LICENSE applies. No engine dispatch changes.
"""
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
sys.dont_write_bytecode = True
import build_gu_compact as build

P, B, RESULT = build.P, build.B, build.RESULT
ROOT, KEY = B/'gu-compact-device-v3', 'gu_compact_device_v3'
THIS = 'tools/sycl/check_gu_compact.py'
STAGES = ('smoke', 'numerics', 'perf0', 'perf1', 'perf2')
ROWS = (1, 16, 32, 64, 65, 96, 127, 128, 129, 256, 640, 1280)
original, component = build.device.original, build.device.component
require, sha = build.require, build.sha


def qualified():
    built = json.loads((build.ROOT/'record.json').read_text())
    native = build.current(built, extra_observers={THIS})
    require(built['stage'] == 'built' and built['passed'] and built['complete']
        and built['ownership_closed'] and not built['active'] and not built['GPU_executed'],
        'committed closed component build required')
    headers = built['actual_headers']
    require(sha(headers['path']) == headers['sha256'], 'fixture header receipt changed')
    files = json.loads(Path(headers['path']).read_text())
    require(len(files) == headers['files'] and all(sha(p) == h for p,h in files.items()),
        'actual compiled fixture/header input changed')
    binary = built['binary']
    require(sha(binary['path']) == binary['sha256']
        and Path(binary['path']).stat().st_size == binary['bytes'], 'fixture ELF changed')
    basis = original.validation.committed(RESULT, (build.device.KEY,))
    require(basis['stage'] == 'closed' and basis['passed'] and basis['complete']
        and basis['ownership_closed'] and not basis['active']
        and basis['binaries'] == native['binaries'], 'same-build nine native gates required')
    return built, native, basis


def save(row):
    (ROOT/'record.json').write_text(json.dumps(row, indent=2)+'\n')
    public = json.loads(RESULT.read_text()); public[KEY] = row
    RESULT.write_text(json.dumps(public, indent=2)+'\n')


def verify(row):
    require(original.validation.committed(RESULT, (KEY,)) == row, 'commit exact preceding stage')
    raw = original.common.source.git(P, 'show', row['source_commit']+':'+THIS)
    require(hashlib.sha256(raw).hexdigest() == row['controller_sha256'] == sha(P/THIS),
        'committed controller changed')
    built, native, basis = qualified()
    require(built['binary'] == row['binary'] and native['observer_admission'] == row['native_admission']
        and original.common.signature(basis['environment']) == row['environment_sha256']
        and original.guards.driver_guard() == row['driver'] == basis['driver']
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'] == basis['boot'],
        'fixture/native/environment/platform admission changed')
    return basis


def parsed(text, stage):
    lines = text.splitlines()
    mode = 'bench' if stage.startswith('perf') else stage
    require(lines and lines[-1] == 'fixture_status=PASS mode='+mode+' profiling=0',
        'normal fixture completion marker required')
    records = [json.loads(line) for line in lines[:-1]]
    require(all(isinstance(v, dict) and v.get('kind') in ('correctness','calibration','sample')
        for v in records), 'unexpected fixture output')
    correct = [v for v in records if v['kind'] == 'correctness']
    expected = ([(3,0,0),(3,0,1)] if stage == 'smoke' else
        [(r,t,0) for t in (0,1) for r in (1,3,16,17,32,33,64,65,96,127,128,129,256,640,1280)]+[(17,t,1) for t in (0,1)]
        if stage == 'numerics' else [(r,0,0) for r in ROWS])
    require(len(correct) == len(expected) and
        sorted((v['rows'],v['tile'],v['edges']) for v in correct) == sorted(expected),
        'complete exact numerical coverage required')
    require(all(v['status'] == 'PASS' and v['BF16_capture_exact'] is True
        and v['guards'] is True and v['inputs_immutable'] is True and v['production_baseline'] is True
        and type(v['max_half_ulp']) is int and 0 <= v['max_half_ulp'] <= 1 for v in correct),
        'BF16 capture/FP16 bound/guard/input qualification failed')
    answer = dict(correctness_cases=len(correct),max_half_ulp=max(v['max_half_ulp'] for v in correct),
        BF16_capture_exact=True,guards=True,inputs_immutable=True)
    if not stage.startswith('perf'):
        require(len(records) == len(correct), 'timing output in correctness stage')
        return answer
    process = int(stage[-1])
    calibrations = [v for v in records if v['kind'] == 'calibration']
    samples = [v for v in records if v['kind'] == 'sample']
    require(len(calibrations) == len(ROWS) and [v['rows'] for v in calibrations] == list(ROWS)
        and len(samples) == len(ROWS)*8, 'twelve shapes/eight samples per shape required')
    valid_ms = lambda v, floor: type(v) in (int,float) and math.isfinite(v) and v >= floor
    shapes = []
    for ordinal,(rows,cal) in enumerate(zip(ROWS,calibrations)):
        require(type(cal['repeat']) is int and cal['repeat'] in tuple(2**n for n in range(6,19))
            and valid_ms(cal['separate_ms'],150) and valid_ms(cal['fused_ms'],150),
            'declared calibration floor/repeat budget failed')
        values = [v for v in samples if v['rows'] == rows]
        order = [(r,((r+process+ordinal)&1)^p) for r in range(4) for p in range(2)]
        require([(v['round'],v['arm']) for v in values] == order, 'matched rotated sample order changed')
        require(all(v['process'] == process and v['order'] == ((v['round']+process+ordinal)&1)
            and v['repeat'] == cal['repeat'] and valid_ms(v['batch_wall_ms'],100)
            and type(v['max_half_ulp']) is int and 0 <= v['max_half_ulp'] <= 1
            and v['status'] == 'PASS' for v in values), 'ineligible or unmatched speed sample')
        separate = [next(v['batch_wall_ms'] for v in values if v['round'] == r and v['arm'] == 0) for r in range(4)]
        fused = [next(v['batch_wall_ms'] for v in values if v['round'] == r and v['arm'] == 1) for r in range(4)]
        ratios = [b/a for a,b in zip(separate,fused)]
        shapes.append(dict(rows=rows,repeat=cal['repeat'],calibration_ms=[cal['separate_ms'],cal['fused_ms']],
            separate_batch_ms=separate,fused_batch_ms=fused,
            separate_median_us=1000*statistics.median(separate)/cal['repeat'],
            fused_median_us=1000*statistics.median(fused)/cal['repeat'],
            median_paired_fused_over_separate=statistics.median(ratios)))
    answer['shapes'] = shapes
    return answer


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh prospective plan only')
    built, native, basis = qualified()
    require(original.guards.driver_guard() == basis['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == basis['boot'], 'qualified platform changed')
    ROOT.mkdir(mode=0o700)
    row = dict(stage='prepared',source_commit=original.common.source.git(P,'rev-parse','HEAD').decode().strip(),
        controller_sha256=sha(P/THIS),fixture_build_key=build.KEY,binary=built['binary'],
        native_admission=native['observer_admission'],environment_basis=build.device.KEY,
        environment_sha256=original.common.signature(basis['environment']),boot=basis['boot'],driver=basis['driver'],
        plan=dict(order=list(STAGES),commands=['--smoke','--numerics','--bench 0','--bench 1','--bench 2'],
            correctness='Actual production auto baseline versus compact/wide fused adapter. Exact BF16 capture;independent integer-dot oracle for M1/3;<=1FP16ULP,NaN/saturation,guards and immutable inputs.32full numeric cases.',
            timing='Resident same FP16 X/W;baseline actual try_gu/f16_out16/native SwiGLU with auto(T<128compact,elsewide);candidate compact fused adapter plus common finish. Candidate production binding remains excluded. No profiling queue or GPU timing events. Warm8each;calibrate both >=150ms;64..262144 same repeats;4 rotated pairs/shape/process;all samples>=100ms. Three fresh processes. Reset/oracle/guards outside timing.',
            eligibility='A shape is a follow-up candidate only when all three independent-process median paired fused/separate ratios are <1. No blanket minimum gain. Retain all4matched pairs and process spread. This does not advance a model route.',
            safety='First smoke full UR trace and parameter validation;later parameter validation without API trace. Preserve copy-offload/direct-submission avoidance. Stop first failure;no retry/reset/foreign signalling.'),
        budget=dict(wall=240,cpu=720,rss_cap=4<<30,text_cap=64<<20,file_cap=64<<20,total_cap=80<<20),
        limits='Baseline includes actual production entry and finish;candidate uses existing adapter plus finish,without future production binding. Excludes GGUF dequant,gather,PCIe and model handoff. No model PP/TG/TTFT or default adoption claim.',
        retention='Commit useful numeric summary and individual speed pairs,then retire successful API/build logs. Keep fixture only for this decision and imminent production-dispatch comparison;old native proof holds remain separate.',
        results=[],passed=False,complete=False,active=False,ownership_closed=True,GPU_executed=False)
    save(row);print('Prepared compact-GU smoke/numerics/three-process timing;commit before execution.',flush=True)


def execute(stage):
    row = json.loads((ROOT/'record.json').read_text());basis = verify(row);frozen = copy.deepcopy(row)
    require(row['stage'] in ('prepared','in_progress') and len(row['results']) == STAGES.index(stage)
        and all(v['passed'] for v in row['results']), 'fresh next stage only;no retry')
    root = ROOT/stage;root.mkdir(mode=0o700)
    for name in ('tmp','cache'):(root/name).mkdir(mode=0o700)
    cursor = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],
        text=True,timeout=15).strip().split('-- cursor: ')[-1]
    require(cursor and '\n' not in cursor, 'kernel cursor required')
    scope = None
    def persist():
        (root/'execution.json').write_text(json.dumps(dict(active=bool(scope and scope.active),
            commands=scope.commands if scope else []),indent=2)+'\n')
    result = dict(mode=stage,passed=False,complete=False,GPU_executed=False)
    try:
        scope = original.owner.Owner(root/'owned',persist)
        trace = stage == 'smoke';probe = basis['native_validation_probe']
        env = dict(basis['environment'],STRATA_TEST_MAIN_GATES='0',STRATA_TRACE='1',
            LD_PRELOAD=probe['library']['path'],STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'],
            TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),STRATA_MEASUREMENT_ROOT=str(root),
            UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if trace else 'UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(root/'validation.log'))
        env.update(STRATA_SYCL_PREFILL_NATIVE16='bf16',STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm',
            STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto',STRATA_SYCL_PREFILL_GU_SWIGLU='large')
        if not trace:env.pop('UR_LOG_TRACING',None)
        args = ['--bench',stage[-1]] if stage.startswith('perf') else ['--'+stage]
        result['native_execution_attempted'] = True
        e,out,err = scope.run(stage,[row['binary']['path'],*args],env,P,**row['budget'])
        errors = err.read_text();proof = original.validation.preload_evidence(errors,e['pid'])
        component.diagnostics(errors,e['pid'])
        require(errors.count('[native16-route] engine=bf16 backend=vllm tile=auto(T<128:compact;else:wide) aot=bmg-g21-a0 isolated=1 experimental\n') == 1,
            'actual immutable production-auto configuration missing')
        require(original.owner.closed(e) and e['exit_code'] == 0
            and proof['loader'] == probe['loader']['path'], 'normal validated fixture closure required')
        log = root/'validation.log'
        require(not log.exists() or log.stat().st_size == 0, 'runtime parameter validation warning')
        if trace:require('urEnqueue' in errors, 'actual traced GPU submission missing')
        result.update(parsed(out.read_text(),stage),validation=proof,native_exit_code=0,
            stdout_sha256=sha(out),stderr_sha256=sha(err))
        verify(frozen)
        result.update(passed=True,complete=True,host_probe_accepted=True,GPU_executed=True)
    except BaseException as error:
        result.update(passed=False,error=repr(error))
    finally:
        if scope is not None:original.validation.finish(result,scope,cursor,row['boot'])
        else:result.update(ownership_closed=True,fresh_faults=None,boot_unchanged=False)
        result['passed'] = bool(result['passed'] and result['complete'] and result.get('driver_after') == row['driver'])
        row['results'].append(result)
        row.update(stage=('closed' if stage == STAGES[-1] else 'in_progress') if result['passed'] else 'failed',
            active=bool(scope and scope.active),ownership_closed=result['ownership_closed'],GPU_executed=True)
        row['complete'] = row['stage'] == 'closed';row['passed'] = row['complete'] and row['ownership_closed']
        if row['passed']:
            row['decision'] = dict(eligible_rows=[r for r in ROWS if all(next(s for s in v['shapes'] if s['rows'] == r)
                ['median_paired_fused_over_separate'] < 1 for v in row['results'][2:])],
                scope='Component follow-up eligibility only;production wrapper/model/independent decode gates remain open.')
        save(row)
    print(json.dumps({k:result.get(k) for k in ('mode','passed','complete','ownership_closed','fresh_faults','error')}),flush=True)
    require(result['passed'],result.get('error','compact-GU component gate failed'))


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare',*STAGES), 'prepare|'+'|'.join(STAGES))
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);original.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1])
