#!/usr/bin/env python3
"""Qualify the production-linked Q6 head before a full-model trial.

Main-authored; repository LICENSE. Adapt our closed native supervision from484d
to canonical/sole-packed entries, with fresh source/runtime checks and processes.
No old executable or private standalone source tree is a dependency.
"""
import ast
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import select
import subprocess
import statistics
import sys

sys.dont_write_bytecode = True
import benchmark_runtime as python_runtime
import build_q6_head_model as build
import measurement_guards as guards
import measurement_owner as owner
import ur_validation as validation
from q6_head_owner import parse_native, finish_gpu_probe
from run_q6_columns import validation_log
from check_verify_window_phase import SUBMISSION, PHASE

P, B, RESULT = build.P, build.B, build.RESULT
ROOT, KEY = B / 'q6-head-packed-device-v1', 'q6_head_packed_device_v1'
require, sha, write = guards.require, build.digest, build.write
ORIGINAL = '484d403e60c49361422f00a16dfdd774e4fe90d5'
ORIGINAL_PATH = 'tools/sycl/q6_head_owner.py'
PRIOR_DEVICE = '0e6c9385e87a42c56253fafbc5aa682fc7e2cd1e'
RUNTIME = B / 'decode-iq3-audit-speed-v2/runtime-inputs.json'
RUNTIME_SHA = '43de9b527585addcc6da1f8fdc6f1e8266e2d982ba4c752d909578ed8c7bde21'


def archived_gate():
    # Supervision is authored here from our archived484d owner,with explicit
    # 0/1/2format selection. The actual shared fault finalizer stays identical.
    raw=build.source.git(P,'show',ORIGINAL+':'+ORIGINAL_PATH)
    return run_gpu,hashlib.sha256(raw).hexdigest()


def model_assets():
    root=Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S')
    paths=sorted(root.glob('*.gguf'))
    require(len(paths)==2 and all(p.is_file() and p.stat().st_size>1<<30 for p in paths),
            'two existing real model shards required')
    return [dict(path=str(p),realpath=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]


def run_gpu(row,scope,mode,outcome,order=None,label=None):
    packed={'reference':0,'canonical':1,'sole-packed':2}[mode]
    label=label or mode
    require(order is None or mode=='sole-packed' and order in ('rc','cr'),'explicit timing order/format')
    require(model_assets()==row['assets'],'full pre-exec real-shard identities')
    env=dict(row['environment'],STRATA_SYCL_DECODE_Q6_DPAS_HEAD=str(packed),STRATA_Q6_HEAD_OWNER_GATE='1',
        LD_PRELOAD=row['native_validation_probe']['library']['path'],
        STRATA_UR_PROBE_EXPECTED_LOADER=row['native_validation_probe']['loader']['path'],
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,'+str(ROOT/(label+'-validation.log')))
    if order is not None:
        env['UR_ENABLE_LAYERS']='UR_LAYER_PARAMETER_VALIDATION';env.pop('UR_LOG_TRACING',None)
    cursor=next(v.split(': ',1)[1] for v in subprocess.check_output(
        ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15).splitlines()
        if v.startswith('-- cursor: '))
    outcome['native_execution_attempted']=True
    try:
        def interact(io,command):
            keys=('LD_PRELOAD','STRATA_UR_PROBE_EXPECTED_LOADER','UR_ADAPTERS_FORCE_LOAD','ZE_ENABLE_ALT_DRIVERS')
            all_required={str(Path(env[k]).resolve()) for k in keys}
            for phase,tag in enumerate(('STRATA_Q6_HEAD_OWNER_GATE ','STRATA_Q6_HEAD_OWNER_DEVICE_GATE ')):
                while True:
                    line=io.line(120)
                    if line.startswith(tag):break
                    require(not line.startswith('STRATA_Q6_HEAD_OWNER_'),'native gate order')
                identity=owner.identity(command['pid'])
                require(identity and identity['pid']==identity['sid']==command['pid']
                    and all(identity[k]==command['direct_identity'][k] for k in ('pid','start_ticks','sid'))
                    and line==tag+'pid='+str(command['pid']),'actual native owner at both gates')
                require(str(Path('/proc',str(command['pid']),'exe').resolve())==row['binary']['path'],'actual native ELF')
                actual=dict(v.split('=',1) for v in Path('/proc',str(command['pid']),'environ').read_text().split('\0') if '=' in v)
                mapped=set(python_runtime.code_maps(command['pid']))
                required=all_required if phase else {str(Path(env[k]).resolve()) for k in keys[:2]}
                missing=sorted(required-mapped);extra=sorted(mapped-(set(row['runtime_pins'])|{row['binary']['path']}))
                outcome.setdefault('provider_admission',[]).append(dict(phase='post-device-pre-fixture' if phase else 'pre-main',
                    mapped_files=len(mapped),mapped_path_sha256=hashlib.sha256('\0'.join(sorted(mapped)).encode()).hexdigest(),
                    missing_required=missing,unexpected_mapped_paths=extra,
                    not_yet_mapped_forced_providers=sorted(all_required-mapped),environment_immutable=actual==env))
                require(actual==env and not missing and not extra,'immutable environment and phase-appropriate providers')
                outcome['native_identity']=identity
                io.send(b'continue\n')
                if phase:row['GPU_executed']=True
            io.proc.stdin.close()
            while not io.exit_watch.poll(0) or not io.eof:
                io.poll()
                if select.select([io.proc.stdout],[],[],.02)[0]:io.read();io.pending.clear()
            io.exit(1)
        args=['--check'] if order is None else ['--timing',order]
        e,out,err=scope.run('native-'+label,[row['binary']['path'],*args,*[v['path'] for v in row['assets']]],
            env,ROOT,interaction=interact,**row['native_budget'])
        require(owner.closed(e) and e['exit_code']==0,'owned NativeHead process failed')
        text,diagnostic=out.read_text(),err.read_text()
        outcome['native_exit_code']=e['exit_code']
        outcome['validation']=validation.preload_evidence(diagnostic,e['pid'])
        require(outcome['validation']['loader']==env['STRATA_UR_PROBE_EXPECTED_LOADER'],'actual validation loader')
        log=ROOT/(label+'-validation.log')
        require(not log.exists() or log.stat().st_size==0,'parameter validation error')
        outcome['owner']=parse_native(text,packed)
        if order is not None:outcome['timing']=timing_samples(text,order)
        elif packed:
            require(('Q6DpasC4Canonical' if packed==1 else 'Q6DpasC4') in diagnostic and 'urEnqueueCommandBufferExp' in diagnostic,'traced selected head graph path')
            if packed==2:require('Q6DpasColumns' in diagnostic,'traced exact-width packed head path')
        require(model_assets()==row['assets'],'full post-run real-shard identities')
        outcome['assets_pre_post_match']=True
        outcome['passed']=True
    except BaseException as e:
        outcome.update(passed=False,error=repr(e))
    finally:
        finish_gpu_probe(outcome,scope,cursor,row['boot'])
        if not outcome['passed'] or outcome.get('driver_after')!=row['driver'] or 'error' in outcome:
            outcome['passed']=False
    require(outcome['passed'],outcome.get('error','native owner/fault/closure gate failed'))


def timing_samples(text,order):
    require(order in ('rc','cr'),'known timing order')
    lines=text.splitlines();timers=[v for v in lines if v.startswith('TIMER ')]
    require(len(timers)==1,'one timer control required')
    m=re.fullmatch(r'TIMER median_overhead_ns=(\d+) unit_ns=(\d+) floor_ns=(\d+)',timers[0])
    require(m is not None,'timer control schema')
    overhead,unit,floor=map(int,m.groups())
    require(unit>=max(1,overhead) and floor==max(10000000,300*unit),'timer/floor eligibility')
    samples=[]
    for line in lines:
        if not line.startswith('TIMING '):continue
        m=re.fullmatch(r'TIMING sample=(\d+) arm=([01]) replays=(\d+) elapsed_ns=(\d+) floor_ns=(\d+) scope=quantize-projection-graph-submit-complete',line)
        require(m is not None,'timing sample schema')
        sample,arm,replays,elapsed,sample_floor=map(int,m.groups())
        require(replays==256 and sample_floor==floor and elapsed>=floor,'timing sample below declared extent/floor')
        samples.append(dict(sample=sample,arm=arm,replays=replays,elapsed_ns=elapsed,ns_per_replay=elapsed/replays))
    sequence=(0,1) if order=='rc' else (1,0)
    require([(v['sample'],v['arm']) for v in samples]==[(i,a) for i in range(5) for a in sequence],
        'exact paired repetition order required')
    require([v for v in lines if v.startswith('TIMING_DONE ')]==['TIMING_DONE samples_per_arm=5 model_speed=unqualified'],
        'complete timing footer required')
    medians={str(a):statistics.median(v['ns_per_replay'] for v in samples if v['arm']==a) for a in (0,1)}
    return dict(order=order,samples=samples,median_ns=medians,candidate_over_reference=medians['1']/medians['0'],
        timer=dict(median_overhead_ns=overhead,unit_ns=unit,floor_ns=floor),timed_replays=2560,
        warmup_replays=6,numerical_replays=12,scope='C4 quantize+projection and graph submission/completion;parameter validation/retained-loader charged. No model throughput.')


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('controller', 'runtime_pins', 'commands')}
    out['controller'] = {k: v for k, v in row['controller'].items() if k != 'source_sha256'}
    out['commands'] = [dict(label=c['label'], exit_code=c.get('exit_code'),
        ownership_closed=owner.closed(c), error=c.get('error'), errors=c.get('errors', []))
        for c in row['commands']]
    return out


def save(row):
    write(ROOT / 'record.json', row)
    d = json.loads(RESULT.read_text()); d[KEY] = compact(row); write(RESULT, d)


def native():
    row = json.loads((build.ROOT / 'record.json').read_text())
    require(validation.committed(RESULT,(build.KEY,))==build.compact(row),'closed native proof changed')
    require(row['stage'] == 'built' and row['passed'] and row['complete'] and
        row['ownership_closed'] and row['actual_inputs']['uncovered'] == 0, 'closed full-engine build required')
    require(build.prerequisites()==row['prerequisites'],'fallback or exact shared fault finalizer changed')
    previous=row['admission_project'];current=build.source.pin(P,build.PREFIXES)
    before,after=previous['source_sha256'],current['source_sha256']
    require(before=={n:hashlib.sha256(raw).hexdigest() for n,raw in
        build.source.blobs(P,previous['revision'],before)},'original admission inputs not recoverable')
    allowed={'tools/sycl/check_q6_head_model.py','tools/sycl/q6_head_owner.py',
        'tools/sycl/test_q6_dpas_component.py','tools/sycl/q6_head_model_entry.py',
        'tools/sycl/q6_head_generation_speed.py'}
    require(set(before)==set(after) and {n for n in before if before[n]!=after[n]}<=allowed,
        'only predeclared observers may change;native/fixtures/recipe/readers stay exact')
    build.build.check_inputs(dict(row,project=current))
    for path,digest in (row.get('generated_inputs',{})|row.get('configured_files',{})).items():
        require(sha(path)==digest,'original generated/configured recipe changed')
    for binary in row['binaries'].values():
        require(sha(binary['path']) == binary['sha256'] and
            Path(binary['path']).stat().st_size == binary['bytes'], 'fresh same-build executable changed')
    actual = row['actual_inputs']
    require(sha(actual['path']) == actual['sha256'], 'actual-input inventory changed')
    census=json.loads(Path(actual['path']).read_text())
    require(all(sha(p)==h for field in ('compiled','generated','links') for p,h in census[field].items()),
        'original actual compiler/fixture/link bytes changed')
    return row


def verify(row):
    require(validation.committed(RESULT, (KEY,)) == compact(row), 'commit exact device plan/result first')
    pin = row['controller']
    require(build.signature(pin['source_sha256']) == pin['aggregate_sha256'] and
        {p: hashlib.sha256(raw).hexdigest() for p, raw in build.source.blobs(P,
            pin['revision'], pin['source_sha256'])} == pin['source_sha256'] and
        all(sha(P / p) == h for p, h in pin['source_sha256'].items()), 'device controller changed')
    n = native()
    require(sha(build.ROOT / 'record.json') == row['native_record']['sha256'] and
        n['binaries']['sycl_q6_head_owner_test'] == row['binary'], 'native build changed')
    require(sha(RUNTIME) == RUNTIME_SHA and json.loads(RUNTIME.read_text()) == row['runtime_pins'] and
        all(sha(p) == h for p, h in row['runtime_pins'].items()), 'qualified runtime changed')
    require(archived_gate()[1] == row['archived_gate']['sha256'], 'archived gate changed')
    for asset in row['native_validation_probe'].values():
        if isinstance(asset, dict) and 'path' in asset:
            require(sha(asset['path']) == asset['sha256'], 'native preload/provider changed')
    require(guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'boot/driver changed')
    guards.no_model_owners([a['path'] for a in row['assets']])


def prepare():
    require(not ROOT.exists() and KEY not in build.public(), 'fresh device plan only')
    n = native()
    old = json.loads(build.source.git(P, 'show', PRIOR_DEVICE + ':' + str(RESULT.relative_to(P))))[
        'q6_head_model_device_v1']
    require(old['passed'] and old['complete'] and old['ownership_closed'], 'closed owner prerequisite')
    require(sha(RUNTIME) == RUNTIME_SHA, 'fixed84ELF runtime manifest required')
    env = copy.deepcopy(old['environment'])
    env.update(TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
        STRATA_MEASUREMENT_ROOT=str(ROOT),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(ROOT / 'validation.log'))
    require(env['STRATA_PREFILL_RELEASE_HEADS'] == '1' and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1'
        and env['EnableDirectSubmission'] == '0' and env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == '0',
        'keep qualified immutable safety/head environment')
    row = dict(stage='prepared', source_commit=build.source.git(P, 'rev-parse', 'HEAD').decode().strip(),
        controller=build.source.pin(P, ['tools/sycl']), native_source_commit=n['project']['revision'],
        native_record=dict(path=str(build.ROOT / 'record.json'), sha256=sha(build.ROOT / 'record.json')),
        binary=n['binaries']['sycl_q6_head_owner_test'], binaries=n['binaries'],
        prior_environment_source=PRIOR_DEVICE + ':q6_head_model_device_v1',
        archived_gate=dict(commit=ORIGINAL, path=ORIGINAL_PATH, sha256=archived_gate()[1],
            functions=['model_assets', 'run_gpu']),
        environment=env, assets=old['assets'], runtime_manifest=dict(path=str(RUNTIME), sha256=RUNTIME_SHA),
        runtime_pins=json.loads(RUNTIME.read_text()), native_validation_probe=old['native_validation_probe'],
        boot=old['boot'], driver=old['driver'], native_budget=old['native_budget'],
        host_budget=old['host_budget'],
        native_qualification_reuse=dict(native_source_commit=n['native_source_commit'],
            native_build_key=build.KEY,admission_source_commit=n['admission_source_commit'],
            exact_four_ELFs_and_actual_inputs_and_recipe=True,GPU_result_inherited=False,
            changed_observers=['tools/sycl/check_q6_head_model.py','tools/sycl/q6_head_owner.py',
                'tools/sycl/test_q6_dpas_component.py'],
            scope='Controller-only new plan;native source6dd and all four artifacts stay exact. No old GPU/model result inherited.'),
        plan='47CPU parser/census/protocol controls plus seven fresh CPU-only empty-owner/invalid-switch processes,'
             'then submission10 and phase-cache16 traced checks,then fresh reference0/canonical1/sole-packed2'
             ' real-GGUF owner processes. Retain484d supervision,explicit new format bindings and unchanged'
             ' fault finalizer;keep two gates/native identity/finite budgets. Full shard hashes before/after'
             ' each head process. Four numerical cases,two retire/overwrite/restore cycles,12graph replays'
             ' per arm;mode2also checks eight partial-column cases and immutable inputs/packed weights.'
             ' Commit host closure before device. Stop first failure;no retries.',
        limits='Fresh production-linked fixture only. Not full model/production Verifier ownership,clean'
               ' PP/TTFT/TG,task quality,physical262144/RESTORE,ordinary teardown or default adoption.',
        retention='Current model-entry prerequisite. Retire normal traces/caches and obsolete standalone'
                  ' source/build roots after committed useful results and exact input recovery;preserve'
                  ' selected90f and552b/e1b phase guards. No new engine copy.',
        results=[], commands=[], passed=False, complete=False, active=False, ownership_closed=True,
        GPU_executed=False, model_executed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir(mode=0o700)
    save(row)
    print('Prepared production-linked head device gate;commit before host.', flush=True)


def execute(mode):
    row = json.loads((ROOT / 'record.json').read_text()); verify(row)
    require(row['stage'] == ('prepared' if mode == 'host' else 'host_passed'), 'no retry/duplicate device attempt')
    scope = None
    def persist():
        write(ROOT / 'execution.json', dict(active=bool(scope and scope.active),
            commands=scope.commands if scope else []))
    try:
        scope = owner.Owner(ROOT / ('owned-' + mode), persist)
        if mode == 'host':
            e,out,err=scope.run('reader-controls',[build.HOST_PYTHON,'-B','-m','unittest','-v',
                'tools.sycl.test_q6_dpas_component','tools.sycl.test_compile_census',
                'tools.sycl.test_q6_head_generation_speed'],row['environment'],P,**row['host_budget'])
            control=out.read_text()+err.read_text()
            require(owner.closed(e) and e['exit_code']==0 and 'Ran 47 tests' in control and '\nOK\n' in control,
                'all47prospective host controls required')
            row['host_controls']=dict(passed=True,cases=47,GPU_executed=False)
            cases = [('empty', ['--host'], {}, 'HOST owner_negative_cases=3 GPU=0\nPACKED_HOST owner_negative_cases=1 GPU=0'),
                ('flag3', ['--reject-switch', 'must be 0 or 1'], {'STRATA_SYCL_DECODE_Q6_DPAS_HEAD': '3'}, 'REJECT_SWITCH passed=1 GPU=0'),
                ('empty-flag', ['--reject-switch', 'must be 0 or 1'], {'STRATA_SYCL_DECODE_Q6_DPAS_HEAD': ''}, 'REJECT_SWITCH passed=1 GPU=0'),
                ('release0', ['--reject-switch', 'requires STRATA_PREFILL_RELEASE_HEADS=1'],
                    {'STRATA_SYCL_DECODE_Q6_DPAS_HEAD': '1', 'STRATA_PREFILL_RELEASE_HEADS': '0'}, 'REJECT_SWITCH passed=1 GPU=0'),
                ('release-absent', ['--reject-switch', 'requires STRATA_PREFILL_RELEASE_HEADS=1'],
                    {'STRATA_SYCL_DECODE_Q6_DPAS_HEAD': '1', 'STRATA_PREFILL_RELEASE_HEADS': None}, 'REJECT_SWITCH passed=1 GPU=0'),
                ('packed-release0', ['--reject-switch','requires STRATA_PREFILL_RELEASE_HEADS=1'],
                    {'STRATA_SYCL_DECODE_Q6_DPAS_HEAD':'2','STRATA_PREFILL_RELEASE_HEADS':'0'},'REJECT_SWITCH passed=1 GPU=0'),
                ('packed-release-absent', ['--reject-switch','requires STRATA_PREFILL_RELEASE_HEADS=1'],
                    {'STRATA_SYCL_DECODE_Q6_DPAS_HEAD':'2','STRATA_PREFILL_RELEASE_HEADS':None},'REJECT_SWITCH passed=1 GPU=0')]
            for label, args, changes, expected in cases:
                env = dict(row['environment']); env.pop('LD_PRELOAD', None)
                for k, v in changes.items():
                    if v is None: env.pop(k, None)
                    else: env[k] = v
                e, out, err = scope.run(label, [row['binary']['path'], *args], env, ROOT, **row['host_budget'])
                require(owner.closed(e) and e['exit_code'] == 0 and out.read_text().strip() == expected and
                    not re.search(r'--->\s*ur[A-Z]|<---\s*ur[A-Z]', err.read_text()), 'CPU owner/switch gate: ' + label)
                row['results'].append(dict(mode=label, passed=True, ownership_closed=True, GPU_executed=False))
            row['stage'] = 'host_passed'
        else:
            env = dict(row['environment'], LD_PRELOAD=row['native_validation_probe']['library']['path'],
                STRATA_UR_PROBE_EXPECTED_LOADER=row['native_validation_probe']['loader']['path'])
            cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
                ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
                if v.startswith('-- cursor: '))
            for label, target, args, expected in [
                ('submission', 'sycl_submission_test', ['--submission'], SUBMISSION),
                ('phase', 'sycl_verifier_retirement_test', ['--phase-cache'], [PHASE, 'phase_fixture_status=PASS cases=1'])]:
                outcome = dict(mode=label, passed=False)
                try:
                    e, out, err = scope.run(label, [row['binaries'][target]['path'], *args], env, ROOT, **row['native_budget'])
                    text, diagnostic = out.read_text(), err.read_text()
                    require(owner.closed(e) and e['exit_code'] == 0, 'small native fixture failed: ' + label)
                    require(([v for v in text.splitlines() if v.startswith('case ')] == expected and
                        text.count('fixture_status=PASS mode=submission') == 1) if label == 'submission'
                        else text.splitlines() == expected, 'small fixture extent differs: ' + label)
                    outcome.update(validation=validation_log(diagnostic, e['pid'],
                        row['native_validation_probe']['loader']['path'], True), host_probe_accepted=True, native_exit_code=0)
                finally:
                    validation.finish(outcome, scope, cursor, row['boot'])
                    row['results'].append(outcome)
                require(outcome['passed'], 'small fixture/fault/closure gate failed: ' + label)
                print('Production-linked device gate passed: ' + label, flush=True)
            run_gpu, _ = archived_gate()
            for arm in ('reference', 'canonical','sole-packed'):
                outcome = dict(mode=arm, passed=False)
                row['results'].append(outcome)
                run_gpu(row, scope, arm, outcome)
                print('Production-linked device gate passed: ' + arm, flush=True)
            row.update(stage='closed', passed=True, complete=True)
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, error=repr(error))
    finally:
        row['commands'] += scope.commands if scope else []
        row['active'] = bool(scope and scope.active)
        row['ownership_closed'] = not row['active'] and all(owner.closed(c) for c in row['commands'])
        row['passed'] = bool(row['passed'] and row['ownership_closed'])
        save(row)
    require(row['stage'] != 'failed' and row['ownership_closed'], row.get('error', 'device gate did not close'))
    print('Closed production-linked head ' + mode + ';commit before advancing.', flush=True)


def prepare_timing():
    require(not ROOT.exists() and KEY not in build.public(),'fresh timing cohort only')
    prior=json.loads((B/'q6-head-packed-device-v1/record.json').read_text())
    require(validation.committed(RESULT,('q6_head_packed_device_v1',))==compact(prior) and
        prior['stage']=='closed' and prior['passed'] and prior['complete'] and prior['ownership_closed'] and
        all(any(v['mode']==arm and v['passed'] and v['fresh_faults']==[] for v in prior['results'])
            for arm in ('reference','canonical','sole-packed')),'close all three fresh device owners first')
    n=native();require(n['binaries']==prior['binaries'],'same qualified four native artifacts required')
    row=copy.deepcopy(prior)
    row.update(stage='prepared',source_commit=build.source.git(P,'rev-parse','HEAD').decode().strip(),
        controller=build.source.pin(P,['tools/sycl']),results=[],commands=[],passed=False,complete=False,
        active=False,ownership_closed=True,GPU_executed=False,orders=['rc','cr','rc'],
        device_qualification=dict(key='q6_head_packed_device_v1',evidence_commit='5145e8eb7229cdec7b33711f708cc95949d853cd',
            native_artifacts_unchanged=True,model_or_speed_result_inherited=False),
        plan='47CPU timing/owner/census/protocol controls first;then three fresh sole-packed fixture processes'
             ' in R/C,C/R,R/C order. Each reruns four numerical and eight partial-column cases,two owner'
             ' retire/overwrite/restore cycles and12numerical replays,before three warmups/arm and'
             ' five paired samples/arm of256replays. Native and reader both enforce max(10ms,300timer-units).'
             ' Remove only full API tracing before exec;parameter validation,retained loader and all'
             ' runtime safety settings stay. Full shard hashes pre/post each process,normal0/closed/no fault.'
             ' Require all three process median ratios<1 for component advancement;no minimum gain.'
             ' Stop first failure,no retries. This is head C4 only,not MTP/model/PP/TTFT/TG adoption.',
        retention='Keep the30eligible per-process timing samples,paired ratios and numerical/fault/ownership summary.'
                  ' Retire normal API/stage logs and caches after the current model-entry consumer is updated;'
                  ' keep current1db candidate plus5a6 fallback and named90f/552b/e1b references.')
    row.pop('host_controls',None)
    original_root=str(B/'q6-head-packed-device-v1')
    row['environment']={k:v.replace(original_root,str(ROOT)) for k,v in row['environment'].items()}
    ROOT.mkdir(mode=0o700)
    for name in ('tmp','cache'):(ROOT/name).mkdir(mode=0o700)
    save(row);print('Prepared fresh three-process C4 timing cohort;commit before controls.',flush=True)


def execute_timing(mode):
    row=json.loads((ROOT/'record.json').read_text());verify(row)
    require(row['stage']==('prepared' if mode=='timing-host' else 'host_passed'),'no timing retry/duplicate')
    scope=None
    def persist():write(ROOT/'execution.json',dict(active=bool(scope and scope.active),commands=scope.commands if scope else []))
    try:
        scope=owner.Owner(ROOT/('owned-'+mode),persist)
        if mode=='timing-host':
            e,out,err=scope.run('reader-controls',[build.HOST_PYTHON,'-B','-m','unittest','-v',
                'tools.sycl.test_q6_dpas_component','tools.sycl.test_compile_census','tools.sycl.test_q6_head_generation_speed'],
                row['environment'],P,**row['host_budget'])
            text=out.read_text()+err.read_text()
            require(owner.closed(e) and e['exit_code']==0 and 'Ran 47 tests' in text and '\nOK\n' in text,
                'all47timing/owner controls required')
            row.update(stage='host_passed',host_controls=dict(passed=True,cases=47,GPU_executed=False))
        else:
            for i,order in enumerate(row['orders']):
                result=dict(mode='sole-packed',process=i,order=order,passed=False);row['results'].append(result)
                run_gpu(row,scope,'sole-packed',result,order=order,label='timing'+str(i))
                print('Completed C4 timing process'+str(i)+' '+order,flush=True)
            ratios=[v['timing']['candidate_over_reference'] for v in row['results']]
            gm=math.exp(sum(math.log(v) for v in ratios)/len(ratios))
            row.update(stage='closed',passed=True,complete=True,
                decision=dict(process_ratios=ratios,geometric_time_ratio=gm,component_speedup=1/gm,
                    head_component_gain=all(v<1 for v in ratios),model_speed_qualified=False,inference_adopted=False,
                    next='Only a repeatable full-head gain permits the new same-engine0/2 MTP/model proposal;'
                         'MTP subset arithmetic,placement,phase preservation,long/agent/task and full context stay open.'))
    except BaseException as error:row.update(stage='failed',passed=False,complete=False,error=repr(error))
    finally:
        row['commands']+=scope.commands if scope else [];row['active']=bool(scope and scope.active)
        row['ownership_closed']=not row['active'] and all(owner.closed(c) for c in row['commands'])
        row['passed']=bool(row['passed'] and row['ownership_closed']);save(row)
    require(row['stage']!='failed' and row['ownership_closed'],row.get('error','timing did not close'))
    print('Closed '+mode+';commit before advancing.',flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['host'], ['device'],['prepare-timing'],['timing-host'],['timing']),
        'check_q6_head_model.py prepare|host|device|prepare-timing|timing-host|timing')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if sys.argv[1] in ('prepare-timing','timing-host','timing'):
            ROOT,KEY=B/'q6-head-packed-timing-v1','q6_head_packed_timing_v1'
            prepare_timing() if sys.argv[1]=='prepare-timing' else execute_timing(sys.argv[1])
        else:prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1])
