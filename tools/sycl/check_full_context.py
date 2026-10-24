#!/usr/bin/env python3
"""Check CLI admission and a bounded, traced device fixture; never run a model.

Main-authored; repository LICENSE. The full token fixture is for the subsequent
physical-capacity diagnostic, not a speed or task-quality benchmark.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import build_full_context as native
import ur_validation as validation

P, B = native.P, native.B
ROOT = B / 'full-context-cli-v1'
RESULT = native.RESULT
KEY = 'native_cli'
INPUTS = ['tools/sycl/check_full_context.py', 'tools/sycl/build_full_context.py',
          'tools/sycl/ur_validation.py', *native.source.INPUTS]
REJECTION = 'strata generate: --consume-prompt requires a one-shot native SYCL token-file '


def write(path, value):
    native.write(path, value)


def inputs(revision):
    result = {}
    for name, data in native.build.blobs(P, revision, sorted(set(INPUTS))):
        native.guards.require(data == (P / name).read_bytes(), 'commit exact CLI input: ' + name)
        result[name] = hashlib.sha256(data).hexdigest()
    return result


def engine():
    record = json.loads(native.ENGINE_RECORD.read_text())
    native.guards.require(record['stage'] == 'closed_link' and record['passed'] and record['complete'] and
        record['ownership_closed'] and not record['active'] and not record['GPU_executed'] and
        record['actual_dependency_closure']['uncovered_files'] == 0, 'closed pinned native link required')
    native.check_inputs(record)
    native.guards.require(native.mkl_inputs() == record['external_inputs'] and
        native.build.digest(record['binary']['path']) == record['binary']['sha256'], 'native engine inputs changed')
    for name, sha256 in record['actual_dependency_closure']['generated_sha256'].items():
        native.guards.require(native.build.digest(name) == sha256, 'generated native input changed')
    return record


def token_bytes(n):
    return ''.join(str(i % 4096) + '\n' for i in range(n)).encode()


def cases(binary):
    base = [binary, '--consume-prompt', '--native', str(ROOT / 'absent-model.gguf'),
            '--max-context', '262144', '--max-new', '0', '--spec', '4', '--suffix-draft', '0',
            '--prefill', '8192', '--tokens-file', str(ROOT / 'tokens262144.txt')]
    result = [dict(name='help', argv=[binary, '--help'], exit_code=0),
              dict(name='missing-full-request', argv=[binary, '--consume-prompt'], exit_code=2)]
    for name, flag, value in [('under-context', '--max-context', '262143'),
        ('over-context', '--max-context', '262145'), ('output-room', '--max-new', '1'),
        ('wrong-spec', '--spec', '2'), ('suffix', '--suffix-draft', '3'),
        ('wrong-chunk', '--prefill', '4096'), ('under-input', '--tokens-file', str(ROOT / 'tokens262143.txt')),
        ('over-input', '--tokens-file', str(ROOT / 'tokens262145.txt'))]:
        argv = list(base)
        argv[argv.index(flag) + 1] = value
        result.append(dict(name=name, argv=argv, exit_code=2))
    argv = list(base)
    del argv[argv.index('--native'):argv.index('--native') + 2]
    result.append(dict(name='missing-native', argv=argv, exit_code=2))
    return result


def summary(record):
    compact = {k: v for k, v in record.items() if k not in ('environment', 'commands', 'cases', 'outcomes')}
    compact['case_names'] = [case['name'] for case in record['cases']]
    if 'outcomes' in record:
        compact['outcomes'] = [{k: v for k, v in row.items() if not k.endswith('_sha256')}
                               for row in record['outcomes']]
    return compact


def prepare(container):
    native.guards.require(KEY not in container and not ROOT.exists(), 'fresh CLI gate only;inspect existing owner')
    full = engine()
    revision = native.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = inputs(revision)
    ROOT.mkdir(mode=0o700)
    (ROOT / 'tmp').mkdir()
    (ROOT / 'cache').mkdir()
    tokens = []
    for n in (262143, 262144, 262145):
        path = ROOT / ('tokens' + str(n) + '.txt')
        data = token_bytes(n)
        path.write_bytes(data)
        tokens.append(dict(path=str(path), rows=n, first_id=0, last_id=(n - 1) % 4096,
                           sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    env = dict(full['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
        CUDA_MODULE_LOADING='EAGER', UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:stderr')
    record = dict(stage='prepared', source_commit=revision, input_sha256=pins, binary=full['binary'],
        native_source_commit=full['source_commit'], native_record_sha256=native.build.digest(native.ENGINE_RECORD),
        token_files=tokens, cases=cases(full['binary']['path']), environment=env,
        active=False, passed=False, complete=False, ownership_closed=True,
        GPU_work_submitted=False, physical262144_qualified=False, model_adoption_qualified=False,
        criteria='Help exposes diagnostic flag;all10 invalid consume requests exit2 at the exact pre-device rejection and emit no stdout. No UR context/queue/kernel/copy submission. Normal owned exits and unchanged native/original assets. These are CLI refusal checks,not physical-context or decode qualification.',
        token_scope='Synthetic integer IDs cycle0..4095. Under/full/over input fixtures exercise actual parser extent. No tokenizer,thinking,generated answer,throughput or task-quality evidence. Keep full fixture only for imminent physical full-fill consumer;retire tails and successful raw output after this gate.',
        environment_note='All variables fixed before exec;CUDA_MODULE_LOADING=EAGER also avoids the legacy startup setenv branch. Existing qualified SYCL safety settings are unchanged.',
        budgets=dict(wall_s=15,cpu_s=15,rss_bytes=1 << 30,text_bytes=1 << 20,file_bytes=4 << 20,total_bytes=16 << 20),
        next_gate='Actual bounded device validation,262144 allocation/processed extent,last-slot commit,near-cap decode,RESTORE and failure-owner checks;independently repeated phase preservation before adoption.')
    write(ROOT / 'record.json', record)
    container[KEY] = summary(record)
    write(RESULT, container)
    print('Prepared executable CLI gate and deterministic full-context IDs;commit before run.', flush=True)


def admission(record, container):
    native.guards.require(RESULT.read_bytes() == native.build.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))) and
        container[KEY] == summary(record) and inputs(record['source_commit']) == record['input_sha256'],
        'commit exact CLI source/plan before run')
    full = engine()
    native.guards.require(native.build.digest(native.ENGINE_RECORD) == record['native_record_sha256'] and
        full['binary'] == record['binary'], 'native link identity differs')
    for row in record['token_files']:
        path = Path(row['path'])
        native.guards.require(path.read_bytes() == token_bytes(row['rows']) and
            native.build.digest(path) == row['sha256'], 'generated token input changed')


def run(container):
    record = json.loads((ROOT / 'record.json').read_text())
    native.guards.require(record['stage'] == 'prepared', 'no duplicate/retry CLI gate')
    admission(record, container)
    original = json.loads((ROOT / 'record.json').read_text())
    owner = None
    def persist():
        record['active'] = bool(owner and owner.active is not None)
        write(ROOT / 'record.json', dict(record, commands=owner.commands if owner else []))
    owner = native.supervision.Owner(ROOT / 'owned', persist)
    record['stage'] = 'running'
    persist()
    try:
        b = record['budgets']
        for case in record['cases']:
            native.guards.require(native.build.digest(record['binary']['path']) == record['binary']['sha256'],
                                  'executable changed before CLI case')
            entry, out, err = owner.run(case['name'], case['argv'], record['environment'], ROOT,
                wall=b['wall_s'], cpu=b['cpu_s'], rss_cap=b['rss_bytes'], text_cap=b['text_bytes'],
                file_cap=b['file_bytes'], total_cap=b['total_bytes'])
            stdout, stderr = out.read_text(errors='replace'), err.read_text(errors='replace')
            ok = bool(native.supervision.closed(entry) and entry['exit_code'] == case['exit_code'] and
                (('--consume-prompt' in stdout + stderr) if case['name'] == 'help'
                 else not stdout and REJECTION in stderr) and not any(
                    marker in stderr for marker in ('urEnqueue', 'urQueueCreate', 'urContextCreate')))
            record.setdefault('outcomes', []).append(dict(name=case['name'], passed=ok,
                exit_code=entry.get('exit_code'), ownership_closed=native.supervision.closed(entry),
                stdout_sha256=native.build.digest(out), stderr_sha256=native.build.digest(err)))
            native.guards.require(ok, 'CLI case failed: ' + case['name'] + ':' + stderr[-2048:])
        admission(original, container)
        record.update(stage='closed', passed=True, complete=True)
    except BaseException as error:
        record.update(stage='failed', passed=False, complete=False, error=repr(error))
    finally:
        record['ownership_closed'] = owner.active is None and all(native.supervision.closed(c) for c in owner.commands)
        record['passed'] = bool(record['passed'] and record['ownership_closed'])
        persist()
        container[KEY] = summary(record)
        write(RESULT, container)
    print(json.dumps({k: record.get(k) for k in ('stage', 'passed', 'complete', 'ownership_closed', 'error')}), flush=True)
    return 0 if record['passed'] else 1


DEVICE_ROOT = B / 'full-context-device-v2'
DEVICE_KEY = 'device_smoke_v2'


def device_admission(record, container):
    native.guards.require(RESULT.read_bytes() == native.build.git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))) and
        container[DEVICE_KEY] == {k:v for k,v in record.items() if k not in ('environment','commands')} and
        inputs(record['source_commit']) == record['input_sha256'], 'commit exact device plan first')
    native.guards.require(hashlib.sha256(json.dumps(record['environment'],sort_keys=True).encode()).hexdigest() ==
        record['environment_sha256'], 'effective device environment differs from committed plan')
    native.guards.require(record['environment'].get('STRATA_UR_PROBE_EXPECTED_LOADER') ==
        record['preload']['loader']['path'], 'preloader provider pin missing')
    full, reference = native.source.baseline()
    native.guards.require(reference == record['selected_original'] and
        full['fixtures']['sycl_expert_bank_test'] == record['fixture'], 'original fixture identity differs')
    native.guards.require(engine()['binary'] == record['diagnostic_binary'] and
        validation.preload_qualification() == record['preload'], 'runtime/diagnostic identity differs')


def prepare_device(container):
    native.guards.require(DEVICE_KEY not in container and not DEVICE_ROOT.exists(), 'fresh device smoke only')
    cli = container['native_cli']
    native.guards.require(cli['passed'] and cli['complete'] and cli['ownership_closed'] and
        not cli['active'] and len(cli['outcomes']) == 11, 'closed executable CLI gate required')
    diagnostic = engine()
    full, reference = native.source.baseline()
    preload = validation.preload_qualification()
    revision = native.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = inputs(revision)
    # Audit uses the unchanged original source, not an inferred source identity
    # for a newly compiled fixture. The actual fixture bytes are preserved.
    name = 'tests/sycl/expert_bank.cpp'
    native.guards.require(native.build.digest(P / name) == full['project']['source_sha256'][name],
                          'smoke source audit no longer describes original fixture')
    DEVICE_ROOT.mkdir(mode=0o700)
    (DEVICE_ROOT / 'tmp').mkdir()
    (DEVICE_ROOT / 'cache').mkdir()
    env = dict(diagnostic['environment'], TMPDIR=str(DEVICE_ROOT/'tmp'), XDG_CACHE_HOME=str(DEVICE_ROOT/'cache'),
        CUDA_MODULE_LOADING='EAGER', UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
        UR_LOG_TRACING='level:info;flush:info;output:stderr',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr')
    native.guards.require(not env.get('LD_PRELOAD'), 'unexpected existing preload')
    env['LD_PRELOAD'] = preload['library']['path']
    env['STRATA_UR_PROBE_EXPECTED_LOADER'] = preload['loader']['path']
    fixture = full['fixtures']['sycl_expert_bank_test']
    record = dict(stage='prepared', source_commit=revision, input_sha256=pins,
        selected_original=reference, fixture=fixture, fixture_source_commit=full['source_commit'],
        diagnostic_binary=diagnostic['binary'], preload=preload,
        command=[fixture['path'], '--smoke'], environment=env,
        environment_sha256=hashlib.sha256(json.dumps(env,sort_keys=True).encode()).hexdigest(),
        driver_before=native.guards.driver_guard(), boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        active=False, passed=False, complete=False, ownership_closed=True, GPU_attempted=False,
        GPU_work_submitted=None, physical262144_qualified=False, model_adoption_qualified=False,
        budgets=dict(wall_s=90,cpu_s=180,rss_bytes=4<<30,text_bytes=16<<20,file_bytes=16<<20,total_bytes=64<<20),
        source_audit='Original6e40 smoke:256/512-byte slots,explicit256-item padded workgroups with uniform collectives,in-order upload/read/reuse,device/host guard checks,foreign-context rejection before enqueue,and owners outliving checked error unwind. No process-environment mutation or model arithmetic.',
        criteria='Original fixture6cases and final PASS;qualified same-child before-main validation/first-init proof before GPU work;trace shows kernel enqueue. No unexpected post-canary UR/LevelZero/teardown errors. Normal owned exit,unchanged source/assets/boot/driver,no fresh fault. No model/physical/decode/speed claim.',
        retention='Reuse original same-build fixture;no additional native generation. Retain compact case/activation/fault results,then retire successful full API trace. Diagnostic engine/full262144 IDs remain for imminent model-capacity consumer.',
        next_gate='Diagnostic engine short model validation,then actual262144 allocation/last-slot completion,near-cap/RESTORE/fault-owner checks and repeated phase preservation before adoption.')
    write(DEVICE_ROOT/'record.json',record)
    container[DEVICE_KEY] = {k:v for k,v in record.items() if k not in ('environment','commands')}
    write(RESULT,container)
    print('Prepared original-fixture traced device smoke;commit before run.',flush=True)


def run_device(container):
    record = json.loads((DEVICE_ROOT/'record.json').read_text())
    native.guards.require(record['stage']=='prepared','no duplicate/retry device smoke')
    device_admission(record,container)
    native.guards.require(record['driver_before']==native.guards.driver_guard() and
        record['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'driver/boot changed')
    cursor_text = subprocess.check_output(['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'],text=True,timeout=15)
    cursor = next(s.split(': ',1)[1] for s in cursor_text.splitlines() if s.startswith('-- cursor: '))
    original = json.loads((DEVICE_ROOT/'record.json').read_text())
    owner = None
    def persist():
        record['active'] = bool(owner and owner.active is not None)
        write(DEVICE_ROOT/'record.json',dict(record,commands=owner.commands if owner else []))
    owner = native.supervision.Owner(DEVICE_ROOT/'owned',persist)
    record.update(stage='running',GPU_attempted=True)
    persist()
    try:
        b = record['budgets']
        entry,out,err = owner.run('original-bank-smoke',record['command'],record['environment'],DEVICE_ROOT,
            wall=b['wall_s'],cpu=b['cpu_s'],rss_cap=b['rss_bytes'],text_cap=b['text_bytes'],
            file_cap=b['file_bytes'],total_cap=b['total_bytes'])
        stdout,stderr = out.read_text(errors='replace'),err.read_text(errors='replace')
        record['exit_code'] = entry.get('exit_code')
        record['activation'] = validation.preload_evidence(stderr,entry['pid'])
        boundary = re.search(r'^strata UR validation: .*phase=before-main[^\n]*\n',stderr,re.M)
        native.guards.require(boundary is not None and record['activation']['loader']==record['preload']['loader']['path'],
                              'same-child validation activation missing')
        body = stderr[boundary.end():]
        record['GPU_work_submitted'] = 'urEnqueueKernelLaunch' in body
        lines = stdout.splitlines()
        native.guards.require(native.supervision.closed(entry) and entry['exit_code']==0 and len(lines)==7 and
            all(s.startswith('expert_bank_case ') and ' status=0 ' in s and
                s.endswith('immutable_slab=PASS guards=PASS private_map=PASS checked_drain=PASS') for s in lines[:6]) and
            lines[-1]=='expert_bank_fixture PASS model=UNRUN physical262144_RESTORE=UNQUALIFIED' and
            record['GPU_work_submitted'] and not re.search(
                r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain|cleanup failed',body),
            'traced expert-bank smoke failed:'+stderr[-4096:])
        record['cases'] = 6
        record['checked_retirement'] = True
        device_admission(original,container)
        record.update(stage='closed',passed=True,complete=True)
    except BaseException as error:
        record.update(stage='failed',passed=False,complete=False,error=repr(error))
    finally:
        record['ownership_closed'] = owner.active is None and all(native.supervision.closed(c) for c in owner.commands)
        try:
            text = subprocess.check_output(['journalctl','-k','-b','--after-cursor',cursor,'--no-pager'],text=True,timeout=15)
            record['fresh_faults'] = [s for s in text.splitlines() if re.search(
                r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))',s)]
            record['driver_after'] = native.guards.driver_guard()
            record['boot_unchanged'] = record['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        except BaseException as error:
            record['final_probe_error'] = repr(error)
        record['passed'] = bool(record['passed'] and record['ownership_closed'] and
            record.get('fresh_faults')==[] and record.get('boot_unchanged') and
            record.get('driver_after')==record['driver_before'] and 'final_probe_error' not in record)
        record['complete'] = record['passed']
        if not record['passed']: record['stage'] = 'failed'
        persist()
        container[DEVICE_KEY] = {k:v for k,v in record.items() if k not in ('environment','commands')}
        write(RESULT,container)
    print(json.dumps({k:record.get(k) for k in ('stage','passed','complete','ownership_closed','GPU_work_submitted','fresh_faults','error','final_probe_error')}),flush=True)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    native.guards.require(sys.argv[1:] in (['prepare'], ['run'], ['prepare-device'], ['device']),
                          'prepare|run|prepare-device|device')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        native.guards.no_model_owners([])
        container = json.loads(RESULT.read_text())
        if sys.argv[1]=='prepare-device': prepare_device(container)
        elif sys.argv[1]=='device': sys.exit(run_device(container))
        else: prepare(container) if sys.argv[1]=='prepare' else sys.exit(run(container))
