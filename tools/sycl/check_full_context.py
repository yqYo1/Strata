#!/usr/bin/env python3
"""Check the diagnostic executable's pre-device CLI boundary; never run a model.

Main-authored; repository LICENSE. The full token fixture is for the subsequent
physical-capacity diagnostic, not a speed or task-quality benchmark.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import build_full_context as native

P, B = native.P, native.B
ROOT = B / 'full-context-cli-v1'
RESULT = native.RESULT
KEY = 'native_cli'
INPUTS = ['tools/sycl/check_full_context.py', 'tools/sycl/build_full_context.py', *native.source.INPUTS]
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


if __name__ == '__main__':
    native.guards.require(sys.argv[1:] in (['prepare'], ['run']), 'prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        native.guards.no_model_owners([])
        container = json.loads(RESULT.read_text())
        prepare(container) if sys.argv[1] == 'prepare' else sys.exit(run(container))
