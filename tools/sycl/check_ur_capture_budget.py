#!/usr/bin/env python3
"""Bounded host-only qualification of subscriber loss handling; repository LICENSE.

Never enumerate a device or load this candidate into a model. Keep the qualified
observer as fallback until a separate device/observer-cost plan selects a replacement.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generation_engine as gen
import measurement_owner as supervision
import ur_call_subscriber as subscriber

P, B, RESULT = subscriber.P, subscriber.B, subscriber.RESULT
ROOT = B / 'ur-call-budget-v2'
KEY = 'ur_subscriber_budget_v2'
DIRECT_ROOT = B / 'ur-call-budget-direct-v1'
DIRECT_KEY = 'ur_subscriber_budget_direct'
require, digest = subscriber.require, subscriber.digest
BUDGET = dict(wall=120, cpu=120, rss_cap=4 << 30, text_cap=4 << 20,
              file_cap=32 << 20, total_cap=64 << 20)


def write(path, value, key=KEY):
    if path == RESULT:
        # Preserve unrelated compact tables byte-for-byte. This controller owns
        # one result row, not the formatting or retention of older experiments.
        base = gen.git('show', 'HEAD:' + str(path.relative_to(P))).decode()
        tag = json.dumps(key) + ':'
        encoded = json.dumps(value['event_reconciliation'][key], indent=2).replace('\n', '\n    ')
        if tag in base:
            require(base.count(tag) == 1, 'ambiguous control row')
            start = base.index(tag) + len(tag)
            while base[start].isspace(): start += 1
            _, end = json.JSONDecoder().raw_decode(base[start:])
            text = base[:start] + encoded + base[start + end:]
        else:
            label = '"event_reconciliation":'
            start = base.index('{', base.index(label) + len(label))
            _, end = json.JSONDecoder().raw_decode(base[start:])
            text = (base[:start] + base[start:start + end - 1].rstrip() +
                    ',\n    ' + tag + ' ' + encoded + '\n  }' + base[start + end:])
        require(json.loads(text) == value, 'unexpected change outside the owned result row')
        path.write_text(text)
        return
    path.write_text(json.dumps(value, indent=2) + '\n')


def inputs(revision):
    pins = gen.benchmark_pins(revision)
    for name in (subscriber.SOURCE, subscriber.DRIVER_SOURCE, 'tests/core/test_ur_call_subscriber.py'):
        raw = gen.git('show', revision + ':' + name)
        require(raw == (P / name).read_bytes(), 'uncommitted native/control input')
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def admit(row, binary=False, recover_original=False):
    if recover_original:
        # A later controller does not relabel the already compiled observer.
        # Recover every original input; the actual native sources/headers/tools
        # and ELF bytes are independently verified below.
        for name, raw in gen.blobs(P, row['source_commit'], row['inputs']):
            require(hashlib.sha256(raw).hexdigest() == row['inputs'][name], 'original source not recoverable')
    else:
        require(inputs(row['source_commit']) == row['inputs'], 'source/controller changed')
    _, proof = subscriber.qualified_native()
    require(proof == row['baseline_host_proof'], 'qualified fallback changed')
    for item in row.get('compiler_inputs', []):
        require(digest(item['path']) == item['sha256'], 'compiler manifest changed')
        for path, sha in json.loads(Path(item['path']).read_text()).items():
            require(digest(path) == sha, 'compiler input changed: ' + path)
    if binary:
        for value in row['artifacts'].values():
            require(digest(value['path']) == value['sha256'], 'candidate ELF changed')


def prepare(row, owner):
    require(not row, 'fresh host candidate only')
    baseline, proof = subscriber.qualified_native()
    source = gen.git('rev-parse', 'HEAD').decode().strip()
    row.update(stage='preparing', source_commit=source, inputs=inputs(source),
        baseline_host_proof=proof, active=False, ownership_closed=True,
        complete=False, passed=False, GPU_qualified=False, model_qualified=False,
        retained_generations='Qualified v3 observer retained;this tiny host candidate serves imminent loss controls and later separately planned device/cost gates. Selected6e40 engine and12fixtures untouched.',
        consumer='Bounded observer loss handling for upcoming actual-route attribution.',
        retention='Retain source/recipe and compact controls;retire successful raw callbacks/build output after extraction. Select observer only after device/cost qualification.',
        gates='Two original metadata fixtures,four immutable-environment controls,seven concurrent overflow caps,two first-omitted invalid begin/end boundary controls,eight invalid pre-exec caps. Stop first failure. Overflow must continue application but fail complete-capture parser. No performance/phase/device/ordinary teardown claim.',
        environment=dict(baseline['environment'], TMPDIR=str(ROOT / 'tmp')),
        commands={}, compiler_inputs=[], outcomes=[])
    (ROOT / 'tmp').mkdir()
    for key, old, output in (('library', 'command', 'libstrata_ur_calls.so'),
                              ('driver', 'driver_command', 'ur_call_guard_test')):
        args = list(baseline[old])
        args[-1] = str(ROOT / output)
        if '-pthread' not in args:
            args.insert(1, '-pthread')
        row['commands'][key] = args
        source_path = str(P / (subscriber.SOURCE if key == 'library' else subscriber.DRIVER_SOURCE))
        prefix = [a for a in args[:args.index(source_path)] if a != '-shared']
        dep = ROOT / (key + '.d')
        entry, _, _ = owner.run('deps-' + key, prefix + ['-M', '-MT', 'target', '-MF', str(dep), source_path],
                                row['environment'], P, **BUDGET)
        require(supervision.closed(entry) and entry['exit_code'] == 0, 'dependency scan failed')
        text = dep.read_text().replace('\\\n', ' ').split(':', 1)[1]
        require('\\' not in text, 'unexpected dependency escaping')
        paths = {Path(n).resolve() for n in text.split()}
        # Recover the verified baseline's toolchain/link inputs too; source paths
        # are hashed at their current committed revision, not relabelled as old.
        legacy = baseline['compiler_inputs' if key == 'library' else 'driver_compiler_inputs']
        paths.update(Path(n).resolve() for n in json.loads(Path(legacy['path']).read_text()))
        paths.update(Path(n) for n in baseline['compiler_tools'])
        manifest = ROOT / (key + '-inputs.json')
        write(manifest, {str(p): digest(p) for p in sorted(paths)})
        row['compiler_inputs'].append(dict(path=str(manifest), sha256=digest(manifest), files=len(paths)))
    admit(row)
    row['stage'] = 'prepared'


def build(row, owner):
    require(row['stage'] == 'prepared' and not (ROOT / 'libstrata_ur_calls.so').exists(), 'single-use frozen build')
    admit(row)
    for name, command in row['commands'].items():
        entry, _, _ = owner.run('build-' + name, command, row['environment'], P, **BUDGET)
        require(supervision.closed(entry) and entry['exit_code'] == 0, 'host compile failed')
    row['artifacts'] = {k: dict(path=v[-1], bytes=Path(v[-1]).stat().st_size, sha256=digest(v[-1]))
                        for k, v in row['commands'].items()}
    admit(row, True)
    row['stage'] = 'built'


def check(row, owner):
    require(row['stage'] == 'built' and not row['outcomes'], 'single-use frozen checks')
    admit(row, True)
    dispatcher = subscriber.runtime_assets()['dispatcher']['path']
    library = row['artifacts']['library']['path']
    cases = [('metadata', 'metadata', None), ('buffers', 'buffers', None)]
    cases += [(name, name, None) for name in ('setenv', 'unsetenv', 'putenv', 'clearenv')]
    cases += [('overflow-' + str(cap), 'overflow', cap) for cap in (None, 2, 3, 4, 5, 8, 511)]
    cases += [(mode, mode, cap) for mode, cap in (('overflow-boundary-begin', 4), ('overflow-boundary-end', 3))]
    cases += [('invalid-' + str(i), 'metadata', cap) for i, cap in enumerate(('', '0', '1', '65537', '-1', '+3', '3x', '99999999999999999999'))]
    for name, mode, cap in cases:
        log = ROOT / (name + '.jsonl')
        env = dict(row['environment'], LD_PRELOAD=library, XPTI_TRACE_ENABLE='1',
                   XPTI_FRAMEWORK_DISPATCHER=dispatcher, XPTI_SUBSCRIBERS=library, STRATA_UR_CALL_LOG=str(log))
        if cap is not None:
            env['STRATA_UR_CALL_RECORD_LIMIT'] = str(cap)
        entry, stdout, stderr = owner.run(name, [row['artifacts']['driver']['path'], mode], env, P, **BUDGET)
        require(supervision.closed(entry), 'host owner did not close')
        stdout, stderr = stdout.read_text(), stderr.read_text()
        sample = dict(case=name, exit_code=entry['exit_code'], ownership_closed=True)
        if name.startswith('invalid-'):
            require(entry['exit_code'] == 86 and not stdout and not log.exists() and
                    'incomplete diagnostic' in stderr, 'invalid cap was accepted')
        elif mode in ('setenv', 'unsetenv', 'putenv', 'clearenv'):
            require(entry['exit_code'] == 87 and 'mutation-case-begin:' + mode in stdout and
                    'environment mutation blocked' in stderr, 'environment control failed')
        else:
            require(entry['exit_code'] == 0 and not stderr, 'host callback fixture failed')
            text = log.read_text()
            if mode.startswith('overflow'):
                limit = 65536 if cap is None else cap
                rows = [json.loads(line) for line in text.splitlines()]
                require(len(rows) == limit + 1 and rows[-1]['kind'] == 'overflow' and
                        sum(x['kind'] == 'overflow' for x in rows) == 1 and
                        rows[-1]['limit'] == rows[-1]['records'] == limit and
                        '"application_continued":true' in stdout and len(text.encode()) <= limit * 2048 + 256,
                        'loss marker,finite output or application continuation failed')
                if mode.startswith('overflow-boundary'):
                    require(rows[-2]['kind'] == ('begin' if mode.endswith('end') else 'end'),
                            'wrong first-omitted callback boundary')
                try:
                    subscriber.parse_calls(text, entry['pid'])
                except RuntimeError as error:
                    require('record budget exhausted' in str(error), 'wrong loss refusal')
                else:
                    raise RuntimeError('incomplete capture accepted')
                sample.update(records=limit, loss_markers=1, complete_capture_rejected=True, application_continued=True)
            else:
                sample['fixture'] = (subscriber.metadata_fixture if mode == 'metadata' else subscriber.buffer_fixture)(text, entry['pid'])
        row['outcomes'].append(sample)
    admit(row, True)
    row.update(stage='host-checked', complete=True, passed=True)


def direct_host():
    host = subscriber.r.validation.committed(RESULT, ('event_reconciliation', KEY))
    require(host['passed'] and host['complete'] and host['ownership_closed'] and not host['active']
            and len(host['outcomes']) == 23 and not host.get('native_artifacts_retired'),
            'closed boundary-safe host qualification required')
    admit(host, True, recover_original=True)
    return host


def direct_prepare(row, owner):
    require(not row, 'fresh direct runtime controls only')
    host = direct_host()
    assets = subscriber.runtime_assets()
    component = subscriber.r.validation.committed(subscriber.r.validation.INDEX,
        ('UR_silent_validation_native_child_scope', 'preload_component'))
    ref = component['original_plan']
    raw = gen.git('show', ref['commit'] + ':' + ref['path'])
    require(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'original canary recipe changed')
    original = json.loads(raw)['UR_silent_validation_native_child_scope']['canary_plan']
    row.update(stage='prepared', source_commit=gen.git('rev-parse', 'HEAD').decode().strip(),
        observer_native_source=host['source_commit'], host_reference_commit=gen.git('rev-parse', 'HEAD').decode().strip(),
        host_sha256=hashlib.sha256(json.dumps(host, sort_keys=True).encode()).hexdigest(),
        assets=assets, canary=component['canary'], cases=[], outcomes=[], active=False,
        ownership_closed=True, complete=False, passed=False, gpu_work=False, model_qualified=False,
        limits='Actual UR result/output control only. No device enumeration,queue,enqueue,GPU,model,observer cost,ordinary runtime teardown or phase-coverage qualification.',
        gates='Complete and cap3 fresh host processes:before-main validation,actual46 result and UINT32_MAX sentinel preserved,two rejected API pairs in complete arm,one explicit loss marker and parser refusal in truncated arm. Full UR trace must contain no device/queue/enqueue work. Stop first failure.',
        retention='Keep compact actual results and source/recipe. Retire successful call/trace logs after extraction. Same tiny boundary-safe observer and existing canary,no new build.')
    row['inputs'] = inputs(row['source_commit'])
    for name, cap in (('complete', None), ('loss', '3')):
        root = DIRECT_ROOT / name
        root.mkdir(); (root / 'tmp').mkdir(); (root / 'cache').mkdir()
        env = {k: v.replace(original['root'], str(root)) for k, v in original['native_environment'].items()}
        env.update(LD_PRELOAD=assets['preload']['library']['path'] + ':' + host['artifacts']['library']['path'],
            XPTI_FRAMEWORK_DISPATCHER=assets['dispatcher']['path'], XPTI_SUBSCRIBERS=host['artifacts']['library']['path'],
            XPTI_TRACE_ENABLE='1', SYCL_UR_TRACE='0', STRATA_UR_CALL_LOG=str(root / 'calls.jsonl'),
            STRATA_UR_PROBE_EXPECTED_LOADER=assets['preload']['loader'], NEOReadDebugKeys='1',
            EnableDirectSubmission='0', UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1')
        if cap is not None: env['STRATA_UR_CALL_RECORD_LIMIT'] = cap
        row['cases'].append(dict(name=name, root=str(root), environment=env, command=[row['canary']['path']]))


def direct_check(row, owner):
    require(row['stage'] == 'prepared' and not row['outcomes'], 'single-use direct runtime controls')
    host = direct_host()
    require(hashlib.sha256(json.dumps(host, sort_keys=True).encode()).hexdigest() == row['host_sha256'] and
            inputs(row['source_commit']) == row['inputs'] and subscriber.runtime_assets() == row['assets'] and
            digest(row['canary']['path']) == row['canary']['sha256'], 'direct inputs changed')
    for case in row['cases']:
        entry, out, err = owner.run(case['name'], case['command'], case['environment'], P,
            wall=30, cpu=20, rss_cap=2 << 30, text_cap=1 << 20, file_cap=4 << 20, total_cap=8 << 20)
        require(supervision.closed(entry) and entry['exit_code'] == 0, 'actual UR canary failed')
        actual = json.loads(out.read_text())
        require(actual['pid'] == entry['pid'] and actual['initialized'] == actual['closed'] == 0 and
                actual['checked'] == actual['expected'] == 46 and actual['version'] == 4294967295,
                'actual UR result/output changed')
        proof = subscriber.r.validation.preload_evidence(err.read_text(), entry['pid'])
        require(proof['loader'] == row['assets']['preload']['loader'], 'actual loader changed')
        root = Path(case['root'])
        trace = (root / 'trace.log').read_text()
        import re
        require('urPlatformGetApiVersion' in trace and 'UR_RESULT_ERROR_INVALID_NULL_HANDLE' in trace and
                not re.search(r'ur(?:Enqueue|Kernel|Queue|Device|PlatformGet\()', trace),
                'missing actual UR response or unexpected device work')
        require(not (root / 'validation.log').exists() or not (root / 'validation.log').stat().st_size,
                'parameter warning/error')
        text = (root / 'calls.jsonl').read_text()
        sample = dict(case=case['name'], exit_code=0, ownership_closed=True,
            actual=actual, before_main=proof, unexpected_device_work=False)
        if case['name'] == 'complete':
            parsed = subscriber.parse_calls(text, entry['pid'])
            require(parsed['call_pairs'] == 2 and parsed['outcomes'] ==
                    [dict(name='urPlatformGetApiVersion', status=46, calls=2)], 'actual UR call census differs')
            sample['calls'] = parsed
        else:
            rows = [json.loads(line) for line in text.splitlines()]
            require(len(rows) == 4 and rows[-1]['kind'] == 'overflow' and
                    rows[-1]['records'] == rows[-1]['limit'] == 3, 'actual UR loss marker differs')
            try: subscriber.parse_calls(text, entry['pid'])
            except RuntimeError as error:
                require('record budget exhausted' in str(error), 'wrong actual UR loss refusal')
            else: raise RuntimeError('incomplete actual UR capture accepted')
            sample.update(loss_markers=1, complete_capture_rejected=True)
        row['outcomes'].append(sample)
    direct_host()
    require(inputs(row['source_commit']) == row['inputs'] and subscriber.runtime_assets() == row['assets'],
            'post-run direct input mismatch')
    row.update(stage='direct-host-checked', passed=True, complete=True)


def main(action):
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gen.guards.no_model_owners([])
        container = gen.committed(RESULT)
        direct = action.startswith('direct-')
        key, root = (DIRECT_KEY, DIRECT_ROOT) if direct else (KEY, ROOT)
        row = container['event_reconciliation'].get(key, {})
        if action in ('prepare', 'direct-prepare'):
            require(not root.exists(), 'fresh root; inspect ownership before retry')
            root.mkdir(mode=0o700)
        owned = root / (action + '-owned')
        require(not owned.exists(), 'inspect existing owner before retry')
        owner = None
        def persist():
            write(owned / 'record.json', dict(row, active=bool(owner and owner.active is not None),
                                            commands_owned=owner.commands if owner else []))
        owner = supervision.Owner(owned, persist)
        try:
            {'prepare': prepare, 'build': build, 'check': check,
             'direct-prepare': direct_prepare, 'direct-check': direct_check}[action](row, owner)
        except BaseException as error:
            row.update(stage='failed', passed=False, complete=False, error=repr(error))
        finally:
            row['active'] = owner.active is not None
            row['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            row['passed'] = row.get('passed', False) and row['ownership_closed']
            row['complete'] = row.get('complete', False) and row['passed']
            write(owned / 'record.json', dict(row, commands_owned=owner.commands))
            container['event_reconciliation'][key] = row
            write(RESULT, container, key)
        print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'complete', 'ownership_closed', 'error')}), flush=True)
        return int(row['stage'] == 'failed' or not row['ownership_closed'])


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'check', 'direct-prepare', 'direct-check'),
            'prepare|build|check|direct-prepare|direct-check')
    sys.exit(main(sys.argv[1]))
