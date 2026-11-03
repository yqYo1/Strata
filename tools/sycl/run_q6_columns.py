#!/usr/bin/env python3
"""Audit the completed Q6 ELF, then run the separately pinned component plan.

Repository LICENSE. The original V2 build-audit refusal stays failed. Compiler
temporary host objects and device wrappers are generated link outputs, not
permanent source/library inputs. No source recompilation or engine adoption.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import measure_q6_columns as q

P, B, ROOT, RESULT = q.P, q.B, q.ROOT, q.RESULT
KEY = 'q6_columns_component_v4'
STATE = ROOT / 'record-v4.json'
SELF = Path(__file__).resolve()
require, digest, write = q.require, q.digest, q.write


def save(row):
    write(STATE, row)
    data = json.loads(RESULT.read_text())
    data[KEY] = row
    write(RESULT, data)


def original():
    row = json.loads((ROOT / 'record.json').read_text())
    require(row['stage'] == 'failed-build' and row['ownership_closed'] and
        row['error'].startswith("RuntimeError('nonfile link input: " + str(ROOT / 'tmp/icpx-')),
        'only the original compiler-temporary audit refusal may be reused')
    q.source_current(row)
    return row


def current(row):
    native = original()
    require(digest(ROOT / 'record.json') == row['original_build']['record_sha256'],
        'original failed build receipt changed')
    require(digest(q.EXE) == row['binary']['sha256'] and q.EXE.stat().st_size == row['binary']['bytes'],
        'audited final ELF changed')
    require(build_bytes(row['controller_commit']) == SELF.read_bytes() and
        hashlib.sha256(SELF.read_bytes()).hexdigest() == row['controller_sha256'],
        'commit and freeze exact run controller')
    inputs = row['actual_inputs']
    require(digest(inputs['path']) == inputs['sha256'], 'actual input receipt changed')
    actual = json.loads(Path(inputs['path']).read_text())
    expected = q.allowed_inputs(native)
    for path, sha in actual.items():
        require(path in expected and sha == expected[path] and digest(path) == sha,
            'compiled/linked input changed: ' + path)
    return native


def build_bytes(revision):
    return q.build.git(P, 'show', revision + ':' + str(SELF.relative_to(P)))


def validation_log(text, pid, loader, traced):
    proof = q.validation.preload_evidence(text, pid)
    require(proof['loader'] == loader, 'wrong validation loader')
    # The qualified preload intentionally submits one null-platform query to
    # prove parameter validation before main. Full API tracing logs its expected
    # INVALID_NULL_HANDLE; every other diagnostic must still fail admission.
    boundary = text.index('strata UR validation: pid=' + str(pid) + ' phase=before-main ')
    pattern = (r'(?m)^[ \t]*<--- urPlatformGetApiVersion\(\.hPlatform = nullptr, '
        r'\.pVersion = 0x[0-9a-f]+(?: \([^\r\n]*\))?\) '
        r'-> UR_RESULT_ERROR_INVALID_NULL_HANDLE;[ \t]*$')
    expected = list(re.finditer(pattern, text[:boundary]))
    require(len(expected) == int(traced), 'wrong expected negative-probe trace count')
    if expected:
        match = expected[0]
        text = text[:match.start()] + text[match.end():]
    require(not any(word in text for word in
        ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', '[WARNING]', '[ERROR]')),
        'unexpected validation/provider diagnostic')
    return proof


def audit():
    require(KEY not in q.committed() and not STATE.exists(), 'fresh post-build audit only')
    previous = q.committed().get('q6_columns_component_v3', {})
    require(previous.get('stage') == 'failed-smoke' and
        previous.get('error') == "RuntimeError('validation/provider diagnostic')" and
        previous.get('ownership_closed') is True, 'preserve the original traced-probe refusal')
    native = original()
    revision = q.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    require(build_bytes(revision) == SELF.read_bytes(), 'commit audit controller before use')
    expected = q.allowed_inputs(native)
    actual = {}
    for dep in sorted((ROOT / 'objects').glob('*.d')):
        for name in q.shlex.split(dep.read_text().replace('\\\n', ' ').split(':', 1)[1]):
            path = str(Path(name).resolve())
            require(path in expected and digest(path) == expected[path], 'uncovered compiler input: ' + path)
            actual[path] = expected[path]
    generated = []
    linked = {}
    for line in (ROOT / 'build-commands/link.stdout').read_text().splitlines():
        if not line.startswith('/'):
            continue
        path = Path(line)
        if path.parent.parent == ROOT / 'tmp' and re.fullmatch('icpx-[0-9a-f]+', path.parent.name):
            require(re.fullmatch(r'(?:[0-9]+-[0-9a-f]+|sycl_q6_columns_bench-wrapper-[0-9a-f]+)\.o', path.name)
                and not path.exists(), 'unknown compiler-generated link input')
            generated.append(path.name)
        else:
            require(path.is_file(), 'missing permanent link input: ' + line)
            name = str(path.resolve())
            require(name in expected and digest(name) == expected[name], 'uncovered link input: ' + name)
            linked[name] = expected[name]
    indices = {int(name.split('-', 1)[0]) for name in generated if name[0].isdigit()}
    require(indices == set(range(len(q.SOURCES))) and len(generated) == len(q.SOURCES) + 1 and
        sum(name.startswith('sycl_q6_columns_bench-wrapper-') for name in generated) == 1 and
        len(actual) >= 300 and len(linked) >= 10, 'incomplete source/link accounting')
    actual.update(linked)
    inputs_path = ROOT / 'actual-inputs-v3.json'
    require(not inputs_path.exists(), 'do not overwrite input proof')
    write(inputs_path, actual)
    row = dict(stage='compiled-audited', source_commit=native['source_commit'],
        controller_commit=revision, controller_sha256=digest(SELF),
        original_build=dict(key=q.KEY, record_sha256=digest(ROOT / 'record.json'),
            original_stage='failed-build', original_failure_preserved=True),
        binary=dict(path=str(q.EXE), sha256=digest(q.EXE), bytes=q.EXE.stat().st_size),
        build_audit=dict(passed=True, project_sources=len(native['project_inputs']['files']),
            actual_compiler_and_library_inputs=len(actual), permanent_link_inputs=len(linked),
            generated_host_objects=len(indices), generated_device_wrappers=1,
            limits='DPC++ transient link objects are generated by the pinned successful linker invocation;their deleted bytes are not retained or claimed as permanent source inputs.'),
        actual_inputs=dict(path=str(inputs_path), sha256=digest(inputs_path)),
        plan=native['plan'], processes=[], passed=False, complete=False, active=False, ownership_closed=True,
        retention=native['retention'])
    row['plan'] = dict(row['plan'], validation_trace_contract='Full-trace smoke requires exactly one qualified preload null-platform INVALID_NULL_HANDLE before the first before-main proof;all other UR/ZE warnings/errors reject. Quiet speed requires no error trace. Original V3 failure remains failed;new smoke evidence required.')
    current(row)
    save(row)
    print('Audited unchanged final ELF and original source/recipe;original V2 refusal remains failed.', flush=True)


def run(smoke):
    data = q.committed()
    row = json.loads(STATE.read_text())
    require(row == data[KEY] and row['stage'] == ('compiled-audited' if smoke else 'smoke-passed'),
        'commit frozen component progression before run')
    native = current(row)
    environment = q.native_environment(native, smoke)
    owner = None

    def persist():
        if owner is not None:
            write(ROOT / 'v4-ownership.json', dict(active=owner.active is not None,
                commands=[{key: entry[key] for key in ('pid', 'direct_identity', 'exit_code',
                    'observation_complete', 'session_empty', 'direct_child_reaped', 'errors') if key in entry}
                    for entry in owner.commands]))

    owner = q.supervision.Owner(ROOT / ('v4-smoke' if smoke else 'v4-speed'), persist)
    deadline = time.monotonic() + 2400
    try:
        for process in range(1 if smoke else 3):
            print('Q6 component', 'smoke' if smoke else 'speed', process, flush=True)
            cursor = q.journal_cursor()
            remaining = deadline - time.monotonic()
            require(remaining > 0, 'whole component execution deadline')
            entry, out, err = owner.run('process-' + str(process),
                [str(q.EXE), '--smoke' if smoke else '--speed'], environment, ROOT,
                **dict(native['budgets'], wall=min(remaining, native['budgets']['wall'])))
            require(q.supervision.closed(entry) and entry['exit_code'] == 0,
                'native component failed: ' + (out.read_text() + err.read_text())[-4000:])
            text, errors = out.read_text(), err.read_text()
            require('Q6_COLUMNS_PASS mode=' + ('smoke' if smoke else 'speed') in text,
                'missing normal completion')
            proof = validation_log(errors, entry['pid'],
                native['native_reference']['validation']['loader']['path'], smoke)
            faults = q.fresh_faults(cursor)
            require(not faults, 'new GPU fault')
            current(row)
            result = dict(native_exit_code=0, ownership_closed=True, validation=proof,
                passed_cases=text.count('Q6_CASE_PASS '), fresh_faults=faults)
            require(result['passed_cases'] == (32 if smoke else 35), 'missing correctness case')
            if smoke:
                row.update(stage='smoke-passed', smoke=result)
            else:
                result['samples'] = q.parse_speed(text)
                row['processes'].append(result)
            save(row)
        if not smoke:
            shapes = []
            for i in range(35):
                samples = [p['samples'][i] for p in row['processes']]
                ratios = [s['median_time_ratio'] for s in samples]
                shapes.append(dict(shape=samples[0]['shape'], process_time_ratios=ratios,
                    all_three_faster=all(r < 1 for r in ratios)))
            row.update(stage='closed', passed=True, complete=True,
                decision=dict(shapes=shapes, inference_adopted=False,
                    next='Only individually faster shapes may enter a new explicitly decode-scoped model plan;preserve prefill/MTP and validate long-generation,quality and safety.'))
    except BaseException as error:
        row.update(stage='failed-smoke' if smoke else 'failed-speed', error=repr(error))
        raise
    finally:
        row['ownership_closed'] = owner.active is None and all(q.supervision.closed(e) for e in owner.commands)
        persist()
        save(row)


def main():
    require(len(sys.argv) == 2 and sys.argv[1] in ('audit-build', 'smoke', 'speed'),
        'expected audit-build/smoke/speed')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        q.guards.no_model_owners([])
        if sys.argv[1] == 'audit-build': audit()
        else: run(sys.argv[1] == 'smoke')


if __name__ == '__main__':
    main()
