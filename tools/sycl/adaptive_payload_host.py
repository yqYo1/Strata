#!/usr/bin/env python3
"""Qualify separate adaptive expert/cycle limits without changing native inputs.

Main-authored; repository LICENSE. Host compilation and fixtures only. Run at
a closed model boundary. The current public299218 header remains unchanged.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

sys.dont_write_bytecode = True
import adaptive_payload_sources as source
import adaptive_residency_host as host

P, B, RESULT, base = host.P, host.B, host.RESULT, host.base
ROOT, KEY = B / 'adaptive-payload-audit-host-v1', 'adaptive_payload_audit_host_v1'
SELF = Path(__file__).resolve()
EXPERT = 'include/strata/core/expert_cache.hpp'
BUDGET = 'tests/core/adaptive_payload_budget_test.cpp'
PROJECT = (source.HEADER, source.FIXTURE, EXPERT, BUDGET)
require, digest, write = host.require, host.digest, host.write


def blob(revision, name):
    return base.basis.build.git(P, 'show', revision + ':' + name)


def sources(revision):
    pins = base.controls(revision, controller=SELF)
    for name in PROJECT:
        raw = blob(revision, name)
        require((P / name).read_bytes() == raw, 'commit host input: ' + name)
        pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def generated(revision):
    original = {n: blob(source.HEADER_SOURCE, n) for n in PROJECT[:3]}
    require(all((P / n).read_bytes() == raw for n, raw in original.items()),
            'preserve original public header/fixture/cache inputs')
    original[source.HEADER] = source.header(original[source.HEADER])
    original[source.FIXTURE] = source.legacy_fixture(original[source.FIXTURE])
    original[BUDGET] = blob(revision, BUDGET)
    return original


def compact(row):
    value = {k: v for k, v in row.items() if k not in ('sources', 'commands')}
    value['source_inputs'] = dict(count=len(row['sources']), sha256=host.width.signature(row['sources']),
        project={n: row['sources'][n] for n in PROJECT})
    return value


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def prepare():
    require(not ROOT.exists() and KEY not in base.committed(RESULT), 'one fresh host attempt')
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins, values = sources(revision), generated(revision)
    image, _ = host.image()
    compiler = '/opt/intel/oneapi/compiler/2026.2/bin/icpx'
    row = dict(stage='prepared', source_commit=revision, sources=pins, generated={}, cases=[],
        toolchain_image=image, compiler=dict(path=compiler, sha256=digest(compiler)),
        original_header_commit=source.HEADER_SOURCE, original_header_sha256=source.HEADER_SHA,
        expert_limit_bytes=source.EXPERT_BYTES, cycle_limit_bytes=source.CYCLE_BYTES,
        environment=dict(PATH='/opt/intel/oneapi/compiler/2026.2/bin:/usr/bin:/bin', LANG='C', LC_ALL='C',
            TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache')),
        budgets=dict(compile=dict(wall=90, cpu=120, rss_cap=2 << 30, text_cap=256 << 10,
            file_cap=4 << 20, total_cap=8 << 20),
            test=dict(wall=15, cpu=15, rss_cap=512 << 20, text_cap=64 << 10,
                file_cap=4 << 20, total_cap=8 << 20)),
        whole_wall_seconds=600, whole_output_bytes=32 << 20,
        passed=False, complete=False, active=False, ownership_closed=True,
        GPU_executed=False, model_executed=False, inference_adopted=False,
        plan='Commit source,projection,compiler image and both recipes before execution. Compile '
            'legacy41 with only its obsolete80MiB overflow replaced by five valid64MiB extents;'
            'compile separate16-case bounds/order fixture. Require normal0,exact receipts,quiet '
            'stderr,actual project/header/link closure and no GPU provider dependency. Reject '
            'source/image/input mismatch;no retry or pin refresh. Model32 publication/dispatch/'
            'post-positive loan,second-record refusal and clean performance remain pending.',
        retention='Retain source/recipe,57-case outcome and actual-input signatures. Retire the '
            'two one-use ELFs and successful output after pushed results/source recovery. '
            'The current552b/429b generations and their15outputs remain unchanged.')
    ROOT.mkdir(mode=0o700)
    for n in ('source', 'tmp', 'cache'):(ROOT / n).mkdir(mode=0o700)
    for name, raw in values.items():
        path = ROOT / 'source' / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        row['generated'][name] = hashlib.sha256(raw).hexdigest()
    for name, fixture, expected in (('legacy', source.FIXTURE,
            'adaptive_residency_audit cases=41 status=PASS'),
            ('budget', BUDGET, 'adaptive_payload_budget cases=16 status=PASS')):
        root = ROOT / name; root.mkdir(mode=0o700); exe = str(root / 'audit_test')
        row['cases'].append(dict(name=name, fixture=fixture, executable=exe, expected=expected,
            command=[compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13', '-I' + str(ROOT / 'source/include'),
                '-MD', '-MT', exe, '-MF', str(root / 'headers.d'), '-Wl,-t',
                str(ROOT / 'source' / fixture), '-o', exe]))
    save(row)
    print('Prepared separate64/256MiB host limits;commit before compilation.', flush=True)


def current(row):
    require(base.committed(RESULT)[KEY] == compact(row) and sources(row['source_commit']) == row['sources'],
            'commit the exact host plan/source')
    values = generated(row['source_commit'])
    require({n: hashlib.sha256(v).hexdigest() for n, v in values.items()} == row['generated'] and
        all(not (ROOT / 'source' / n).is_symlink() and (ROOT / 'source' / n).read_bytes() == v
            for n, v in values.items()), 'private projection changed')
    pin, image = host.image()
    require(pin == row['toolchain_image'] and digest(row['compiler']['path']) == row['compiler']['sha256'],
            'qualified host compiler image changed')
    require(base.supervision.artifact_bytes(ROOT) <= row['whole_output_bytes'], 'whole host output budget')
    return image


def closure(row, case, image, text):
    allowed = {v['realpath']: v['sha256'] for v in image['files'].values() if v['kind'] == 'file'}
    allowed.update({str((ROOT / 'source' / n).resolve()): h for n, h in row['generated'].items()})
    tokens = shlex.split((ROOT / case['name'] / 'headers.d').read_text().replace('\\\n', ''))
    links = [v.strip() for v in text.splitlines() if v.startswith('/')]
    require(tokens and tokens[0] == case['executable'] + ':' and tokens[1:] and links,
            'actual host headers/link trace missing')
    actual, derived = {}, 0
    for name in dict.fromkeys(tokens[1:] + links):
        path = Path(name); require(path.is_absolute(), 'relative host compiler input')
        if path.is_relative_to(ROOT / 'tmp'):
            derived += 1; continue
        real = str(path.resolve(strict=True))
        require(real in allowed and digest(path) == allowed[real], 'unqualified host input: ' + name)
        actual[real] = allowed[real]
    require(all(str((ROOT / 'source' / n).resolve()) in actual for n in
        (source.HEADER, EXPERT, case['fixture'])) and
        any(Path(n).name.startswith('libc.') for n in links) and
        any(Path(n).name.startswith('libstdc++.') for n in links), 'host project/link closure missing')
    return dict(files=len(actual), sha256=host.width.signature(actual), headers=len(tokens) - 1,
        link_inputs=len(links), derived_private_inputs=derived)


def check():
    row = json.loads((ROOT / 'record.json').read_text())
    require(row['stage'] == 'prepared' and not (ROOT / 'execution.json').exists(), 'one host attempt only')
    result, owner, started = dict(passed=False, complete=False, cases=[]), None, time.monotonic()
    def persist():
        write(ROOT / 'execution.json', dict(result, active=bool(owner and owner.active),
            commands=owner.commands if owner else []))
    try:
        with (ROOT / 'execution.json').open('x') as stream: stream.write('{}\n')
        image = current(row); owner = base.supervision.Owner(ROOT / 'owned', persist)
        for case in row['cases']:
            command, out, _ = owner.run(case['name'] + '-compile', case['command'], row['environment'],
                ROOT, **row['budgets']['compile'])
            require(base.supervision.closed(command) and command['exit_code'] == 0, 'host compile failed')
            evidence = closure(row, case, image, out.read_text())
            sha = digest(case['executable'])
            command, out, _ = owner.run(case['name'] + '-needed', ['/usr/bin/readelf', '-d', case['executable']],
                row['environment'], ROOT, **row['budgets']['test'])
            require(base.supervision.closed(command) and command['exit_code'] == 0 and not any(
                n in out.read_text().lower() for n in ('libsycl', 'libur_', 'libze_', 'libopencl')),
                'host fixture linked a GPU provider')
            command, out, err = owner.run(case['name'] + '-test', [case['executable']], row['environment'],
                ROOT, **row['budgets']['test'])
            require(base.supervision.closed(command) and command['exit_code'] == 0 and
                out.read_text().strip() == case['expected'] and err.stat().st_size == 0,
                'host fixture failed or returned incomplete output')
            result['cases'].append(dict(name=case['name'], passed=True, receipt=case['expected'],
                executable_sha256=sha, compiler_inputs=evidence))
            current(row)
            require(time.monotonic() - started < row['whole_wall_seconds'], 'whole host deadline')
        result.update(passed=True, complete=True)
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        row.update(active=bool(owner and owner.active), ownership_closed=owner is None or
            owner.active is None and all(base.supervision.closed(c) for c in owner.commands))
        result['passed'] = bool(result['passed'] and row['ownership_closed'])
        row.update(stage='closed' if result['passed'] else 'failed', passed=result['passed'],
            complete=result['complete'], outcome=result, commands=owner.commands if owner else [])
        persist(); save(row)
    require(row['passed'], str(result))
    print('Separate64/256MiB host limits pass41+16cases;native32 remains unqualified.', flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['check']), 'use prepare|check')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); base.guards.no_model_owners([])
        (prepare if sys.argv[1] == 'prepare' else check)()
