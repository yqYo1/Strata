#!/usr/bin/env python3
"""Bounded host contract for the future positive-swap diagnostic.

Main-authored; repository LICENSE. No SYCL compilation, model or GPU execution.
Preparation/checking is allowed only at a closed owned-process boundary; all
frozen MTP comparison inputs and native/fallback outputs remain unchanged.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

sys.dont_write_bytecode = True
import check_mtp_width_model as width
import device_fp64_probe as dependencies

P, B, RESULT, base = width.P, width.B, width.RESULT, width.base
SELF = Path(__file__).resolve()
ROOT = B / 'adaptive-residency-audit-host-v1'
KEY = 'adaptive_residency_audit_host_v1'
NATIVE_RECORD_SHA = '07c38cc31954f83c8296a81746d0f223e7b609d60fdc8ce5a961c15e5333addb'
IMAGE_SHA = '2648a8683d79ee9b327c3c9a5ebeaa37ebdf51a055a52d91f90d21a1bf402838'
CPP = ('include/strata/core/adaptive_residency_audit.hpp',
       'include/strata/core/expert_cache.hpp', 'tests/core/adaptive_residency_audit_test.cpp')
require, digest, write = width.require, width.digest, width.write


def compact(row):
    value = {k: v for k, v in row.items() if k not in ('sources', 'commands')}
    value['source_inputs'] = dict(count=len(row['sources']),
        sha256=width.signature(row['sources']), project={n: row['sources'][n] for n in CPP})
    return value


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = compact(row)
    write(RESULT, data)


def sources(revision):
    values = base.controls(revision, controller=SELF)
    for name in CPP:
        raw = base.basis.build.git(P, 'show', revision + ':' + name)
        require((P / name).read_bytes() == raw, 'commit exact host input: ' + name)
        values[name] = hashlib.sha256(raw).hexdigest()
    return values


def boundary():
    cohort = json.loads((width.SPEED_ROOT / 'record.json').read_text())
    require(base.committed(RESULT)[width.SPEED_KEY] ==
            width.generation_context().compact(cohort) and
            not cohort['active'] and cohort['ownership_closed'] and
            4 <= len(cohort['processes']) <= 6 and
            all(v['passed'] and v['complete'] and v['ownership_closed'] and
                v['native_exit_code'] == 0 and v['fresh_faults'] == []
                for v in cohort['processes']),
            'close and commit process3 before this serialized host check')
    base.guards.no_model_owners([])


def image():
    record = width.short.build.ROOT / 'record.json'
    require(digest(record) == NATIVE_RECORD_SHA, 'native dependency record changed')
    pin = json.loads(record.read_text())['toolchain_image']
    require(pin['sha256'] == IMAGE_SHA and digest(pin['path']) == IMAGE_SHA,
            'original complete compiler image changed')
    value = json.loads(Path(pin['path']).read_text())
    require(dependencies.toolchain() == value, 'compiler/package bytes or membership changed')
    return pin, value


def prepare():
    boundary()
    require(not ROOT.exists() and KEY not in base.committed(RESULT), 'fresh host contract only')
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = sources(revision)
    toolchain, _ = image()
    ROOT.mkdir()
    for name in ('tmp', 'cache'):
        (ROOT / name).mkdir()
    compiler = '/opt/intel/oneapi/compiler/2026.2/bin/icpx'
    exe = str(ROOT / 'audit_test')
    row = dict(stage='prepared', source_commit=revision, sources=pins,
        toolchain_image=toolchain, native_dependency_record_sha256=NATIVE_RECORD_SHA,
        compiler=dict(path=compiler, sha256=digest(compiler)), executable=exe,
        command=[compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
            '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13',
            '-I' + str(P / 'include'), '-MD', '-MT', exe, '-MF', str(ROOT / 'headers.d'),
            '-Wl,-t', str(P / CPP[2]), '-o', exe],
        environment=dict(PATH='/opt/intel/oneapi/compiler/2026.2/bin:/usr/bin:/bin',
            LANG='C', LC_ALL='C', TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache')),
        budgets=dict(compile=dict(wall=90, cpu=120, rss_cap=2 << 30,
            text_cap=256 << 10, file_cap=4 << 20, total_cap=8 << 20),
            test=dict(wall=15, cpu=15, rss_cap=512 << 20,
            text_cap=64 << 10, file_cap=4 << 20, total_cap=8 << 20)),
        expected='adaptive_residency_audit cases=41 status=PASS',
        passed=False, complete=False, active=False, ownership_closed=True,
        GPU_executed=False, model_executed=False, inference_adopted=False,
        plan='Commit before check. One host compile,ELF dependency check and fixture;normal0 and closed owners. Actual headers/link inputs must match the prebuild complete compiler/package image plus the three exact project files. No retry,source/image refresh or GPU/SYCL work. This tests host identity/payload/fence assertions,not native callbacks,positive swaps,queue lifetime,speed or full context.',
        retention='Commit compact source/recipe/case/closure evidence;then retire this one-use executable,compiler temporaries and successful full output after exact Git source/recipe recovery. Native and fallback generations remain untouched.')
    save(row)
    print('Prepared host-only adaptive identity contract;commit before check.', flush=True)


def current(row):
    boundary()
    require(base.committed(RESULT)[KEY] == compact(row) and
            sources(row['source_commit']) == row['sources'], 'host plan or committed inputs changed')
    pin, value = image()
    require(pin == row['toolchain_image'] and
            digest(row['compiler']['path']) == row['compiler']['sha256'], 'host compiler changed')
    return value


def closure(row, toolchain, text):
    allowed = {v['realpath']: v['sha256'] for v in toolchain['files'].values() if v['kind'] == 'file'}
    allowed.update({str((P / n).resolve()): row['sources'][n] for n in CPP})
    tokens = shlex.split((ROOT / 'headers.d').read_text().replace('\\\n', ''))
    require(tokens and tokens[0] == row['executable'] + ':', 'actual dependency target differs')
    links = [v.strip() for v in text.splitlines() if v.startswith('/')]
    require(tokens[1:] and links, 'actual headers/link trace missing')
    actual, derived = {}, 0
    for name in dict.fromkeys(tokens[1:] + links):
        path = Path(name)
        require(path.is_absolute(), 'relative compiler input: ' + name)
        if path.is_relative_to(ROOT / 'tmp'):
            derived += 1
            continue
        real = str(path.resolve(strict=True))
        require(real in allowed and digest(path) == allowed[real], 'unqualified compiler input: ' + name)
        actual[real] = allowed[real]
    require(all(str((P / n).resolve()) in actual for n in CPP) and
            any(Path(n).name.startswith('libc.') for n in links) and
            any(Path(n).name.startswith('libstdc++.') for n in links), 'project/host link closure missing')
    return dict(sha256=width.signature(actual), files=len(actual), headers=len(tokens) - 1,
        link_inputs=len(links), derived_private_inputs=derived)


def check():
    row = json.loads((ROOT / 'record.json').read_text())
    require(row['stage'] == 'prepared' and not (ROOT / 'execution.json').exists(),
            'one attempt only;inspect existing host ownership')
    outcome, owner = dict(passed=False, complete=False), None
    started = time.monotonic()

    def persist():
        write(ROOT / 'execution.json', dict(outcome,
            active=bool(owner and owner.active is not None),
            commands=owner.commands if owner else []))

    try:
        with (ROOT / 'execution.json').open('x') as stream:
            stream.write(json.dumps(dict(outcome, active=False, commands=[])) + '\n')
        toolchain = current(row)
        owner = base.supervision.Owner(ROOT / 'owned', persist)
        command, stdout, _ = owner.run('compile', row['command'], row['environment'], ROOT,
                                      **row['budgets']['compile'])
        require(base.supervision.closed(command) and command['exit_code'] == 0, 'owned host compile failed')
        outcome['compiler_inputs'] = closure(row, toolchain, stdout.read_text())
        outcome['executable_sha256'] = digest(row['executable'])
        require(time.monotonic() - started < 480, 'whole host deadline')
        command, stdout, _ = owner.run('needed', ['/usr/bin/readelf', '-d', row['executable']],
            row['environment'], ROOT, **row['budgets']['test'])
        require(base.supervision.closed(command) and command['exit_code'] == 0 and
                not any(n in stdout.read_text().lower() for n in
                    ('libsycl', 'libur_', 'libze_', 'libopencl')), 'host fixture linked a GPU provider')
        command, stdout, stderr = owner.run('test', [row['executable']], row['environment'], ROOT,
                                           **row['budgets']['test'])
        require(base.supervision.closed(command) and command['exit_code'] == 0 and
                stdout.read_text().strip() == row['expected'] and not stderr.read_text(),
                'host contract failed or returned incomplete output')
        outcome['receipt'] = row['expected']
        current(row)
        require(time.monotonic() - started < 480, 'whole host deadline')
        outcome.update(passed=True, complete=True)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        row.update(active=bool(owner and owner.active is not None),
            ownership_closed=owner is None or (owner.active is None and
                all(base.supervision.closed(c) for c in owner.commands)))
        outcome['passed'] = bool(outcome['passed'] and row['ownership_closed'])
        row.update(stage='closed' if outcome['passed'] else 'failed',
            passed=outcome['passed'], complete=outcome['complete'], outcome=outcome,
            commands=owner.commands if owner else [])
        persist()
        save(row)
    require(row['passed'], str(outcome))
    print('Adaptive identity host contract passed;native integration remains unqualified.', flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['check']), 'use prepare|check')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (prepare if sys.argv[1] == 'prepare' else check)()
