#!/usr/bin/env python3
"""Build the finite collector fixture against the exact current SYCL adapter.

Main-authored; repository LICENSE. No inference engine or GPU execution.
The frozen width cohort and current-ELF active collector check must close first.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

sys.dont_write_bytecode = True
import adaptive_residency_host as host
import q6_half_collector as collector

width = collector.width
P, B, RESULT, base = width.P, width.B, width.RESULT, width.base
SELF = Path(__file__).resolve()
ROOT = B / 'compat-collector-window-v1'
KEY = 'compat_collector_window_build_v1'
CPP = 'tests/sycl/compat_window.cpp'
RUNTIME = ('src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp')
require, digest, write = width.require, width.digest, width.write


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('sources', 'commands')}
    out['source_inputs'] = dict(count=len(row['sources']), sha256=width.signature(row['sources']),
                                fixture=row['sources'][CPP])
    return out


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def sources(revision):
    values = base.controls(revision, controller=SELF)
    raw = base.basis.build.git(P, 'show', revision + ':' + CPP)
    require((P / CPP).read_bytes() == raw, 'commit exact compatibility-window fixture')
    values[CPP] = hashlib.sha256(raw).hexdigest()
    return values


def basis():
    data = base.committed(RESULT)
    cohort = json.loads((width.SPEED_ROOT / 'record.json').read_text())
    require(data[width.SPEED_KEY] == width.generation_context().compact(cohort) and
            cohort['stage'] == 'closed' and cohort['complete'] and cohort['passed'] and
            cohort['ownership_closed'] and not cohort['active'] and len(cohort['processes']) == 6 and
            all(v['passed'] and v['complete'] and v['ownership_closed'] and
                v['native_exit_code'] == 0 and not v['fresh_faults'] for v in cohort['processes']),
            'close and commit all six frozen width processes first')
    active = json.loads((collector.ROOT / 'record.json').read_text())
    require(data[collector.KEY] == collector.context().compact(active) and
            active['stage'] == 'closed' and active['passed'] and active['complete'] and
            active['ownership_closed'] and not active['active'],
            'close and commit the current-ELF active62-kernel control first')
    record = width.short.build.ROOT / 'record.json'
    require(digest(record) == host.NATIVE_RECORD_SHA, 'original native build record changed')
    native = json.loads(record.read_text())
    old, _, values = width.short.build.projection()
    require(width.short.build.pins(values) == native['native_sources'] and
            width.short.build.recipe(old) == native['recipe'] and
            all((width.short.build.ROOT / 'source' / n).read_bytes() == raw for n, raw in values.items()),
            'recover the exact historical567-source composition and recipe')
    require(active['native_source_commit'] == native['source_commit'] and
            active['binary'] == native['binaries']['sycl_submission_test'] and
            active['engine'] == native['binaries']['strata'] and
            all(digest(v['path']) == v['sha256'] for v in native['binaries'].values()),
            'active collector proof and current native executables differ')
    base.guards.no_model_owners([])
    return native


def recipe():
    compiler = '/opt/intel/oneapi/compiler/2026.2/bin/icpx'
    native = width.short.build.ROOT / 'source'
    flags = ['-fsycl', '-fsycl-id-queries-range=size_t', '-std=c++20', '-O3', '-DNDEBUG',
        '-DSTRATA_ENABLE_SYCL=1', '-DSTRATA_CPU_SERVICE_PROFILE=1', '-Wall', '-Wextra',
        '-Werror', '-Wno-error=deprecated-declarations',
        '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13',
        '-I' + str(native / 'include'), '-I' + str(native / 'include/strata/sycl/compat')]
    inputs = [str(native / n) for n in RUNTIME] + [str(P / CPP)]
    objects = [str(ROOT / ('unit' + str(i) + '.o')) for i in range(len(inputs))]
    commands = [[compiler, *flags, '-MD', '-MT', obj, '-MF', obj + '.d', '-c', src, '-o', obj]
                for src, obj in zip(inputs, objects)]
    commands.append([compiler, '-fsycl', '-fsycl-id-queries-range=size_t',
        '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13', '-Wl,-t', *objects,
        '-o', str(ROOT / 'compat_window')])
    return commands


def prepare():
    native = basis()
    require(not ROOT.exists() and KEY not in base.committed(RESULT), 'fresh fixture build only')
    revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
    pins = sources(revision)
    image, _ = host.image()
    ROOT.mkdir()
    for name in ('tmp', 'cache'): (ROOT / name).mkdir()
    env = dict(native['recipe']['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    require('LD_PRELOAD' not in env, 'compiler must not preload validation or collector')
    row = dict(stage='prepared', source_commit=revision, sources=pins,
        native_source_commit=native['source_commit'], native_record_sha256=host.NATIVE_RECORD_SHA,
        native_composition_sha256=width.signature(native['native_sources']),
        toolchain_image=image, recipe=recipe(), environment=env,
        compiler_sha256=digest(recipe()[0][0]), executable=str(ROOT / 'compat_window'),
        budget=dict(wall=180, cpu=240, rss_cap=4 << 30, text_cap=1 << 20,
                    file_cap=32 << 20, total_cap=128 << 20), whole_wall_s=900,
        passed=False, complete=False, active=False, ownership_closed=True,
        GPU_executed=False, model_executed=False, collector_qualified=False,
        plan='Commit before build. Four serial SYCL compilations and one device/host link;all normal0 and closed. Pin complete installed compiler/package image,actual headers and host-link inputs. Use the exact selected runtime/adapter/command-profile sources and a new137-line fixture;no engine or collector rebuild. Do not execute this ELF during the build action.',
        next='Commit the built result,then a separate prospective smoke/prefix-only/prefix-window plan using strict before-main validation and the existing indexed collector. Expected4/10 named kernels;normal20-kernel numerical/guard execution,unique host flows,raw-clock gap exclusion,finite output and fresh-fault/closure checks. No model attribution follows from build success.',
        retention='One-use diagnostic executable/objects/dependencies outside Git. Retire after the bounded current-runtime collector-window decision and original source/recipe recovery;current cb7 and552b engine generations remain unchanged.')
    save(row)
    print('Prepared exact-current-runtime collector fixture build;commit before build.', flush=True)


def current(row):
    native = basis()
    require(base.committed(RESULT)[KEY] == compact(row) and sources(row['source_commit']) == row['sources'] and
            row['recipe'] == recipe() and row['native_composition_sha256'] == width.signature(native['native_sources']) and
            digest(row['recipe'][0][0]) == row['compiler_sha256'], 'fixture source/recipe changed')
    pin, image = host.image()
    require(pin == row['toolchain_image'], 'complete compiler image changed')
    return native, image


def closure(row, native, image, text):
    allowed = {v['realpath']: v['sha256'] for v in image['files'].values() if v['kind'] == 'file'}
    allowed.update({str((width.short.build.ROOT / 'source' / n).resolve()): h
                    for n, h in native['native_sources'].items()})
    allowed[str((P / CPP).resolve())] = row['sources'][CPP]
    objects = [ROOT / ('unit' + str(i) + '.o') for i in range(4)]
    allowed.update({str(p.resolve()): digest(p) for p in objects})
    names = []
    for obj in objects:
        tokens = shlex.split(Path(str(obj) + '.d').read_text().replace('\\\n', ''))
        require(tokens and tokens[0] == str(obj) + ':', 'actual dependency target differs')
        names.extend(tokens[1:])
    links = [s.strip() for s in text.splitlines() if s.startswith('/')]
    require(names and links, 'actual compiler dependency/link evidence missing')
    actual, derived = {}, 0
    for name in dict.fromkeys(names + links):
        path = Path(name); require(path.is_absolute(), 'relative compiler input: ' + name)
        if path.is_relative_to(ROOT / 'tmp'):
            derived += 1; continue
        real = str(path.resolve(strict=True))
        require(real in allowed and digest(path) == allowed[real], 'unqualified compiler input: ' + name)
        actual[real] = allowed[real]
    require(all(str((width.short.build.ROOT / 'source' / n).resolve()) in actual for n in RUNTIME) and
            str((P / CPP).resolve()) in actual and
            all(str(p.resolve()) in actual for p in objects) and
            any(Path(n).name.startswith('libsycl.') for n in links), 'selected runtime/SYCL input closure missing')
    return dict(sha256=width.signature(actual), files=len(actual), dependency_entries=len(names),
                host_link_inputs=len(links), derived_private_inputs=derived)


def build():
    row = json.loads((ROOT / 'record.json').read_text())
    require(row['stage'] == 'prepared' and not (ROOT / 'execution.json').exists(), 'one build attempt only')
    owner, outcome = None, dict(passed=False, complete=False)
    started = time.monotonic()
    def persist():
        write(ROOT / 'execution.json', dict(outcome, active=bool(owner and owner.active is not None),
                                           commands=owner.commands if owner else []))
    try:
        with (ROOT / 'execution.json').open('x') as stream: stream.write('{}\n')
        native, image = current(row)
        owner = base.supervision.Owner(ROOT / 'owned', persist)
        for i, command in enumerate(row['recipe']):
            require(time.monotonic() - started < row['whole_wall_s'], 'whole fixture build deadline')
            entry, stdout, _ = owner.run('compile' + str(i), command, row['environment'], ROOT, **row['budget'])
            require(base.supervision.closed(entry) and entry['exit_code'] == 0, 'fixture compile/link failed')
        outcome['compiler_inputs'] = closure(row, native, image, stdout.read_text())
        outcome['binary'] = dict(path=row['executable'], sha256=digest(row['executable']),
                                 bytes=Path(row['executable']).stat().st_size)
        current(row)
        require(time.monotonic() - started < row['whole_wall_s'], 'whole fixture build deadline')
        outcome.update(passed=True, complete=True)
    except BaseException as error:
        outcome['error'] = repr(error)
    finally:
        row.update(active=bool(owner and owner.active is not None), ownership_closed=owner is None or
            (owner.active is None and all(base.supervision.closed(c) for c in owner.commands)))
        outcome['passed'] = bool(outcome['passed'] and row['ownership_closed'])
        row.update(stage='built' if outcome['passed'] else 'failed', passed=outcome['passed'],
                   complete=outcome['complete'], outcome=outcome, commands=owner.commands if owner else [])
        persist(); save(row)
    require(row['passed'], str(outcome))
    print('Collector-window fixture built;device and collector qualification remain pending.', flush=True)


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['build']), 'use prepare|build')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        (prepare if sys.argv[1] == 'prepare' else build)()
