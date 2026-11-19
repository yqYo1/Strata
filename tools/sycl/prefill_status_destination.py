#!/usr/bin/env python3
"""Isolate the complete four-byte production readback pair. Repository LICENSE."""

import fcntl
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import coherent_model_basis as basis
import measurement_owner as ownership
import prefill_prefix_expert_build as stage
import ur_validation as validation

P, B, RESULT = basis.P, basis.B, basis.RESULT
ROOT = B / 'prefill-status-destination-v3'
KEY = 'prefill_status_destination_v3'
FIRST = 'prefill_status_destination_v1'
SECOND = 'prefill_status_destination_v2'
SOURCE = 'tests/sycl/prefill_status_destination.cpp'
COMPILER = '/opt/intel/oneapi/compiler/2026.2/bin/icpx'
BOOT = Path('/proc/sys/kernel/random/boot_id')
ELF = ROOT / 'sycl_prefill_status_destination'
REFERENCE = 'prefill_prefix_expert_cpu_wait_v2'
ORDER = ('ordinary', 'host-usm', 'host-usm', 'ordinary', 'ordinary', 'host-usm')
SHAPES = {'ready': (512, 1.0), 'queued': (48, 10.0)}
FLAGS = ['-std=c++20', '-O3', '-DNDEBUG', '-fsycl', '-fsycl-id-queries-range=size_t',
         '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13']
BUDGET = dict(wall=180, cpu=120, rss_cap=4 << 30, text_cap=2 << 20,
              file_cap=16 << 20, total_cap=32 << 20)
require, sha, signature = basis.require, basis.sha, basis.signature


def read(path):
    return json.loads(path.read_bytes())


def save(row):
    stage.write(ROOT / 'record.json', row)
    data = read(RESULT)
    data[KEY] = {k: v for k, v in row.items() if k not in ('sources', 'dependencies', 'providers', 'environment')}
    stage.write(RESULT, data)


def platform():
    build = read(stage.STAGE_ROOT / 'record.json')
    require(validation.committed(RESULT, (stage.STAGE_BUILD,)) == stage.builder.compact(build)
            and build['stage'] == 'built' and build['passed'] and build['ownership_closed']
            and build['actual_inputs']['uncovered'] == 0 and not build['active'], 'exact qualified runtime archive required')
    observed = validation.committed(RESULT, (REFERENCE,))
    require(observed['stage'] == 'closed' and observed['passed'] and not observed['active']
            and observed['decision']['normal_route_attribution_qualified']
            and observed['binary'] == build['binaries']['strata'], 'closed current-model CPU wait attribution required')
    _, _, speed = basis.basis()
    snapshot = stage.STAGE_ROOT / 'source'
    archive = stage.STAGE_ROOT / 'build/libstrata_sycl_runtime.a'
    expected = build['actual_inputs']['produced'][str(archive)]
    require(sha(archive) == expected and sha(observed['binary']['path']) == observed['binary']['sha256']
            and observed['boot'] == BOOT.read_text().strip()
            and observed['driver'] == stage.builder.guards.driver_guard(), 'retained binary/runtime or platform changed')
    ar = str(Path('/usr/bin/ar').resolve())
    members = subprocess.check_output([ar, 't', str(archive)], text=True, timeout=10).splitlines()
    names = ('runtime.cpp.o', 'compat.cpp.o', 'command_profile.cpp.o')
    require(set(members) == set(names) and len(members) == len(names), 'unexpected archive members')
    entry = archive.parent / 'CMakeFiles/strata_sycl_runtime.dir/src/sycl'
    objects = {}
    for name in names:
        original = entry / name
        expected_object = build['actual_inputs']['objects'][str(original)]
        require(sha(original) == expected_object
                and stage.hashed({name: subprocess.check_output([ar, 'p', str(archive), name], timeout=10)})[name]
                == expected_object, 'original compiled compatibility archive member differs')
        objects[str(original)] = expected_object
    runtime_paths = [f'src/sycl/{name.removesuffix(".o")}' for name in names]
    for path, raw in stage.source.blobs(P, build['source_commit'], runtime_paths):
        require(stage.hashed({path: raw})[path] == build['source_files'][path]
                and (snapshot / path).read_bytes() == raw, 'cannot recover compatibility wrapper source')
    manifest = Path(speed['runtime_manifest']['path'])
    providers = read(manifest)
    require(sha(manifest) == speed['runtime_manifest']['sha256']
            and all(sha(path) == digest for path, digest in providers.items()), 'qualified runtime providers differ')
    probe = speed['native_validation_probe']
    env = dict(speed['environment_template'])
    old_root = env['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old_root, str(ROOT)) for k, v in env.items()
           if k not in ('LD_PRELOAD', 'STRATA_UR_PROBE_EXPECTED_LOADER', 'STRATA_TRACE')}
    require(env['ZES_ENABLE_SYSMAN'] == '0' and env['EnableDirectSubmission'] == '0'
            and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1'
            and all(v == '0' for k, v in env.items() if '_PROFILE' in k or '_AUDIT' in k),
            'exact startup safety and quiet timing settings required')
    env.update(TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    return dict(original_native=observed['binary'], native_source=build['source_commit'],
        archive=dict(path=str(archive), sha256=expected), objects=objects, ar=ar,
        system_links={str(Path(k).resolve()): v for k, v in build['actual_inputs']['links'].items()},
        observation=signature(observed), provider_sha256=signature(providers), probe=probe,
        boot=observed['boot'], driver=observed['driver']), providers, env


def dependency_paths(output):
    headers = set()
    for line in output.replace('\\\n', ' ').splitlines():
        if not line.strip() or line.startswith(('# __CLANG_OFFLOAD_BUNDLE____START__ ',
                                               '# __CLANG_OFFLOAD_BUNDLE____END__ ')):
            continue
        target, separator, inputs = line.partition(':')
        require(separator and target.endswith('.o') and inputs.strip(), 'invalid compiler dependency rule')
        paths = [Path(token) for token in shlex.split(inputs)]
        require(all(p.is_absolute() and p.is_file() for p in paths), 'invalid compiler dependency file')
        headers.update(str(p.resolve()) for p in paths)
    require(headers, 'compiler dependency rules are empty')
    return headers


def tool_paths(flags, env):
    found = {str(Path(COMPILER).resolve()), str(Path('/usr/bin/ar').resolve())}
    for tool in ('ld', 'as', 'clang-offload-wrapper', 'sycl-post-link', 'llvm-link'):
        output = subprocess.check_output([COMPILER, *flags, '-print-prog-name=' + tool],
                                         env=env, text=True, timeout=8).strip()
        path = Path(output)
        if not path.is_file():
            path = Path(COMPILER).parent / output
        if path.is_file():
            found.add(str(path.resolve()))
    return found


def prepare():
    require(not ROOT.exists() and KEY not in validation.committed(RESULT), 'one fresh component plan required')
    previous = validation.committed(RESULT, (FIRST,))
    compiled = validation.committed(RESULT, (SECOND,))
    require(previous['stage'] == 'blocked-prepare' and previous['ownership_closed'] and not previous['GPU_executed'],
            'keep the original pre-GPU dependency census refusal')
    require(compiled['stage'] == 'failed' and compiled['ownership_closed'] and not compiled['active']
            and not compiled['processes'] and compiled['compile_commands'] == 1
            and 'binary' not in compiled and 'different header set' in compiled['error'],
            'keep the original compile census refusal')
    admitted, providers, env = platform()
    revision = stage.source.git(P, 'rev-parse', 'HEAD').decode().strip()
    sources = basis.comparison.build.controls(revision, controller=Path(__file__))
    raw = dict(stage.source.blobs(P, revision, [SOURCE]))[SOURCE]
    require(raw == (P / SOURCE).read_bytes(), 'commit the exact fixture before preparation')
    sources[SOURCE] = sha(P / SOURCE)
    ROOT.mkdir(mode=0o700)
    (ROOT / 'tmp').mkdir()
    (ROOT / 'cache').mkdir()
    snapshot = stage.STAGE_ROOT / 'source'
    flags = [*FLAGS, '-I' + str(snapshot / 'include')]
    prefix = [COMPILER, *flags]
    dep = subprocess.check_output([*prefix, '-M', str(P / SOURCE)], env=env, text=True, timeout=40)
    headers = dependency_paths(dep)
    dependencies = headers | tool_paths(flags, env) | {admitted['archive']['path'], *admitted['system_links']}
    require(str(P / SOURCE) in headers and str(snapshot / 'include/strata/sycl/runtime.hpp') in headers,
            'actual fixture or runtime contract missing from source census')
    compile_command = [*prefix, '-MD', '-MF', str(ROOT / 'fixture.d'), '-c', str(P / SOURCE),
                       '-o', str(ROOT / 'fixture.o')]
    link_command = [*prefix, '-Wl,-t', str(ROOT / 'fixture.o'), admitted['archive']['path'],
                    '-lze_loader', '-o', str(ELF)]
    pinned_dependencies = {k: sha(k) for k in sorted(dependencies)}
    require(all(pinned_dependencies[k] == v for k, v in admitted['system_links'].items()),
            'one of the qualified link inputs changed')
    save(dict(stage='prepared', source_commit=revision, sources=sources,
              source_sha256=signature(sources), dependencies=pinned_dependencies,
              dependencies_sha256=signature(pinned_dependencies),
              headers=sorted(headers), provider_sha256=signature(providers), providers=providers, environment=env,
              environment_sha256=signature(env), admitted=admitted, build=[compile_command, link_command],
              earlier_refusals={FIRST: signature(previous), SECOND: signature(compiled)},
              header_eligibility='Pin the union from -M before compilation. Require the same source and necessary '
                                 'project headers in -MD and refuse any file outside that conservative union.',
              steps=[['smoke', 'ordinary'], ['smoke', 'host-usm'], *[['timing', x] for x in ORDER]],
              plan=dict(window='Each request uses the engine compatibility stream and the real memcpy+drain wrappers. '
                               'Ordinary and host USM use identical 4-byte extents, source, order and checks.',
                        scopes='Three adjacent counterbalanced pairs of independent processes. Each measures five '
                               '512-copy batches with a drained producer and five 48-copy batches with two queued '
                               'synthetic producer kernels. Keep all subintervals and compare their total.',
                        gate='At least 1 ms per ready batch and 10 ms per queued batch. Estimated clock overhead '
                             'from the same three clock stamps at most 5% per batch. Advance toward clean model '
                             'comparison only if all queued paired medians improve after one USM allocation and '
                             'ready aggregate time is no worse than 5%. Check correctness and retirement first.',
                        limits='Synthetic producer and direct wrappers only. No model throughput or resource '
                               'saturation claim. Stop first validation, ownership, timing-floor, budget or fault failure.',
                        retention='One small fixture linked to the retained archive. Retire after the placement decision.'),
              budgets=BUDGET, processes=[], active=False, ownership_closed=True, complete=False, passed=False))
    print('Committed recipe required before fixture build.', flush=True)


def current(row):
    compact = validation.committed(RESULT, (KEY,))
    require(compact == {k: v for k, v in row.items() if k not in ('sources', 'dependencies', 'providers', 'environment')},
            'commit exact component state before execution')
    admitted, providers, env = platform()
    require(admitted == row['admitted'] and providers == row['providers'] and env == row['environment']
            and signature(env) == row['environment_sha256'] and signature(providers) == row['provider_sha256']
            and all(signature(validation.committed(RESULT, (key,))) == v
                    for key, v in row['earlier_refusals'].items()),
            'platform or immutable environment changed')
    controller = basis.comparison.build.controls(row['source_commit'], controller=Path(__file__))
    fixture = dict(stage.source.blobs(P, row['source_commit'], [SOURCE]))[SOURCE]
    require(fixture == (P / SOURCE).read_bytes(), 'fixture changed')
    controller[SOURCE] = sha(P / SOURCE)
    require(controller == row['sources'] and signature(controller) == row['source_sha256']
            and signature(row['dependencies']) == row['dependencies_sha256']
            and all(sha(path) == digest for path, digest in row['dependencies'].items()), 'source or compiler inputs changed')
    if 'binary' in row:
        require(sha(ELF) == row['binary']['sha256'] and ELF.stat().st_size == row['binary']['bytes']
                and all(sha(path) == digest for path, digest in row['actual_build_inputs'].items()),
                'built fixture or actual link input differs')


def build_fixture(row):
    current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'compile').exists(), 'build once from the frozen input')
    owner = ownership.Owner(ROOT / 'compile')
    try:
        compile_result, _, _ = owner.run('compile', row['build'][0], row['environment'], P, **BUDGET)
        require(ownership.closed(compile_result) and compile_result['exit_code'] == 0, 'fixture compilation failed')
        actual = dependency_paths((ROOT / 'fixture.d').read_text())
        expected = set(row['headers'])
        project_headers = {name for name in expected if Path(name).is_relative_to(P)
                           or Path(name).is_relative_to(stage.STAGE_ROOT / 'source')}
        require(project_headers <= actual <= expected, 'compile dependency escapes pinned coverage')
        link, out, err = owner.run('link', row['build'][1], row['environment'], P, **BUDGET)
        require(ownership.closed(link) and link['exit_code'] == 0, 'fixture link failed')
        files = {Path(line).resolve() for line in (out.read_text() + '\n' + err.read_text()).splitlines()
                 if line.startswith('/') and Path(line).is_file()}
        require(Path(row['admitted']['archive']['path']) in files and any('libsycl' in p.name for p in files)
                and all(str(p) in row['dependencies'] or p == ROOT / 'fixture.o' for p in files),
                'actual runtime link closure missing or has an unpinned input')
        row.update(stage='built', binary=dict(path=str(ELF), sha256=sha(ELF), bytes=ELF.stat().st_size),
                   actual_build_inputs={str(p): sha(p) for p in (ROOT / 'fixture.o', *files)},
                   actual_header_count=len(actual), conservative_header_count=len(expected),
                   actual_headers_sha256=signature(sorted(actual)),
                   conservative_headers_not_reported=sorted(expected - actual))
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
    finally:
        row.update(compile_commands=len(owner.commands), active=bool(owner.active),
                   ownership_closed=owner.active is None and all(ownership.closed(c) for c in owner.commands))
        save(row)
    require(row['stage'] == 'built' and row['ownership_closed'], row.get('error', 'build ownership did not close'))
    print('Fixture built. Commit binary and dependency evidence before smoke tests.', flush=True)


def fields(line):
    pairs = [part.split('=', 1) for part in line.split()]
    result = dict(pairs)
    require(len(result) == len(pairs), 'duplicate fixture field')
    return result


def parse(log, placement, operation):
    lines = log.splitlines()
    require(len(lines) == (12 if operation == 'timing' else 2) and lines[-1] == 'fixture_status=PASS',
            'wrong fixture result population')
    header = fields(lines[0])
    setup = float(header.pop('allocation_ms'))
    clock = float(header.pop('clock_triplet_median_ms'))
    require(header == dict(placement=placement, guarded_values='2', cancellation_drain='1')
            and all(math.isfinite(v) and v >= 0 for v in (setup, clock)), 'placement or setup/guard receipt differs')
    samples = {shape: [] for shape in SHAPES}
    for line in lines[1:-1]:
        v = fields(line)
        shape = v.pop('shape')
        desired, _ = SHAPES[shape]
        require(int(v.pop('batch')) == len(samples[shape]) and int(v.pop('copies')) == desired
                and set(v) == {'enqueue_ms', 'drain_ms', 'total_ms'}, 'unexpected fixture batch')
        numbers = {k: float(n) for k, n in v.items()}
        require(all(math.isfinite(n) and n >= 0 for n in numbers.values())
                and abs(numbers['enqueue_ms'] + numbers['drain_ms'] - numbers['total_ms']) < 1e-7,
                'timed intervals do not conserve')
        samples[shape].append(numbers)
    require(all(len(v) == (5 if operation == 'timing' else 0) for v in samples.values()), 'incomplete batches')
    return dict(setup_ms=setup, clock_triplet_median_ms=clock, samples=samples)


def decision(processes):
    arms = processes[2:]
    pairs = []
    for first, second in zip(arms[::2], arms[1::2], strict=True):
        ref, cand = sorted((first, second), key=lambda p: p['placement'] == 'host-usm')
        require(ref['placement'] == 'ordinary' and cand['placement'] == 'host-usm', 'mismatched placement pair')
        item = {}
        for shape in SHAPES:
            def median_total(p):
                return p['setup_ms'] + statistics.median(v['total_ms'] for v in p['samples'][shape])
            item[shape] = dict(ref_ms=median_total(ref), candidate_ms=median_total(cand))
            item[shape]['ratio'] = item[shape]['candidate_ms'] / item[shape]['ref_ms']
        pairs.append(item)
    ready = math.prod(p['ready']['ratio'] for p in pairs) ** (1 / len(pairs))
    queued = math.prod(p['queued']['ratio'] for p in pairs) ** (1 / len(pairs))
    return dict(pairs=pairs, ready_geomean_ratio=ready, queued_geomean_ratio=queued,
                model_eligible=ready <= 1.05 and all(p['queued']['ratio'] < 1 for p in pairs))


def execute(row, index):
    current(row)
    require(row['stage'] in ('built', 'running') and row['ownership_closed'] and not row['active']
            and len(row['processes']) == index and all(p['passed'] for p in row['processes']),
            'run each fresh step once in order')
    operation, placement = row['steps'][index]
    root = ROOT / str(index)
    require(not root.exists(), 'existing execution receipt')
    probe = row['admitted']['probe']
    require(all(sha(v['path']) == v['sha256'] for v in probe.values()), 'validation preload changed')
    env = dict(row['environment'], LD_PRELOAD=probe['library']['path'],
               STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'])
    cursor = next(s.split(': ', 1)[1] for s in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=10)
        .splitlines() if s.startswith('-- cursor: '))
    outcome = dict(index=index, operation=operation, placement=placement, host_probe_accepted=False)
    owner = ownership.Owner(root / 'owned')
    try:
        command, out, err = owner.run('fixture', [str(ELF), placement, operation], env, P, **BUDGET)
        require(ownership.closed(command) and command['exit_code'] == 0, 'native fixture failed')
        validated = validation.preload_evidence(err.read_text(), command['pid'])
        require(validated['loader'] == probe['loader']['path'] and
                not any(x in err.read_text() for x in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'retirement failed')),
                'provider or async validation failure')
        outcome.update(parse(out.read_text(), placement, operation),
                       host_probe_accepted=True, native_exit_code=0, validation=validated)
        if operation == 'timing':
            require(all(v['total_ms'] >= SHAPES[shape][1]
                        and SHAPES[shape][0] * outcome['clock_triplet_median_ms'] <= .05 * v['total_ms']
                        for shape, values in outcome['samples'].items() for v in values),
                    'batch failed the prospectively declared duration or clock-overhead gate')
    except BaseException as error:
        outcome.update(error=repr(error), host_probe_accepted=False)
    finally:
        validation.finish(outcome, owner, cursor, row['admitted']['boot'])
        if outcome.get('driver_after') != row['admitted']['driver']:
            outcome.update(passed=False)
        row['processes'].append(outcome)
        row.update(stage='running' if outcome['passed'] else 'failed',
                   active=bool(owner.active), ownership_closed=outcome['ownership_closed'])
        if outcome['passed'] and len(row['processes']) == len(row['steps']):
            row.update(stage='closed', complete=True, passed=True, decision=decision(row['processes']))
        save(row)
    print(json.dumps({k: outcome.get(k) for k in ('index', 'operation', 'placement', 'passed', 'fresh_faults', 'error')}))
    require(outcome['passed'], outcome.get('error', 'fixture validation failed'))


if __name__ == '__main__':
    require(sys.executable == basis.reference.model.PYTHON, 'use pinned Python')
    action, = sys.argv[1:]
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stage.builder.guards.no_model_owners([])
        if action == 'prepare':
            prepare()
        elif action == 'build':
            build_fixture(read(ROOT / 'record.json'))
        else:
            execute(read(ROOT / 'record.json'), int(action))
