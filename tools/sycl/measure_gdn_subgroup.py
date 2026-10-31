#!/usr/bin/env python3
"""Main-owned GDN subgroup component build and comparison; repository LICENSE.

No model dispatch change, inherited GPU qualification or XeStrata input.
Commit prepare/build/each process before the next serialized invocation.
"""
import fcntl
import copy
import hashlib
import json
import math
from pathlib import Path
import select
import shlex
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import build_official_model as build
import engine_once
import runtime_probe
import ur_validation as validation

P, B, RESULT = build.P, build.B, build.RESULT
ROOT, KEY = B / 'official-gdn-subgroup-v1', 'official_gdn_subgroup_v1'
TARGETS = ('sycl_submission_test', 'sycl_gdn_columns_bench')
CASES = [('submission', TARGETS[0], ['--submission']),
         ('smoke', TARGETS[1], ['--subgroup-smoke']),
         ('validate', TARGETS[1], ['--subgroup-validate']),
         *[(f'perf{i}', TARGETS[1], ['--subgroup-bench', str(i)]) for i in range(3)],
         ('context', TARGETS[1], ['--subgroup-context'])]
require, digest, write, owner_api = build.require, build.digest, build.write, build.supervision


def git(*args):
    return subprocess.check_output(['/usr/bin/git', *args], cwd=P)


def sources(revision):
    pins = build.controls(revision, controller=Path(__file__).resolve())
    names = git('ls-files', '-z', 'CMakeLists.txt', 'cmake', 'include', 'src',
                'bench/sycl', 'tests/sycl', 'tools/sycl').decode().split('\0')
    for name in names:
        if name and (Path(name).suffix in ('.cpp', '.hpp', '.h', '.c', '.cmake')
                     or name.endswith('CMakeLists.txt')):
            raw = git('show', revision + ':' + name)
            require((P / name).read_bytes() == raw, 'uncommitted component input: ' + name)
            pins[name] = hashlib.sha256(raw).hexdigest()
    return pins


def compact(row):
    result = {k: v for k, v in row.items() if k not in ('sources', 'commands')}
    result['source_inputs'] = dict(files=len(row['sources']), aggregate_sha256=hashlib.sha256(
        json.dumps(row['sources'], sort_keys=True).encode()).hexdigest())
    return result


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def inventory(item):
    require(digest(item['path']) == item['sha256'], 'inventory changed: ' + item['path'])
    values = json.loads(Path(item['path']).read_text())
    require(len(values) == item['files'] and all(digest(p) == h for p, h in values.items()),
            'inventory input changed: ' + item['path'])
    return values


def qualified_probe(data, providers):
    """Reuse immutable probe code and its current official-provider evidence.

    The original preload qualification also froze historical GPU/OS libraries.
    Those are not the runtime used by the13official component processes and
    the just-closed model cohort. Never refresh their old hashes.
    """
    gate, model, built = (data[k] for k in ('official_prefill_joint_device_v1',
        'official_row64_model_v2', build.KEY))
    require(gate['passed'] and gate['complete'] and gate['ownership_closed'] and not gate['active']
            and gate['fresh_faults'] == [] and len(gate['processes']) == 13
            and all(p['passed'] for p in gate['processes'])
            and gate['binaries'] == built['binaries'] and gate['native_source_commit'] == built['source_commit'],
            'closed same-generation official runtime qualification required')
    require(model['passed'] and model['complete'] and model['ownership_closed'] and not model['active']
            and len(model['processes']) == 6 and all(p['passed'] and not p['fresh_faults'] for p in model['processes'])
            and model['native_source_commit'] == gate['native_source_commit']
            and model['boot'] == gate['boot'] and model['driver'] == gate['driver'],
            'current official model/runtime evidence differs')
    require(inventory(gate['runtime_manifest']) == providers, 'same official provider set required')
    original = runtime_probe.components()  # Recover/check original native artifacts, not old GPU qualification.
    loader = gate['environment']['STRATA_UR_PROBE_EXPECTED_LOADER']
    library = gate['environment']['LD_PRELOAD']
    probe = dict(library=dict(path=library, sha256=providers[library]),
                 loader=dict(path=loader, sha256=providers[loader]))
    require(probe == model['native_validation_probe'] and probe['library'] == original['library']
            and loader == '/opt/intel/oneapi/compiler/2026.2/lib/libur_loader.so.0',
            'unchanged native probe/current official loader required')
    return probe


def prepare():
    data = build.committed(RESULT)
    require(not ROOT.exists() and KEY not in data, 'fresh component generation only')
    model = data['official_row64_model_v2']
    require(model['complete'] and model['ownership_closed'] and not model['active']
            and len(model['processes']) == 6 and all(p['passed'] for p in model['processes']),
            'close the frozen six-process model cohort first')
    controls = selfcheck()
    prior = json.loads((build.ROOT / 'record.json').read_text())
    require(data[build.KEY] == build.compact(prior) and prior['passed'] and prior['ownership_closed']
            and not prior['active'], 'committed retained official build basis required')
    for item in prior['binaries'].values():
        require(digest(item['path']) == item['sha256'], 'retained official executable changed')
    pins = sources(git('rev-parse', 'HEAD').decode().strip())
    providers = inventory(prior['runtime_basis']); external = inventory(prior['external_manifest'])
    sdk = build.sdk()
    probe = qualified_probe(data, providers)
    env = dict(prior['environment'], TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    env.pop('UR_LOG_TRACING', None)
    env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
    cfg = ['/usr/bin/cmake', '-S', str(P), '-B', str(ROOT / 'build'), '-G', 'Ninja',
           '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_COMPILER=' + build.COMPILER,
           '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DSTRATA_ENABLE_SYCL=ON',
           '-DSTRATA_ENABLE_CUDA=OFF', '-DSTRATA_ENABLE_HIP=OFF', '-DSTRATA_BUILD_TESTS=OFF',
           '-DSTRATA_NATIVE_EXPERTS=OFF', '-DSTRATA_PORTABLE=OFF', '-DSTRATA_CPU_ARCH=znver3',
           '-DSTRATA_WERROR=ON',
           '-DSTRATA_SYCL_VLLM_XE2_NATIVE16=OFF', '-DSTRATA_SYCL_DEVICE_ARCH=',
           '-DSTRATA_SYCL_AOT_DOUBLE_GRF=OFF', '-DMKL_DIR=/opt/intel/oneapi/mkl/2026.1/lib/cmake/mkl',
           '-DCMAKE_CXX_FLAGS=-fsycl-id-queries-range=size_t -Wno-error=deprecated-declarations',
           '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-t']
    require(build.host_link_inventory.inventory(build.COMPILER, env) == prior['host_link_inputs'],
            'official host linker inventory changed')
    ROOT.mkdir(mode=0o700)
    for part in ('tmp', 'cache'): (ROOT / part).mkdir()
    write(ROOT / 'runtime-inputs.json', providers)
    row = dict(stage='prepared', source_commit=git('rev-parse', 'HEAD').decode().strip(), sources=pins,
        sdk=dict(path=str(build.SDK), sha256=build.SDK_SHA, files=len(sdk)),
        compiler=dict(path=build.COMPILER, sha256=build.COMPILER_SHA),
        external_manifest=prior['external_manifest'], host_link_inputs=prior['host_link_inputs'],
        runtime_manifest=dict(path=str(ROOT / 'runtime-inputs.json'),
            sha256=digest(ROOT / 'runtime-inputs.json'), files=len(providers)),
        native_validation_probe=probe, environment=env, configure_argv=cfg,
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), driver=build.guards.driver_guard(),
        host_controls=controls, commands=[], processes=[], active=False, ownership_closed=True, build_passed=False,
        passed=False, complete=False, inference_adopted=False, model_qualified=False,
        budgets=dict(wall=360, cpu=1080, rss_cap=6 << 30, text_cap=16 << 20,
            file_cap=16 << 20, total_cap=64 << 20), whole_bytes=512 << 20,
        retention='One two-executable component comparison;retain current official seven-ELF generation '
            'and selected6e40 fallback. Legacy provenance holds remain unresolved. Retire this component '
            'build/raw output after rejection or completed model integration;no engine is overwritten.',
        plan=dict(question='Can subgroup register exchange shorten the measured GDN recurrence path?',
            order='Build only,submission,17-token whole/split+7refusals,1/33/65/129/8191/8192exact '
                'validation,three fresh rotated resident processes,then262144component carry/restore/decode tail '
                'only if every performance pair wins. Stop on first fault/validation failure;no retries or replacements.',
            timing='8192tokens/call;fused,columns4,subgroup4 rotated by process and round. One first-use '
                'round separate,six subsequent rounds per arm. Eight empty-marker controls;require each '
                'sample>=1ms and>=1000profiling timer ticks. Queue interval includes host gaps and both kernels. '
                'Numerical comparisons follow each complete seven-update train,not every timed sample.',
            acceptance='Bit-exact state,FP32output and FP16capture against both owned references;guards, '
                'immutable inputs and rejected-call no-write. All3process subgroup4/columns4 median ratios<1 '
                'and geometric mean<1 permit context/next model proposal;no minimum gain. Normal owner '
                'closure,three native provider/environment gates,pre-main validation,quiet final validation '
                'and no fresh fault required. No model quality,RESTORE,TTFT,TG or1000tok/s follows.',
            source='Main-authored from owned GDN plus Khronos SYCL API contract;no XeStrata derivation. '
                'Core subgroup mapping source-audited at a1cb5de85;later timer/admission/build controls '
                'are new and require their own execution checks.'))
    save(row)
    print('Prepared bounded GDN component source/recipe;commit before build.', flush=True)


def current():
    row = json.loads((ROOT / 'record.json').read_text())
    require(build.committed(RESULT)[KEY] == compact(row), 'commit exact preceding plan/result first')
    require(sources(row['source_commit']) == row['sources'], 'component sources changed')
    allowed = build.sdk(); allowed.update(inventory(row['external_manifest']))
    providers = inventory(row['runtime_manifest']); allowed.update(providers)
    allowed.update({str(P / n): h for n, h in row['sources'].items()})
    require(build.host_link_inventory.inventory(build.COMPILER, row['environment']) == row['host_link_inputs'],
            'host linker changed')
    require(qualified_probe(build.committed(RESULT), providers) == row['native_validation_probe'],
            'private preload/current official qualification changed')
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot']
            and build.guards.driver_guard() == row['driver'], 'boot/driver changed')
    for item in row.get('binaries', {}).values():
        require(digest(item['path']) == item['sha256'], 'component executable changed')
    if row.get('compiled_inputs'):
        inventory(row['compiled_inputs'])
    return row, allowed, providers


def compile_component():
    row, allowed, _ = current()
    require(row['stage'] == 'prepared' and not (ROOT / 'owned-build').exists(), 'one fresh build only')
    owner = None
    def persist():
        write(ROOT / 'record.json', dict(row, commands=owner.commands if owner else [],
            active=bool(owner and owner.active)))
    def monitor(io, entry):
        next_size = 0.
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            if time.monotonic() >= next_size:
                require(owner_api.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole component size budget')
                next_size = time.monotonic() + .25
            if select.select([io.proc.stdout], [], [], .02)[0]: io.read(); io.pending.clear()
        io.exit(1)
    try:
        owner = owner_api.Owner(ROOT / 'owned-build', persist)
        commands = [('configure', row['configure_argv']),
            ('compile', ['/usr/bin/cmake', '--build', str(ROOT / 'build'), '--target', *TARGETS, '-j', '4']),
            ('dependencies', ['/usr/bin/ninja', '-C', str(ROOT / 'build'), '-t', 'deps'])]
        for label, argv in commands:
            print('CPU build:', label, flush=True)
            entry, out, err = owner.run(label, argv, row['environment'], P, interaction=monitor,
                **dict(row['budgets'], wall=1800, cpu=7200, total_cap=row['whole_bytes']))
            require(owner_api.closed(entry) and entry['exit_code'] == 0,
                    label + ': ' + (out.read_text(errors='replace') + err.read_text(errors='replace'))[-3500:])
        found = {str((Path(line.strip()) if Path(line.strip()).is_absolute()
                     else ROOT / 'build' / line.strip()).resolve()) for line in out.read_text().splitlines()
                 if line.startswith('    ') and line.strip()}
        units = {'src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp',
                 'src/kernels/sycl/fused_gdn.cpp', 'bench/sycl/gdn_columns.cpp', 'tests/sycl/compat.cpp'}
        database_path = ROOT / 'build/compile_commands.json'
        database = json.loads(database_path.read_text())
        selected_commands = []
        for entry in database:
            argv = entry.get('arguments') or shlex.split(entry['command'])
            output = argv[argv.index('-o') + 1]
            if not any(output.startswith('CMakeFiles/' + name + '.dir/')
                       for name in (*TARGETS, 'strata_sycl_runtime')): continue
            name = str(Path(entry['file']).resolve().relative_to(P))
            require(name in units and Path(argv[0]).resolve() == Path(build.COMPILER).resolve(),
                    'unexpected component compiler or translation unit')
            expected = ['-fsycl', '-fsycl-id-queries-range=size_t', '-DSTRATA_ENABLE_SYCL=1',
                        '-O3', '-DNDEBUG', '-std=c++20']
            if name in ('bench/sycl/gdn_columns.cpp', 'src/kernels/sycl/fused_gdn.cpp'):
                expected += ['-fno-fast-math', '-ffp-contract=off']
                require(not any(a in argv for a in ('-ffast-math', '-ffp-contract=fast', '-Ofast')),
                        'unexpected relaxed GDN floating-point mode')
            require(all(a in argv for a in expected), 'missing exact compile option: ' + name)
            selected_commands.append(dict(source=name, compiler=argv[0], output=output,
                required_options=expected, argv_sha256=hashlib.sha256(json.dumps(argv).encode()).hexdigest()))
        actual_units = {str(P / e['source']) for e in selected_commands}
        require(len(selected_commands) == 6 and actual_units == {str(P / n) for n in units} and actual_units <= found,
                'six production/fixture translation units missing')
        require({p for p in found if Path(p).is_relative_to(P) and Path(p).suffix == '.cpp'} == actual_units,
                'unexpected project translation unit compiled')
        require(found and all(allowed.get(p) == digest(p) for p in found),
                'unplanned actual compiler input;close build without refreshing hashes')
        write(ROOT / 'compiler-inputs.json', {p: allowed[p] for p in sorted(found)})
        row.update(stage='built', build_passed=True,
            compile_command_attestation=dict(database_sha256=digest(database_path), units=selected_commands),
            compiled_inputs=dict(path=str(ROOT / 'compiler-inputs.json'),
                sha256=digest(ROOT / 'compiler-inputs.json'), files=len(found)),
            actual_link_inputs=build.traced_link_inputs(allowed, ROOT),
            binaries={name: dict(path=str(ROOT / 'build' / name), sha256=digest(ROOT / 'build' / name),
                bytes=(ROOT / 'build' / name).stat().st_size) for name in TARGETS})
        require(all(digest(p) == h for p, h in allowed.items()), 'post-build frozen input changed')
    except BaseException as error:
        row.update(stage='failed', error=repr(error), build_passed=False)
    finally:
        row.update(commands=owner.commands if owner else [], active=bool(owner and owner.active),
            ownership_closed=bool(owner and owner.active is None and all(owner_api.closed(c) for c in owner.commands)))
        save(row)
    require(row['build_passed'] and row['ownership_closed'], row.get('error', 'build owner incomplete'))
    print('Two component executables built;GPU qualification remains pending.', flush=True)


def performance(processes):
    arms, ratios = {}, []
    for name in ('fused', 'columns4', 'subgroup4'):
        arms[name] = [statistics.median(e['queue_interval_ms'] for e in p['events']
            if e['kind'] == 'sample' and e['round'] >= 0 and e['schedule'] == name)
            for p in processes]
    ratios = [a / b for a, b in zip(arms['subgroup4'], arms['columns4'])]
    geometric = math.prod(ratios) ** (1 / 3)
    return dict(process_medians_ms=arms, subgroup_over_columns_ratios=ratios,
        geometric_mean_ratio=geometric, advance=all(r < 1 for r in ratios) and geometric < 1)


def final_driver_guard(result, expected):
    result['final_driver_matches'] = result.get('driver_after') == expected
    if not result['final_driver_matches']:
        result['passed'] = False
        result.setdefault('error', 'Post-run driver differs from the frozen platform')


def events_for(stdout, mode, ordinal):
    events = [json.loads(line) for line in stdout.splitlines() if line.startswith('{')]
    expected = {'submission': 0, 'smoke': 3, 'validate': 6, 'context': 1}.get(mode, 30)
    require(len(events) == expected and all(e.get('status') == 'PASS' for e in events),
            'missing or failing fixture events')
    if mode.startswith('perf'):
        controls = [e for e in events if e['kind'] == 'timer_control']
        samples = [e for e in events if e['kind'] == 'sample']
        checks = [e for e in events if e['kind'] == 'train_validation']
        require([e['control'] for e in controls] == list(range(8)) and len(samples) == 21,
                'missing timer controls or samples')
        require(len(checks) == 1 and checks[0] == dict(kind='train_validation', process=ordinal,
            tokens_per_call=8192, updates_per_arm=7, final_state_output_fp16_exact=True,
            guards=True, inputs_immutable=True, status='PASS'), 'final-train numerical/guard evidence missing')
        require(all(e['process'] == ordinal and type(e['resolution_ns']) is int and e['resolution_ns'] > 0
                    and type(e['empty_marker_gap_ns']) is int and e['empty_marker_gap_ns'] >= 0
                    for e in controls), 'invalid timer control')
        resolution = controls[0]['resolution_ns']
        require(all(e['resolution_ns'] == resolution for e in controls), 'timer resolution changed')
        require([(e['round'], e['schedule']) for e in samples] ==
            [(r, ('fused', 'columns4', 'subgroup4')[(s + r + 1 + ordinal) % 3])
             for r in range(-1, 6) for s in range(3)], 'sample order/arm differs')
        require(all(e['process'] == ordinal and type(e['queue_interval_ms']) in (int, float)
                    and type(e['round']) is int and type(e['process']) is int
                    and math.isfinite(e['queue_interval_ms']) and e['queue_interval_ms'] >= 1
                    and e['queue_interval_ms'] * 1e6 / resolution >= 1000
                    and type(e['host_submit_ms']) in (int, float) and math.isfinite(e['host_submit_ms'])
                    and e['host_submit_ms'] >= 0 for e in samples),
                'invalid/ineligible timer or arithmetic sample')
    elif mode == 'smoke':
        require([(e['kind'], e.get('tokens'), e.get('split')) for e in events[:2]] ==
                [('validation', 17, False), ('validation', 17, True)]
                and events[2]['cases'] == 7 and events[2]['no_write'] is True, 'tiny/no-write coverage differs')
    elif mode == 'validate':
        require([e['tokens'] for e in events] == [1, 33, 65, 129, 8191, 8192], 'tail geometry differs')
    elif mode == 'context':
        require(events[0]['tokens'] == 262144 and events[0]['state_restore'] is True
                and events[0]['existing_decode_tail'] is True, 'component context/carry coverage differs')
    elif mode == 'submission':
        require(all(s in stdout for s in ('case stream_copy_ordering=PASS',
            'case explicit_stream_profiling=PASS', 'case default_stream_profiling=PASS',
            *[f'case queued_kernels={n} numerical=PASS guards=PASS event_timing=PASS kernel_profiling=PASS'
              for n in (1, 119, 120, 121, 241)],
            'case ordinary_host_roundtrips=3 status=PASS', 'case teardown=PASS')),
            'incomplete actual-runtime submission checks')
    if mode in ('smoke', 'validate'):
        require(all(e['schedule'] == 'subgroup4' and e['exact_vs_fused_and_columns4'] is True
                    and e['guards'] is True for e in events if e['kind'] == 'validation'),
                'missing exact arithmetic/guard evidence')
    return events


def selfcheck():
    controls = [dict(kind='timer_control', process=0, control=i, resolution_ns=80,
        empty_marker_gap_ns=120, status='PASS') for i in range(8)]
    samples = [dict(kind='sample', process=0, round=r,
        schedule=('fused', 'columns4', 'subgroup4')[(s + r + 1) % 3],
        queue_interval_ms=100., host_submit_ms=.2, status='PASS')
        for r in range(-1, 6) for s in range(3)]
    checks = [dict(kind='train_validation', process=0, tokens_per_call=8192, updates_per_arm=7,
        final_state_output_fp16_exact=True, guards=True, inputs_immutable=True, status='PASS')]
    valid = controls + samples + checks
    def parse(events):
        return events_for('\n'.join(json.dumps(e) for e in events), 'perf0', 0)
    require(len(parse(valid)) == 30, 'valid parser control failed')
    bad = [controls + samples[:-1] + checks, controls + list(reversed(samples)) + checks,
           controls + samples]
    for key, value in [('queue_interval_ms', 0.), ('queue_interval_ms', float('nan')),
                       ('queue_interval_ms', float('inf')), ('queue_interval_ms', .9),
                       ('queue_interval_ms', True), ('host_submit_ms', -1.),
                       ('host_submit_ms', float('nan')), ('round', True),
                       ('process', 1)]:
        events = copy.deepcopy(valid); events[8][key] = value; bad.append(events)
    for key, value in [('resolution_ns', 0), ('resolution_ns', 2000), ('empty_marker_gap_ns', -1)]:
        events = copy.deepcopy(valid)
        for event in events[:8]: event[key] = value
        if key == 'resolution_ns' and value == 2000:
            for event in events[8:]:
                if event['kind'] == 'sample': event['queue_interval_ms'] = 1.
        bad.append(events)
    for key in ('final_state_output_fp16_exact', 'guards', 'inputs_immutable'):
        events = copy.deepcopy(valid); events[-1][key] = False; bad.append(events)
    for events in bad:
        try: parse(events)
        except (RuntimeError, KeyError, TypeError, ValueError): pass
        else: raise RuntimeError('invalid parser control accepted')
    processes = [dict(events=[dict(e, queue_interval_ms=99.9 if e['schedule'] == 'subgroup4' else 100.)
        for e in samples]) for _ in range(3)]
    require(performance(processes)['advance'], 'small credible improvement was rejected')
    for event in processes[1]['events']:
        if event['schedule'] == 'subgroup4': event['queue_interval_ms'] = 100.01
    require(not performance(processes)['advance'], 'one slower process must reject advancement')
    for actual, prior_pass, expected_pass in (({'device': 'same'}, True, True),
            ({'device': 'changed'}, True, False), ({'device': 'same'}, False, False)):
        result = dict(passed=prior_pass, driver_after=actual)
        final_driver_guard(result, {'device': 'same'})
        require(result['passed'] is expected_pass, 'final driver guard control failed')
    return dict(valid_parser_controls=1, invalid_parser_controls=len(bad), performance_controls=2,
                final_driver_controls=3, GPU_executed=False)


def run(index):
    row, allowed, providers = current()
    require(row['build_passed'] and len(row['processes']) == index and
            all(p['passed'] for p in row['processes']), 'next fresh ordered process only')
    if index == 6: require(row['performance']['advance'], 'component performance gate failed')
    mode, target, args = CASES[index]
    root = ROOT / mode; require(not root.exists(), 'existing process must be inspected;no retries')
    root.mkdir(mode=0o700)
    for part in ('tmp', 'cache'): (root / part).mkdir()
    env = dict(row['environment'], TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'),
        CUDA_MODULE_LOADING='EAGER', STRATA_TEST_MAIN_GATES='1',
        ZE_ENABLE_ALT_DRIVERS='/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.17.39758',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'))
    if index <= 1:
        env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_TRACING='level:info;flush:info;output:file,' + str(root / 'ur.trace'))
    env = engine_once.native_environment(row, env)
    require(env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and env['EnableDirectSubmission'] == '0'
            and env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1' and env['ZES_ENABLE_SYSMAN'] == '0',
            'qualified runtime safety controls required')
    cursor = next(s.split(': ', 1)[1] for s in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines()
        if s.startswith('-- cursor: '))
    result = dict(mode=mode, complete=False, passed=False, host_probe_accepted=False, gates=[],
        environment_delta={k: v for k, v in env.items() if row['environment'].get(k) != v})
    owner, maps = None, set()
    def persist():
        write(root / 'execution.json', dict(result, commands=owner.commands if owner else [],
            active=bool(owner and owner.active)))
    def monitor(io, entry):
        next_size = 0.
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            if time.monotonic() >= next_size:
                require(owner_api.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole-cohort size budget')
                next_size = time.monotonic() + .25
            if select.select([io.proc.stdout], [], [], .01)[0]:
                io.read()
                while b'\n' in io.pending:
                    line, _, tail = io.pending.partition(b'\n'); io.pending[:] = tail
                    if not line.startswith(b'COMPONENT_READY '): continue
                    phase = line.decode().split(' ', 1)[1]
                    require(len(result['gates']) < 3 and phase ==
                        ['before-runtime', 'runtime-ready', 'before-exit'][len(result['gates'])], 'native gate order')
                    who = owner_api.identity(entry['pid']); require(who is not None, 'missing native identity')
                    identity = {k: who[k] for k in ('pid', 'start_ticks', 'sid')}
                    require(result.get('native_identity', identity) == identity, 'native identity changed')
                    result['native_identity'] = identity
                    actual = dict(v.split(b'=', 1) for v in Path('/proc', str(entry['pid']), 'environ').read_bytes()
                                  .split(b'\0') if b'=' in v)
                    require(actual == {k.encode(): v.encode() for k, v in env.items()}, 'native startup environment differs')
                    mapped = runtime_probe.executable_mappings(Path('/proc', str(entry['pid']), 'maps').read_text())
                    require(mapped <= set(providers) | {row['binaries'][target]['path']}, 'unplanned executable provider')
                    maps.update(mapped); result['gates'].append(phase); persist(); io.send(b'X')
        io.exit(1)
    try:
        owner = owner_api.Owner(root / 'owned', persist)
        command, out, err = owner.run('target', [row['binaries'][target]['path'], *args], env, P,
            interaction=monitor, **row['budgets'])
        require(owner_api.closed(command) and command['exit_code'] == 0, 'native process did not close normally')
        require(result['gates'] == ['before-runtime', 'runtime-ready', 'before-exit'], 'native gates incomplete')
        result['pre_main_validation'] = validation.preload_evidence(err.read_text(), command['pid'])
        require(result['pre_main_validation']['loader'] == env['STRATA_UR_PROBE_EXPECTED_LOADER'], 'wrong UR loader')
        require({env['LD_PRELOAD'], env['STRATA_UR_PROBE_EXPECTED_LOADER'], env['ZE_ENABLE_ALT_DRIVERS'],
                 '/opt/intel/oneapi/compiler/2026.2/lib/libsycl.so.9.0.0'} <= maps, 'official providers not observed')
        stdout, stderr = out.read_text(), err.read_text()
        require('fixture_status=PASS' in stdout and not any(s in stderr for s in
            ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'CHECKED_DRAIN_FAILED', 'terminate called')), 'fixture/runtime failed')
        require(not (root / 'validation.log').exists() or (root / 'validation.log').stat().st_size == 0,
                'final parameter validation is not quiet')
        result['events'] = events_for(stdout, mode, index - 3)
        require(all(digest(p) == h for p, h in allowed.items()) and
                all(digest(a['path']) == a['sha256'] for a in row['binaries'].values()), 'post-run input changed')
        result.update(complete=True, host_probe_accepted=True, native_exit_code=0, official_providers_observed=True)
    except BaseException as error:
        result.update(error=repr(error), complete=False, host_probe_accepted=False)
    finally:
        if owner: validation.finish(result, owner, cursor, row['boot'])
        final_driver_guard(result, row['driver'])
        result['passed'] = bool(result.get('passed') and result['complete']); persist()
        row['processes'].append(result)
        if index == 5 and result['passed']: row['performance'] = performance(row['processes'][3:6])
        closed = index == 6 or (index == 5 and result['passed'] and not row['performance']['advance'])
        row.update(stage='closed' if closed else 'running' if result['passed'] else 'failed',
            complete=bool(closed and result['passed']), passed=bool(closed and result['passed']),
            active=bool(owner and owner.active), ownership_closed=result.get('ownership_closed', False))
        save(row)
    print(json.dumps({k: result.get(k) for k in ('mode', 'passed', 'complete', 'ownership_closed', 'error')}), flush=True)
    require(result['passed'], 'component execution failed;stop before next process')


if __name__ == '__main__':
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); build.guards.no_model_owners([])
        if sys.argv[1:] == ['selfcheck']: print(json.dumps(selfcheck()))
        elif sys.argv[1:] == ['prepare']: prepare()
        elif sys.argv[1:] == ['build']: compile_component()
        elif len(sys.argv) == 3 and sys.argv[1] == 'run' and sys.argv[2] in tuple(map(str, range(7))): run(int(sys.argv[2]))
        else: raise RuntimeError('measure_gdn_subgroup.py selfcheck|prepare|build|run INDEX(0..6)')
