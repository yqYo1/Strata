#!/usr/bin/env python3
"""Owned mapped-row copy comparison; independently authored, repository LICENSE.

Build only the component target from a committed private source snapshot. Keep
the production entry, all four proposals and both actual source placements.
This is component evidence, never model speed or physical-memory traffic.
"""
import fcntl
import hashlib
import io
import json
from pathlib import Path
import re
import select
import shlex
import statistics
import subprocess
import sys
import tarfile

sys.dont_write_bytecode = True
import benchmark_runtime as python_runtime
import measurement_guards as guards
import measurement_owner as owner
import ur_validation as validation

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'mapped-row-copy-component-v1'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'mapped_row_copy_component_v1'
TARGET = 'sycl_mapped_row_copy_component'
VARIANTS = ('production', 'row256e1', 'row256e4', 'tile256e4', 'tile64e4')
ORDERS = (VARIANTS, tuple(reversed(VARIANTS)), tuple(VARIANTS[i] for i in (2, 0, 4, 1, 3)))
FIELDS = ('wall_ns', 'cpu_ns', 'submit_wall_ns', 'submit_cpu_ns')
HELPERS = ('tools/sycl/mapped_row_copy_component.py', 'tools/sycl/benchmark_runtime.py',
    'tools/sycl/measurement_guards.py', 'tools/sycl/measurement_owner.py',
    'tools/sycl/ur_validation.py', 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md')
PACKAGES = ('libc6', 'libc6-dev', 'libstdc++6', 'libstdc++-13-dev', 'libgcc-s1', 'libgcc-13-dev', 'binutils')
require, sha = guards.require, owner.sha


def git(*args):
    return subprocess.check_output(['git', *args], cwd=P, timeout=30)


def signature(pins):
    return dict(files=len(pins), aggregate_sha256=hashlib.sha256(
        json.dumps(pins, sort_keys=True).encode()).hexdigest())


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('sources', 'runtime_pins', 'dependency_pins', 'commands')}
    out['source_inputs'] = signature(row['sources'])
    out['runtime_inputs'] = signature(row['runtime_pins'])
    out['dependency_inputs'] = signature(row['dependency_pins'])
    return out


def save(row):
    (ROOT / 'record.json').write_text(json.dumps(row, indent=2) + '\n')
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row)
    RESULT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def packages():
    return subprocess.check_output(['/usr/bin/dpkg-query', '-W',
        '-f=${binary:Package} ${Version}\n', *PACKAGES], text=True, timeout=15).splitlines()


def dependency_pins(roots):
    pending, visited, pins, total, entries = list(map(Path, roots)), set(), {}, 0, 0
    while pending:
        root = pending.pop().resolve()
        if root in visited: continue
        require(root.is_dir(), 'dependency include/config directory missing')
        visited.add(root)
        for path in root.rglob('*'):
            entries += 1
            require(entries <= 32768, 'dependency traversal budget')
            if path.is_symlink() and path.is_dir():
                pending.append(path.resolve())
            elif path.is_file():
                name = str(path.resolve())
                if name in pins: continue
                total += path.stat().st_size
                require(len(pins) < 16384 and total <= 256 << 20, 'finite dependency inventory')
                pins[name] = sha(path)
    return pins


def prepare():
    data = validation.committed(RESULT)
    cohort = data['decode_iq3_generation_speed_v1']
    require(cohort['stage'] == 'closed' and cohort['complete'] and cohort['ownership_closed']
        and not cohort['active'], 'close the frozen IQ3 cohort before component preparation')
    require(not ROOT.exists() and KEY not in data, 'fresh component attempt only')
    revision = git('rev-parse', 'HEAD').decode().strip()
    helpers = {n: sha(P / n) for n in HELPERS}
    require(all((P / n).read_bytes() == git('show', revision + ':' + n) for n in HELPERS),
        'commit exact component controls first')
    native = data['decode_iq3_pair_model_build_v1']
    env = dict(native['recipe']['environment'])
    env = {k: v for k, v in env.items() if not k.startswith('STRATA_IQ') and k != 'STRATA_CPU_SERVICE'}
    env.update(TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
        ZE_ENABLE_ALT_DRIVERS=cohort['environment_template']['ZE_ENABLE_ALT_DRIVERS'])
    provider = cohort['runtime_manifest']
    require(sha(provider['path']) == provider['sha256'], 'qualified provider inventory differs')
    runtime = json.loads(Path(provider['path']).read_text())
    require(len(runtime) == provider['files'] and all(sha(n) == h for n, h in runtime.items()),
        'qualified runtime files differ')
    for asset in cohort['native_validation_probe'].values():
        require(runtime.get(str(Path(asset['path']).resolve())) == asset['sha256'],
            'qualified validation preload/loader differs')
    tools = {name: dict(path=path, realpath=str(Path(path).resolve()), sha256=sha(path)) for name, path in
        dict(compiler='/opt/intel/oneapi/compiler/2026.2/bin/icpx', cmake='/usr/bin/cmake',
             ninja='/usr/bin/ninja', assembler='/usr/bin/as', linker='/usr/bin/ld', python=sys.executable).items()}
    versions = {n: subprocess.check_output([tools[n]['path'], '--version'], env=env,
        text=True, timeout=15).splitlines()[0] for n in ('compiler', 'cmake', 'ninja')}
    dependencies = ['/opt/intel/oneapi/compiler/2026.2/include',
        '/opt/intel/oneapi/mkl/2026.1/lib/cmake/mkl', '/usr/include/c++/13',
        '/usr/include/x86_64-linux-gnu/c++/13', '/usr/lib/gcc/x86_64-linux-gnu/13/include',
        '/usr/include/level_zero']
    dependencies += sorted(str(p) for p in Path('/opt/intel/oneapi/compiler/2026.2/lib/clang').glob('*/include'))
    require(len(dependencies) > 6, 'compiler resource headers required')
    dependency_inputs = dependency_pins(dependencies)
    names = git('ls-files', '-z', 'CMakeLists.txt', 'LICENSE', 'cmake', 'src', 'include',
        'tests', 'bench', 'tools', 'third_party').decode().split('\0')
    names = sorted(n for n in names if n and not n.startswith('bench/results/')
                   and n != 'bench/performance-records.csv')
    require(0 < len(names) <= 4096, 'finite source snapshot')
    archive = git('archive', '--format=tar', revision, *names)
    require(len(archive) <= 64 << 20, 'source snapshot byte budget')
    sources = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for item in source.getmembers():
            if item.isdir(): continue
            require(item.isfile() and item.name in names, 'only named regular source files')
            sources[item.name] = hashlib.sha256(source.extractfile(item).read()).hexdigest()
    require(set(sources) == set(names), 'complete archive source identity')
    configure = [tools['cmake']['path'], '-S', str(ROOT / 'source'), '-B', str(ROOT / 'build'),
        '-G', 'Ninja', '-DCMAKE_CXX_COMPILER=' + tools['compiler']['path'],
        '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
        '-DMKL_DIR=/opt/intel/oneapi/mkl/2026.1/lib/cmake/mkl',
        '-DSTRATA_ENABLE_SYCL=ON', '-DSTRATA_ENABLE_CUDA=OFF', '-DSTRATA_ENABLE_HIP=OFF',
        '-DSTRATA_NATIVE_EXPERTS=OFF', '-DSTRATA_BUILD_TESTS=OFF', '-DSTRATA_SYCL_TESTS=OFF',
        '-DSTRATA_BUILD_PREFILL_HOST_TESTS=OFF', '-DSTRATA_SYCL_VLLM_XE2_NATIVE16=OFF',
        '-DSTRATA_WERROR=ON', '-DCMAKE_CXX_FLAGS=-Wno-error=deprecated-declarations']
    row = dict(stage='prepared', source_commit=revision, sources=sources, helpers=helpers,
        dependency_roots=dependencies, dependency_pins=dependency_inputs,
        tools=tools, tool_versions=versions, system_packages=packages(), runtime_pins=runtime,
        runtime_basis=dict(evidence_commit=revision, key='decode_iq3_generation_speed_v1',
            scope='Reuse qualified provider files,safety settings and retained UR preload only;new component needs its own device validation.'),
        native_validation_probe=cohort['native_validation_probe'], environment=env,
        configure=configure, build=[tools['cmake']['path'], '--build', str(ROOT / 'build'),
            '--target', TARGET, '--parallel', '1'],
        build_budget=budget(300, 1200, file_cap=128 << 20),
        native_budget=budget(600, 1200, file_cap=16 << 20),
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), driver=guards.driver_guard(),
        plan=dict(order='Build,small traced check,full check,three fresh timing processes0/1/2.No retries.',
            dimensions=dict(rows=40, width=2560, sources=['host_usm', 'device'], banks=[1, 64], hits=[4, 20]),
            variants=VARIANTS, variant_orders=ORDERS, samples_per_cell=5, batch_launches=4096,
            warmup_batches=2, minimum_batch_ns=10000000, minimum_timer_multiple=300,
            sample_fields_ns=FIELDS,
            acceptance='Normal0,all bit/guard checks,actual Intel B570,live native ownership/environment/provider check,pre-main parameter validation,unchanged source/runtime/boot,no fresh fault. All600timed samples must meet timing,order and conservation gates. Compare each cell independently:eligible only when all3candidate/reference median wall ratios<1;no percentage floor.',
            limits='Whole dependent-batch host wall/thread-CPU only;submission overlaps device work and wait CPU may poll.No kernel-service,physical traffic,model gain or adoption claim.'),
        retention='Keep source/recipe,compact individual samples and decisions.Retire successful logs/traces and component build after its named comparison/implementation consumer ends;no new inference engine.',
        results=[], timing=[], active=False, ownership_closed=True, passed=False, complete=False,
        GPU_executed=False, model_executed=False)
    ROOT.mkdir(mode=0o700)
    for name in ('source', 'tmp', 'cache'): (ROOT / name).mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for item in source.getmembers():
            if item.isdir(): continue
            path = ROOT / 'source' / item.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source.extractfile(item).read()); path.chmod(item.mode & 0o777)
    save(row)
    print('Prepared committed-source copy component;commit its exact plan before build.', flush=True)


def current(row):
    require(validation.committed(RESULT, (KEY,)) == compact(row), 'commit exact component state')
    require(all((P / n).read_bytes() == git('show', row['source_commit'] + ':' + n)
                and sha(P / n) == h for n, h in row['helpers'].items()), 'component control drift')
    require(all(sha(ROOT / 'source' / n) == h for n, h in row['sources'].items()), 'native snapshot drift')
    require(all(sha(n) == h for n, h in row['runtime_pins'].items()), 'runtime provider drift')
    require(dependency_pins(row['dependency_roots']) == row['dependency_pins'], 'compiler/header/config dependency drift')
    require(all(str(Path(t['path']).resolve()) == t['realpath'] and sha(t['path']) == t['sha256']
                for t in row['tools'].values()) and packages() == row['system_packages'], 'toolchain drift')
    require(guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'driver/boot drift')
    if 'binary' in row:
        require(sha(row['binary']['path']) == row['binary']['sha256'], 'component executable drift')


def budget(wall, cpu, *, file_cap):
    # Owner includes the private source/build tree,not just raw output.
    return dict(wall=wall, cpu=cpu, rss_cap=8 << 30, text_cap=16 << 20,
                file_cap=file_cap, total_cap=512 << 20)


def collect_timing(text, index):
    lines = text.splitlines()
    clocks = [v for v in lines if v.startswith('CLOCK ')]
    require(len(clocks) == 1 and
        re.fullmatch(r'CLOCK median_stamp_ns=\d+ floor_ns=10000000 batch=4096 process=' + str(index), clocks[0]),
        'complete timing clock calibration required')
    timer = int(clocks[0].split()[1].split('=')[1])
    samples = [v for v in lines if v.startswith('SAMPLE ')]
    expected = [(kind, banks, hits, variant, n) for kind in ('host_usm', 'device')
        for banks in (1, 64) for hits in (4, 20) for variant in ORDERS[index] for n in range(5)]
    require(len(samples) == len(expected) == 200 and
        lines.count('TIMING passed=1 samples=200 model_speed=0 physical_traffic=0') == 1,
        'complete timing sample extent required')
    cells = {}
    for line, (kind, banks, hits, variant, n) in zip(samples, expected):
        pairs = [s.split('=', 1) for s in line.split()[1:]]
        f = dict(pairs)
        require(len(f) == len(pairs) and set(f) == {'process', 'variant', 'source', 'rows', 'width',
            'hits', 'banks', 'sample', 'launches', *FIELDS, 'wait_wall_ns', 'wait_cpu_ns'}, 'timing fields')
        require((f['source'], f['variant']) == (kind, variant), 'actual placement/variant order')
        integers = {k: int(v) for k, v in f.items() if k not in ('source', 'variant')}
        require(all(re.fullmatch(r'\d+', f[k]) for k in integers) and
            all(integers[k] == v for k, v in dict(process=index, rows=40, width=2560,
                hits=hits, banks=banks, sample=n, launches=4096).items()), 'sample shape/order differs')
        require(integers['wall_ns'] >= max(10000000, 300 * timer) and
            integers['submit_wall_ns'] + integers['wait_wall_ns'] == integers['wall_ns'] and
            integers['submit_cpu_ns'] + integers['wait_cpu_ns'] == integers['cpu_ns'], 'batch eligibility/conservation')
        cell = cells.setdefault(f'{kind}/banks{banks}/hits{hits}', {})
        cell.setdefault(variant, []).append([integers[k] for k in FIELDS])
    return dict(index=index, timer_stamp_median_ns=timer, cells=cells)


def decision(timing):
    require(len(timing) == 3 and [v['index'] for v in timing] == [0, 1, 2], 'three independent timing processes')
    out = {}
    for name in timing[0]['cells']:
        out[name] = {}
        for variant in VARIANTS[1:]:
            ratios = [statistics.median(v['cells'][name][variant][i][0] for i in range(5)) /
                statistics.median(v['cells'][name]['production'][i][0] for i in range(5)) for v in timing]
            out[name][variant] = dict(candidate_over_production_median_wall=ratios,
                component_gain_qualified=all(v < 1 for v in ratios))
    return dict(cells=out, model_speed_qualified=False, physical_traffic_qualified=False,
        inference_adopted=False, next='Consider only winning placement/shape cells for a separately qualified decode-only proposal;measure the complete model dependency path.')


def execute(mode):
    row = json.loads((ROOT / 'record.json').read_text()); current(row)
    expected = {'build': 'prepared', 'check-small': 'built', 'check': 'small-checked'}
    if mode.startswith('time'):
        index = int(mode[-1]); require(row['stage'] in ('checked', 'timing') and len(row['timing']) == index, 'fresh ordered timing only')
    else: require(row['stage'] == expected[mode], 'fresh ordered component stage only')
    scope = None; cursor = None
    outcome = dict(mode=mode, passed=False, host_probe_accepted=False)
    def persist():
        (ROOT / 'execution.json').write_text(json.dumps(dict(mode=mode,
            active=bool(scope and scope.active), commands=scope.commands if scope else []), indent=2) + '\n')
    try:
        scope = owner.Owner(ROOT / ('owned-' + mode), persist)
        if mode == 'build':
            outcome['build_commands'] = []
            for name, command in [('configure', row['configure']), ('build', row['build'])]:
                e, _, err = scope.run(name, command, row['environment'], ROOT, **row['build_budget'])
                require(owner.closed(e) and e['exit_code'] == 0, name + ' failed:' + err.read_text()[-1500:])
                outcome['build_commands'].append(dict(name=name, exit_code=0, ownership_closed=True))
            path = ROOT / 'build' / TARGET
            row['binary'] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
            database = ROOT / 'build/compile_commands.json'
            commands = [c for c in json.loads(database.read_text()) if
                TARGET + '.dir' in c['command'] or 'strata_sycl_runtime.dir' in c['command']]
            expected_sources = {'src/sycl/runtime.cpp', 'src/sycl/compat.cpp', 'src/sycl/command_profile.cpp',
                'src/kernels/sycl/elementwise.cpp', 'tests/sycl/mapped_row_copy.cpp'}
            require(len(commands) == 5 and
                {str(Path(c['file']).relative_to(ROOT / 'source')) for c in commands} == expected_sources and
                all('-fsycl' in shlex.split(c['command']) and
                    '-fsycl-id-queries-range=size_t' in shlex.split(c['command']) for c in commands),
                'exact five-translation-unit component build required')
            require(all('-fno-fast-math' in shlex.split(c['command']) and
                '-ffp-contract=off' in shlex.split(c['command']) for c in commands if
                TARGET + '.dir' in c['command']), 'copy bit-preservation compiler flags')
            row['compiled_commands'] = commands
            row['stage'] = 'built'
        else:
            env = dict(row['environment'], STRATA_COPY_COMPONENT_GATE='1',
                LD_PRELOAD=row['native_validation_probe']['library']['path'],
                STRATA_UR_PROBE_EXPECTED_LOADER=row['native_validation_probe']['loader']['path'],
                UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(ROOT / (mode + '-validation.log')))
            if mode != 'check-small':
                env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'; env.pop('UR_LOG_TRACING', None)
            args = ['--time', str(index)] if mode.startswith('time') else ['--' + mode]
            cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
                ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines() if v.startswith('-- cursor: '))
            def interact(io, command):
                while True:
                    line = io.line(600)
                    if line.startswith('STRATA_COPY_GATE '): break
                identity = owner.identity(command['pid'])
                require(identity and identity['pid'] == identity['sid'] == command['pid'] and
                    all(identity[k] == command['direct_identity'][k] for k in ('pid', 'start_ticks', 'sid')) and
                    line == 'STRATA_COPY_GATE pid=' + str(command['pid']), 'actual native gate owner')
                require(str(Path('/proc', str(command['pid']), 'exe').resolve()) == row['binary']['path'], 'actual executable')
                actual = dict(v.split('=', 1) for v in Path('/proc', str(command['pid']), 'environ').read_text().split('\0') if '=' in v)
                require(actual == env, 'immutable native environment differs')
                mapped = set(python_runtime.code_maps(command['pid']))
                required = {str(Path(env[k]).resolve()) for k in
                    ('LD_PRELOAD', 'STRATA_UR_PROBE_EXPECTED_LOADER', 'UR_ADAPTERS_FORCE_LOAD', 'ZE_ENABLE_ALT_DRIVERS')}
                require(required <= mapped <= set(row['runtime_pins']) | {row['binary']['path']}, 'actual runtime provider mappings')
                outcome['native_identity'] = identity
                outcome['mapped_provider_count'] = len(mapped)
                row['GPU_executed'] = True
                io.send(b'continue\n'); io.proc.stdin.close()
                while not io.exit_watch.poll(0) or not io.eof:
                    io.poll()
                    if select.select([io.proc.stdout], [], [], .02)[0]:
                        io.read(); io.pending.clear()
                io.exit(1)
            outcome['native_execution_attempted'] = True
            if not row['GPU_executed']: row['GPU_executed'] = None
            e, out, err = scope.run('native', [row['binary']['path'], *args], env, ROOT,
                interaction=interact, **row['native_budget'])
            require(owner.closed(e) and e['exit_code'] == 0, 'owned native component failed')
            outcome['native_exit_code'] = 0
            text, diagnostic = out.read_text(), err.read_text()
            outcome['validation'] = validation.preload_evidence(diagnostic, e['pid'])
            require(outcome['validation']['loader'] == env['STRATA_UR_PROBE_EXPECTED_LOADER'],
                'native validation provider differs')
            log = ROOT / (mode + '-validation.log')
            require(not log.exists() or log.stat().st_size == 0, 'parameter validation diagnostic')
            require(any(v.startswith('DEVICE name=') and 'B570' in v for v in text.splitlines()), 'actual device evidence')
            if mode.startswith('time'):
                row['timing'].append(collect_timing(text, index)); row['stage'] = 'timing'
                if index == 2: row.update(stage='closed', decision=decision(row['timing']), complete=True)
            else:
                footer = ('CHECK passed=1 small=' + str(int(mode == 'check-small')) + ' variants=5 launches=' +
                    ('80' if mode == 'check-small' else '1120') + ' metadata=device source=host_usm,device bitwise=1 guards=1')
                require(text.splitlines().count(footer) == 1, 'complete numerical/guard footer')
                if mode == 'check-small': require('urEnqueueKernelLaunch' in diagnostic, 'small API trace must contain actual kernel launch')
                row['stage'] = 'small-checked' if mode == 'check-small' else 'checked'
            outcome['host_probe_accepted'] = True
        # The originally committed input record is unchanged until final save.
        current(json.loads((ROOT / 'record.json').read_text()))
        outcome['passed'] = True
    except BaseException as error:
        outcome['error'] = repr(error); row.update(stage='failed', passed=False, complete=False)
    finally:
        if mode != 'build' and cursor is not None and scope is not None:
            validation.finish(outcome, scope, cursor, row['boot'])
            if not outcome['passed'] or outcome.get('driver_after') != row['driver'] or 'error' in outcome:
                outcome['passed'] = False
                row.update(stage='failed', passed=False, complete=False)
        row['results'].append(outcome)
        row['commands'] = scope.commands if scope else []
        row['active'] = bool(scope and scope.active)
        row['ownership_closed'] = scope is None or (scope.active is None and all(owner.closed(e) for e in scope.commands))
        row['passed'] = bool(row['complete'] and row['ownership_closed'] and outcome['passed'])
        persist(); save(row)
    print(json.dumps(dict(mode=mode, stage=row['stage'], passed=outcome['passed'], error=outcome.get('error'))), flush=True)
    require(outcome['passed'], outcome.get('error', 'component stage failed'))


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in
        ('prepare', 'build', 'check-small', 'check', 'time0', 'time1', 'time2'), 'use prepare|build|check-small|check|time0|time1|time2')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1])
