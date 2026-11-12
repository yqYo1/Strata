#!/usr/bin/env python3
"""Bounded single-core134/138-byte GU layout comparison; Main-owned.

Repository LICENSE. No production engine change, borrowed native object or GPU.
The old whole-pool failures stay failed; only a new kernel mechanism is tested.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import sys

sys.dont_write_bytecode = True
import cpu_compact_pool as pool

hot, inputs = pool.hot, pool.inputs
P, B = pool.P, pool.B
ROOT = B / 'cpu-compact-expanded-scales-v1'
PLAN = P / 'bench/sycl/cpu_compact_hot/scale-plan-v1.json'
THIS = 'tools/sycl/cpu_compact_scale.py'
SOURCE = 'bench/sycl/cpu_compact_hot/scale_layout.cpp'
CODEC = 'include/strata/kernels/cpu/detail/compact_iq3xxs.hpp'
ORDER = ['reference', 'expanded', 'expanded', 'reference', 'reference', 'expanded']
need, sha, write = hot.need, hot.sha, hot.write
EXTRA_HEADERS = ('/usr/include/x86_64-linux-gnu/sys/resource.h',
    '/usr/include/x86_64-linux-gnu/bits/resource.h',
    '/usr/include/x86_64-linux-gnu/bits/types/struct_rusage.h')


def prepare():
    need(not ROOT.exists() and not PLAN.exists(), 'fresh plan/root required')
    revision = hot.git('rev-parse', 'HEAD').decode().strip()
    names = (THIS, SOURCE, CODEC, 'AGENTS.md', 'docs/ARTIFACT_RETENTION.md',
        'tools/sycl/cpu_compact_pool.py', 'tools/sycl/cpu_compact_hot.py',
        'tools/sycl/cpu_compact_numeric.py', 'tools/sycl/measurement_owner.py',
        'tools/sycl/measurement_guards.py')
    sources = {}
    for name in names:
        raw = hot.git('show', revision + ':' + name)
        need(raw == (P / name).read_bytes(), 'commit source before preparation: ' + name)
        sources[name] = hashlib.sha256(raw).hexdigest()
    native_path = B / 'coherent-prefill-native-v1/record.json'
    native = json.loads(native_path.read_text())
    need(not native['active'] and native['ownership_closed'] and
         native['native_source_commit'] == '14508acbf1b70eb994fdf37f97fb6d0f5cff7d43', 'coherent fallback recipe')
    ggml, = [d for d in native['dependencies'] if d['revision'] == '3cf03257f219afbe7334045ff7c6a06ac68c627d']
    census, external = native['actual_inputs'], native['external_inputs']
    need(sha(census['path']) == census['sha256'] and sha(external['path']) == external['sha256'], 'SDK census continuity')
    environment = dict(PATH='/opt/intel/oneapi/compiler/2026.2/bin:/usr/bin:/bin',
        LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.2/lib:/usr/lib/x86_64-linux-gnu',
        LANG='C.UTF-8', LC_ALL='C.UTF-8', TMPDIR=str(ROOT / 'tmp'))
    flags = ['--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13', '-O3', '-g', '-DNDEBUG',
        '-std=gnu++20', '-march=znver3', '-mtune=znver3', '-mavx2', '-mfma', '-mf16c']
    includes = ['-I' + str(p) for p in (P / 'include', Path(ggml['root']) / 'ggml/src', Path(ggml['root']) / 'ggml/include')]
    headers = pool.std_headers()
    plan = dict(schema='compact-expanded-scales-v1', source_commit=revision, sources=sources,
        native_recipe=dict(path=str(native_path), sha256=sha(native_path)), compiler_inputs=census,
        external_inputs=external, GGML=ggml,
        standard_headers=dict(count=len(headers), sha256=pool.signature(headers)),
        extra_headers={p: sha(p) for p in EXTRA_HEADERS}, topology=pool.topology(),
        tools={p: sha(p) for p in (hot.CXX, hot.READELF, hot.PRLIMIT, hot.PYTHON, '/usr/bin/perf', '/usr/bin/true')},
        metadata=dict(path=str(inputs.METADATA), sha256=sha(inputs.METADATA)), formats=inputs.model_rows(),
        environment=environment,
        counter_capability=dict(command=['/usr/bin/perf', 'stat', '-x,', '-e',
            'cycles,instructions,cache-references,cache-misses', '--', '/usr/bin/true'],
            wall=10, perf_event_paranoid=Path('/proc/sys/kernel/perf_event_paranoid').read_text().strip(),
            meaning='Unprivileged capability probe only. Preserve a refusal;do not change privileges/sysctl or infer saturation. A refusal does not block independent clean kernel timing.'),
        compile=[hot.CXX, *flags, *includes, '-MD', '-MF', str(ROOT / 'scale.d'),
            '-c', str(P / SOURCE), '-o', str(ROOT / 'scale.o')],
        link=[hot.CXX, '--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13', '-O3', '-Wl,-t',
            str(ROOT / 'scale.o'), '-lm', '-o', str(ROOT / 'scale')],
        order=ORDER, pairs=[[0, 1], [3, 2], [4, 5]],
        expected_numeric=dict(passed=True, cases=384, bit_exact=True,
            wrong_scale_control_detected=True, single_core_kernel_only=True),
        expected_ELF=dict(elf_class='ELF64', data="2's complement, little endian",
            machine='Advanced Micro Devices X86-64', elf_type='EXEC (Executable file)',
            interpreter='/lib64/ld-linux-x86-64.so.2'),
        provenance='Main-authored scale precomputation from owned134-byte codec. Original pinned MIT GGML tables only;no XeStrata source/design. New files follow project LICENSE.',
        mechanism='Store eight odd scales instead of four packed scale bytes:138rather than134bytes/block,+2.9851%encoded GU storage. Remove runtime scale nibble extraction and2*s+1;preserve weight codes,integer products,FPblock-FMA order,reduction,0.25factor andstd::exp. Original134-byte production codec is unchanged. Machine-code inspection of its actual c61object showed compiler-balanced integer sums,295instructions,no YMMstack operands;independent-accumulator rewrite was not implemented.',
        scope='Isolate single-core real-weight GU codec/kernel onCPU0;not Cache.lookup,leases,ExpertPool,hidden quantization,Down,actual router or model timing. One binary with explicit immutable arm argument;no process environment mutation. Only selected layout allocated during speed processes. Two actual IQ3_XXS GU layer slices labeled by Down20/42.',
        workloads='32real experts per layer;four synthetic Q8_K activations cover zero,finite scales and every int8value including-128/127. Untimed384full/partial/empty-row bit comparisons and wrong-scale negative control. Timing cycles1/6/32experts via(13*i+3)modN,activationi%4;working sets may hit caches and are not controlled cold DRAM. No model/task precision conclusion.',
        timing='Fresh separate processes per format and arm in enforcedR,C,C,R,R,Corder. For each of3working sets calibrate powers2calls up to65536 until>=2s;then3batches each>=1s,timer-pair overhead<0.1%. Setup,encoding,checks and output outside timing;normal extent/alias guards and caller/dispatch remain timed. Normalize wall/expert,retain individual batches and spread.',
        acceptance='First require exact bit/guard/range/negative-control pass and normal0quiet closed owners for both layers. Each of6format/working-set cells is judged independently:all3matched pairs must improve,without a minimum percentage. A passing cell permits only a prospectively planned corresponding production-pool comparison;other cells stay failed. No global cache/model trial,full-context,PPor70TGadoption from this kernel experiment. Original V1/V2whole-pool failed gates stay unchanged.',
        stopping='Single-use actions. Stop on source/SDK/model/executable/ELF/owner/order/numeric/counter/timing/budget mismatch;preserve failure,no hash refresh or retry. Do not continue toward a model trial after this component screen.',
        budgets=dict(build_wall=120, check_wall=120, measure_wall=180, rss=512 << 20,
            output=128 << 10, total=32 << 20, cpu=240),
        retention='Current standalone kernel comparison only. Keep coherent GPUfallback and qualified CPUpool library unchanged;retire ended old V2timing fixture after exact original source/recipe recovery before this build. Keep useful individual samples,source/recipe and gates in maintained CPUresult,not routine logs. Remove rejected prototype/ELF when its named consumer ends;recover it from original Git.')
    ROOT.mkdir(mode=0o700); (ROOT / 'tmp').mkdir(); write(PLAN, plan)
    print('Prepared single-core scale experiment;commit plan before build/check/measure.', flush=True)


def verify(plan):
    need(hot.git('show', 'HEAD:' + str(PLAN.relative_to(P))) == PLAN.read_bytes(), 'commit exact scale plan')
    for name, expected in plan['sources'].items():
        raw = hot.git('show', plan['source_commit'] + ':' + name)
        need(hashlib.sha256(raw).hexdigest() == expected == sha(P / name), 'source drift: ' + name)
    for name in ('native_recipe', 'compiler_inputs', 'external_inputs', 'metadata'):
        need(sha(plan[name]['path']) == plan[name]['sha256'], name + ' drift')
    need(all(sha(p) == h for p, h in plan['tools'].items()), 'tool drift')
    seed = json.loads(Path(plan['compiler_inputs']['path']).read_text())
    need(all(sha(p) == h for p, h in (dict(seed['compiled']) | dict(seed['links'])).items()
             if p.startswith(('/usr/', '/opt/'))), 'SDK drift')
    for name, expected in plan['GGML']['source_sha256'].items():
        p = Path(plan['GGML']['root']) / name
        need(sha(p) == expected, 'GGML dependency drift')
    headers = pool.std_headers()
    need(len(headers) == plan['standard_headers']['count'] and pool.signature(headers) == plan['standard_headers']['sha256'], 'standard header drift')
    need(all(sha(p) == h for p, h in plan['extra_headers'].items()) and pool.topology() == plan['topology'], 'OS/topology drift')
    need(plan['order'] == ORDER, 'arm order drift')
    for row in plan['formats']:
        need(inputs.identity(row['shard']) == row['identity'] and inputs.fixture_hash(row) == row['sha256'], 'real model slice drift')


def dependencies(plan):
    seed = json.loads(Path(plan['compiler_inputs']['path']).read_text())
    known = dict(seed['compiled']) | dict(seed['links']) | pool.std_headers() | plan['extra_headers']
    known.update({str(P / p): h for p, h in plan['sources'].items()})
    known.update({str(Path(plan['GGML']['root']) / p): h for p, h in plan['GGML']['source_sha256'].items()})
    known = {str(Path(p).resolve()): h for p, h in known.items()}
    tokens = shlex.split((ROOT / 'scale.d').read_text().replace('\\\n', ' '))
    need(tokens[0] == str(ROOT / 'scale.o') + ':', 'object dependency target')
    actual = {}
    for name in tokens[1:]:
        p = str(Path(name).resolve()); need(p in known and sha(p) == known[p], 'uncovered compiler input: ' + p)
        actual[p] = known[p]
    need(str(P / SOURCE) in actual and str(P / CODEC) in actual and len(actual) >= 200, 'complete single-TU compiler census')
    return actual


def closed(record, plan):
    return record['passed'] and record['complete'] and record['ownership_closed'] and not record['active'] and \
        record['source_commit'] == plan['source_commit'] and record['plan_sha256'] == sha(PLAN) and \
        all(hot.owner.closed(c) for c in record['commands'])


def execute(action):
    with (ROOT / (action + '-attempt.json')).open('x') as f: json.dump(dict(action=action, no_retry=True), f)
    state = dict(stage=action, passed=False, active=True, complete=False, GPU_executed=False, commands=[])
    def persist(): write(ROOT / (action + '.json'), state)
    persist()
    try:
        plan = json.loads(PLAN.read_text()); verify(plan)
        scope = hot.owner.Owner(ROOT / 'owned', persist); state['commands'] = scope.commands
        state.update(source_commit=plan['source_commit'], plan_sha256=sha(PLAN))
        def run(label, argv, wall):
            entry, out, err = scope.run(action + '-' + label, [hot.PRLIMIT, '--as=8589934592', '--', *argv],
                plan['environment'], P, wall=wall, text_cap=plan['budgets']['output'],
                rss_cap=plan['budgets']['rss'], total_cap=plan['budgets']['total'],
                file_cap=plan['budgets']['total'], cpu=plan['budgets']['cpu'])
            need(hot.owner.closed(entry) and entry['exit_code'] == 0, 'failed/unclosed ' + label)
            return out.read_text(), err.read_text()
        if action == 'counter-capability':
            entry, out, err = scope.run('counter-capability', plan['counter_capability']['command'],
                plan['environment'], P, wall=10, text_cap=16 << 10, rss_cap=256 << 20,
                total_cap=32 << 20, file_cap=32 << 20, cpu=20)
            need(hot.owner.closed(entry), 'capability owner unclosed')
            available = entry['exit_code'] == 0
            state.update(passed=available, complete=True, ownership_closed=True,
                counters_available=available, exit_code=entry['exit_code'],
                stderr=err.read_text(), stdout=out.read_text(),
                perf_event_paranoid=Path('/proc/sys/kernel/perf_event_paranoid').read_text().strip())
            verify(plan)
            return
        if action == 'build':
            need(not (B / 'cpu-compact-production-pool-timing-v2').exists(), 'retire ended V2fixture before another generation')
            _, error = run('compile', plan['compile'], plan['budgets']['build_wall']); need(not error, 'compile stderr')
            deps = dependencies(plan)
            text, error = run('link', plan['link'], plan['budgets']['build_wall']); need(not error, 'link stderr')
            allowed = {str(Path(p).resolve()): h for p, h in json.loads(Path(plan['external_inputs']['path']).read_text()).items()}
            allowed[str(ROOT / 'scale.o')] = sha(ROOT / 'scale.o'); links = {}
            for line in text.splitlines():
                p = Path(line.strip())
                if not p.is_absolute(): continue
                name = str(p.resolve()); need(name in allowed and sha(p) == allowed[name], 'uncovered link: ' + name)
                links[name] = allowed[name]
            need(len(links) >= 20, 'link census')
            text, error = run('ELF', [hot.READELF, '--file-header', '--program-headers', '--dynamic', str(ROOT / 'scale')], 10)
            libraries = re.findall(r'Shared library: \[([^\]]+)\]', text)
            need(not error and set(libraries) == {'libm.so.6', 'libstdc++.so.6', 'libgcc_s.so.1', 'libc.so.6'}, 'CPU-only libraries')
            def field(name):
                values = re.findall(r'^\s*' + re.escape(name) + r':\s*(.+)$', text, re.MULTILINE)
                need(len(values) == 1, 'ELF field'); return values[0].strip()
            interpreter = re.findall(r'\[Requesting program interpreter: ([^\]]+)\]', text)
            need(len(interpreter) == 1, 'ELF interpreter')
            elf = dict(elf_class=field('Class'), data=field('Data'), machine=field('Machine'),
                elf_type=field('Type'), interpreter=interpreter[0])
            need(elf == plan['expected_ELF'], 'ELF architecture/type/interpreter')
            state.update(binary=dict(path=str(ROOT / 'scale'), sha256=sha(ROOT / 'scale')),
                compiler_inputs=deps, links=links, ELF=elf, libraries=libraries)
        else:
            build = json.loads((ROOT / 'build.json').read_text())
            need(closed(build, plan) and build['compiler_inputs'] == dependencies(plan) and
                 sha(build['binary']['path']) == build['binary']['sha256'] and
                 all(sha(p) == h for p, h in build['links'].items()), 'exact source/build/executable required')
            arm = 'check' if action == 'check' else ORDER[int(action.split('-')[1])]
            if action != 'check':
                check = json.loads((ROOT / 'check.json').read_text())
                need(closed(check, plan) and check['binary'] == build['binary'], 'numerical pass first')
                index = int(action.split('-')[1])
                for j in range(6):
                    if j < index:
                        old = json.loads((ROOT / f'measure-{j}.json').read_text())
                        need(closed(old, plan) and old['binary'] == build['binary'], 'previous arm unclosed/failed')
                    elif j > index: need(not (ROOT / f'measure-{j}-attempt.json').exists(), 'later arm already invoked')
            results = []
            for row in plan['formats']:
                text, error = run('down' + str(row['down']), [build['binary']['path'], row['shard'],
                    str(row['offsets'][0]), str(row['offsets'][1]), arm],
                    plan['budgets']['check_wall' if action == 'check' else 'measure_wall'])
                need(not error, 'native stderr'); result = json.loads(text)
                if action == 'check': need(result == plan['expected_numeric'], 'bits/guards/control differ')
                else:
                    need(result['passed'] and result['arm'] == arm and result['cpu'] == 0 and
                         result['block_bytes'] == (138 if arm == 'expanded' else 134) and
                         len(result['cells']) == 3, 'actual arm/size/core')
                    seen = []
                    for cell in result['cells']:
                        seen.append(cell['working_set_experts'])
                        need(cell['calibration_s'] >= 2 and len(cell['samples']) == 3 and
                             all(len(s) == 2 and s[0] >= 1 and s[1] >= 0 and
                                 result['timer_pair_s'] / s[0] < .001 for s in cell['samples']), 'timing eligibility')
                        cell['median_seconds_per_expert'] = statistics.median(s[0] / cell['rounds'] for s in cell['samples'])
                    need(seen == [1, 6, 32], 'working-set coverage')
                results.append(dict(down=row['down'], **result))
            state.update(binary=build['binary'], results=results)
        verify(plan); state.update(passed=True, complete=True, ownership_closed=True)
    except BaseException as e:
        state.update(error=repr(e), ownership_closed=all(hot.owner.closed(c) for c in state['commands'])); raise
    finally:
        state['active'] = False; persist()


if __name__ == '__main__':
    need(sys.executable == hot.PYTHON and len(sys.argv) == 2 and
         sys.argv[1] in ('prepare', 'counter-capability', 'build', 'check', *(f'measure-{i}' for i in range(6))), 'qualified Python;prepare|counter-capability|build|check|measure-0..5')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); hot.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1])
