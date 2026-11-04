#!/usr/bin/env python3
"""Main-owned IQ3_XXS pair component gate; repository LICENSE.

Commit source, prepare, commit plan, build, commit, check, commit, then 0..5.
No GPU execution or production dispatch change. Never retry a closed stage.
"""
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(P / 'tools/sycl'))
import measurement_owner as owned
import measurement_guards as guards
import host_link_inventory as links

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'cpu-iq3-pair-component-v1'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/fused-moe-service-current.json'
KEY = 'cpu_iq3_pair_component_v1'
GGML = Path('/home/yayoi/ghq/github.com/ggml-org/llama.cpp/.worktree/dep-ggml-3cf-xe-e32-20261011')
GGML_REV = '3cf03257f219afbe7334045ff7c6a06ac68c627d'
SDK = B / 'prefill-expert-staging-v3/sdk-inputs.json'
SDK_SHA = 'c8879f0b8d772979bf2096dc64a3bba49853bd15dad221b46427a922030647d8'
CC = '/opt/intel/oneapi/compiler/2026.2/bin/icx'
CXX = '/opt/intel/oneapi/compiler/2026.2/bin/icpx'
METADATA = Path('/home/yayoi/.local/share/strata-sycl/packs/qwen3.8-flash-next-iq3_s/native_experts.txt')
MODEL = Path('/home/yayoi/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S')
ORDER = (0, 1, 1, 0, 0, 1)
TARGETS = ('cpu_iq_single_exact_test', 'cpu_iq3_pair')
PROJECT = ('bench/sycl/cpu_roofline', 'src/kernels/cpu', 'include', 'third_party/ggml',
    'tests/core/iq_single_exact_test.cpp', 'tools/sycl/measurement_owner.py',
    'tools/sycl/measurement_guards.py', 'tools/sycl/host_link_inventory.py')
need, digest = owned.require, owned.sha


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], timeout=30)


def pins(root, revision, prefixes):
    names = git(root, 'ls-tree', '-r', '--name-only', revision, '--', *prefixes).decode().splitlines()
    need(names and not git(root, 'ls-files', '--others', '--exclude-standard', '--', *prefixes).strip(),
         'empty scope or uncommitted measurement input')
    result = {}
    for at in range(0, len(names), 64):
        batch = names[at:at + 64]
        raw = subprocess.check_output(['git', '-C', str(root), 'cat-file', '--batch'],
            input=''.join(revision + ':' + n + '\n' for n in batch).encode(), timeout=30)
        pos = 0
        for name in batch:
            end = raw.index(b'\n', pos)
            header = raw[pos:end].split()
            need(len(header) == 3 and header[1] == b'blob', 'recover original source: ' + name)
            size = int(header[2]); content = raw[end + 1:end + 1 + size]
            pos = end + 2 + size
            need(raw[pos - 1:pos] == b'\n' and (root / name).read_bytes() == content,
                 'changed/uncommitted source: ' + name)
            result[str((root / name).resolve())] = hashlib.sha256(content).hexdigest()
        need(pos == len(raw), 'Git batch extent')
    return result


def machine():
    return dict(boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        kernel=os.uname().release, affinity=sorted(os.sched_getaffinity(0)),
        cpuinfo_sha256=hashlib.sha256('\n'.join(s for s in Path('/proc/cpuinfo').read_text().splitlines()
            if s.startswith(('model name', 'microcode', 'flags'))).encode()).hexdigest())


def identity(path):
    s = Path(path).stat()
    return dict(path=str(Path(path).resolve()), dev=s.st_dev, ino=s.st_ino,
        bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def fixture_hash(row):
    before = identity(row['shard']); h = hashlib.sha256()
    with Path(row['shard']).open('rb', buffering=0) as f:
        for expert in range(252):
            for offset, n in ((row['gate'], 627200), (row['up'], 627200),
                              (row['down'], 921600)):
                start = offset + expert * n
                need(0 <= start and start + n <= before['bytes'], 'model fixture bounds')
                f.seek(start); raw = f.read(n)
                need(len(raw) == n, 'model fixture short read'); h.update(raw)
    need(identity(row['shard']) == before, 'model changed during hash')
    return h.hexdigest()


def compact(row):
    out = {k:v for k,v in row.items() if k not in ('project_inputs', 'ggml_inputs', 'commands')}
    for key in ('project_inputs', 'ggml_inputs'):
        out[key] = dict(files=len(row[key]), aggregate_sha256=fingerprint(row[key]))
    return out


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); write(RESULT, data)


def prepare():
    need(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'one fresh component plan')
    revision = git(P, 'rev-parse', 'HEAD').decode().strip()
    need(git(GGML, 'rev-parse', 'HEAD').decode().strip() == GGML_REV, 'official GGML revision')
    project, dependency = pins(P, revision, PROJECT), pins(GGML, GGML_REV, ('ggml',))
    need(digest(SDK) == SDK_SHA, 'existing official SDK inventory changed')
    allowed = json.loads(SDK.read_text())
    need(len(allowed) == 10945 and all(digest(p) == sha for p,sha in allowed.items()), 'SDK drift')
    env = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
        LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.2/lib', OMP_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', STRATA_IQ_SINGLE_EXACT='0',
        TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    host = links.inventory(CXX, env)
    allowed.update(host['files'])
    tools = {str(Path(p).resolve()):digest(p) for p in
        (CC, CXX, '/usr/bin/cmake', '/usr/bin/ninja', '/usr/bin/git', '/usr/bin/ldd', sys.executable)}
    allowed.update(tools)
    for directory in Path('/usr/share').glob('cmake-*/Modules'):
        for path in directory.rglob('*'):
            if path.is_file(): allowed[str(path.resolve())] = digest(path)
    # Reuse one real layer, with explicit format and geometry; no copied model fixture.
    lines = METADATA.read_text().splitlines()
    default = lines[0].split('absolute offsets in ')[1].split(',')[0]
    fields = next(line.split() for line in lines if line and not line.startswith('#')
                  and list(map(int, line.split()[1:3])) == [18, 20])
    layer, gu, down, offset, size, gate, up, dn = map(int, fields[:8])
    need(size == 2176000 and gu == 18 and down == 20, 'actual IQ3_XXS/IQ4_NL layout')
    fixture = dict(layer=layer, shard=str(MODEL / (fields[8] if len(fields) == 9 else default)),
        gate=gate, up=up, down=dn, blob_bytes=size, total_experts=512, measured_experts=252)
    fixture.update(file_identity=identity(fixture['shard']), sha256=fixture_hash(fixture))
    ROOT.mkdir(mode=0o700); (ROOT / 'tmp').mkdir(); (ROOT / 'cache').mkdir()
    write(ROOT / 'external-inputs.json', allowed)
    short = git(GGML, 'rev-parse', '--short', 'HEAD').decode().strip()
    need(not git(GGML, 'diff', '--name-only', 'HEAD', '--', 'ggml').strip(), 'dirty GGML source')
    header = '#pragma once\n\n#define GGML_VERSION "0.24.0"\n#define GGML_COMMIT  "' + short + '"\n'
    row = dict(stage='prepared', source_commit=revision, project_inputs=project, ggml_inputs=dependency,
        dependency=dict(origin='https://github.com/ggml-org/llama.cpp', revision=GGML_REV),
        official_sdk=dict(path=str(SDK), sha256=SDK_SHA, files=10945), host_link_inputs=host,
        external_inputs=dict(path=str(ROOT / 'external-inputs.json'), sha256=digest(ROOT / 'external-inputs.json'), files=len(allowed)),
        tools=tools, machine=machine(), environment=env, metadata_sha256=digest(METADATA), fixture=fixture,
        generated_expected={str(ROOT / 'build/ggml/src/ggml-version.h'):hashlib.sha256(header.encode()).hexdigest()},
        configure=['/usr/bin/cmake', '-S', str(P / 'bench/sycl/cpu_roofline'), '-B', str(ROOT / 'build'),
            '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
            '-DCMAKE_C_COMPILER=' + CC, '-DCMAKE_CXX_COMPILER=' + CXX,
            '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-t', '-DGGML_CCACHE=OFF', '-DSTRATA_GGML_DIR=' + str(GGML)],
        build=['/usr/bin/cmake', '--build', str(ROOT / 'build'), '--target', *TARGETS, '--parallel', '4'],
        order=list(ORDER), processes=[], commands=[], active=False, ownership_closed=True,
        passed=False, complete=False, GPU_executed=False, production_selected=False,
        protocol=dict(question='Does sharing packed IQ3_XXS gate/up activation loads lower actual NT1 GU cost?',
            correctness='Same-library fixture must exit0,not ISA-skip77:384 original cases,80 pair cases,GGML bit parity,finite outputs,padded/unaligned inputs,canaries and actual alias/refusal target snapshots. Each measured process also validates all252 actual experts/four activations/full640 and128 rows before/after timing and checks every sample output.',
            comparison='Single binary,six independent processes off/on,on/off,off/on. Eight cells:640/128rows,prequant/one production quantization per six experts,reuse6/stream252. Direct candidate entry includes bounds/layout checks;reference is actual native_gu_rows/decode NT1 with SINGLE_EXACT0. Not production-pool dispatch.',
            timing='Four warm passes;five samples per cell,target0.35s,floor0.25s,cap1e6passes,timer<0.1percent fastest warm pass. Setup/load/hash/reference outside timing;pack cells include activation quantization. Preserve all individual values and all process results.',
            decision='For each cell,three candidate/reference median ns-per-expert directions separately;all three<1 supports component gain,no arbitrary minimum. Require identical input fingerprints,geometry and actual pinned core across all6. No model trial follows from a cell that failed its performance gate.',
            stopping='Stop first source,ELF,ownership,ISA,numerical,nonfinite,canary,timer,duration,budget or input mismatch. No retry,replacement,hash refresh or changed criteria. Keep original failure.',
            limits='Logical footprint is not physical DRAM traffic/saturation. No model speed,quality,PP preservation,70TG,262144/RESTORE or adoption qualification.'),
        retention='Auxiliary CPU-only build,not a third inference generation. Keep two named CPU ELFs/library/objects only through this immediate correctness/comparison;after compact evidence and exact source/recipe recovery,retire successful output/build. Preserve479a fallback and live observer consumer unchanged.',
        budgets=dict(build_wall=300, build_cpu=1200, build_rss=8 << 30, check_wall=60,
            measure_wall=180, measure_rss=1 << 30,
            text_cap=1 << 20, file_cap=16 << 20, total_cap=384 << 20))
    save(row)
    print('Prepared CPU-only pair plan;commit before compilation.', flush=True)


def load():
    row = json.loads((ROOT / 'record.json').read_text())
    data = json.loads(git(P, 'show', 'HEAD:' + str(RESULT.relative_to(P))))
    need(data[KEY] == compact(row), 'commit exact component state before next stage')
    need(pins(P, row['source_commit'], PROJECT) == row['project_inputs'] and
         pins(GGML, GGML_REV, ('ggml',)) == row['ggml_inputs'], 'source drift')
    need(machine() == row['machine'] and digest(METADATA) == row['metadata_sha256'] and
         identity(row['fixture']['shard']) == row['fixture']['file_identity'], 'machine/model drift')
    need(all(digest(p) == h for p,h in row['tools'].items()), 'compiler/tool drift')
    for binary in row.get('binaries', {}).values():
        need(digest(binary['path']) == binary['sha256'], 'CPU ELF drift')
    if 'same_build_library' in row:
        need(digest(row['same_build_library']['path']) == row['same_build_library']['sha256'], 'shared CPU archive drift')
    need(all(digest(p) == h for p,h in row.get('runtime_inputs', {}).items()), 'CPU runtime library drift')
    return row


def input_map(row):
    asset = row['external_inputs']
    need(digest(asset['path']) == asset['sha256'], 'external inventory drift')
    allowed = json.loads(Path(asset['path']).read_text())
    need(len(allowed) == asset['files'] and all(digest(p) == h for p,h in allowed.items()), 'external input drift')
    allowed.update(row['project_inputs']); allowed.update(row['ggml_inputs'])
    allowed.update(row['generated_expected'])
    return allowed


def receipt(entry):
    return {k:entry[k] for k in ('pid', 'direct_identity', 'exit_code', 'elapsed_seconds', 'observation_complete',
        'direct_child_reaped', 'session_empty', 'session_anchored', 'errors', 'cleanup', 'peak_session_rss_bytes')}


def build_check(row, root):
    allowed = input_map(row)
    paths = {str((Path(line.strip()) if Path(line.strip()).is_absolute() else
        ROOT / 'build' / line.strip()).resolve()) for line in
        (root / 'deps.stdout').read_text().splitlines() if line.startswith('    ') and line.strip()}
    need(len(paths) > 150 and all(allowed.get(p) == digest(p) for p in paths),
        'missing/unplanned actual compiler inputs: ' + str([p for p in paths if allowed.get(p) != digest(p)][:5]))
    need(all(p in paths for p in (str(P / 'src/kernels/cpu/iq_single_avx2.cpp'),
        str(P / 'tests/core/iq_single_exact_test.cpp'), str(P / 'bench/sycl/cpu_roofline/iq3_pair.cpp'))),
        'required real production/fixture units absent')
    units = json.loads((ROOT / 'build/compile_commands.json').read_text())
    for name in ('src/kernels/cpu/iq_single_avx2.cpp', 'tests/core/iq_single_exact_test.cpp',
                 'bench/sycl/cpu_roofline/iq3_pair.cpp'):
        matches = [x for x in units if x['file'] == str(P / name)]
        need(len(matches) == 1, 'unique compile unit: ' + name)
        argv = shlex.split(matches[0]['command'])
        need(Path(argv[0]).resolve() == Path(CXX).resolve() and '-O3' in argv and
            '-march=znver3' in argv and '-DNDEBUG' in argv and not any(x in argv
                for x in ('-ffast-math', '-Ofast', '-fsycl')), 'unplanned CPU flags')
    cache = (ROOT / 'build/CMakeCache.txt').read_text()
    need(all(re.search('^' + key + ':BOOL=OFF$', cache, re.M) for key in
        ('GGML_CUDA', 'GGML_HIP', 'GGML_SYCL', 'GGML_OPENMP', 'GGML_BACKEND_DL')),
        'CPU-only configuration required')
    frozen, generated = {}, set()
    for line in (root / 'compile.stdout').read_text().splitlines():
        name = line.strip(); path = Path(name)
        if not name or any(c.isspace() for c in name) or not (path.is_absolute() or path.suffix in ('.o', '.a') or '.so' in path.name):
            continue
        path = (path if path.is_absolute() else ROOT / 'build' / path).resolve()
        if path.is_relative_to(ROOT / 'build') or path.is_relative_to(ROOT / 'tmp'):
            generated.add(str(path))
        else:
            need(path.is_file() and allowed.get(str(path)) == digest(path), 'unplanned external linker input: ' + str(path))
            frozen[str(path)] = digest(path)
    need(len(frozen) > 8, 'actual external linker trace missing')
    write(ROOT / 'compiler-inputs.json', {p:allowed[p] for p in sorted(paths)})
    return dict(compiler_inputs=dict(path=str(ROOT / 'compiler-inputs.json'),
        sha256=digest(ROOT / 'compiler-inputs.json'), files=len(paths), uncovered=0),
        compile_database_sha256=digest(ROOT / 'build/compile_commands.json'),
        external_link_inputs=frozen, owned_link_products=len(generated))


def decision(processes):
    need(len(processes) == 6 and [p['arm'] for p in processes] == list(ORDER), 'complete original six-process order')
    need(len({(p['receipt']['pid'], p['receipt']['direct_identity']['start_ticks'])
        for p in processes}) == 6, 'independent native processes required')
    need(len({(p['input']['weight_input_fingerprint_fnv1a64'], p['input']['activation_input_fingerprint_fnv1a64'],
        p['input']['CPU'], p['input']['experts'], p['input']['blob_bytes']) for p in processes}) == 1,
        'actual input/core/geometry mismatch')
    cells = []
    for rows in (640, 128):
        for pack in (0, 1):
            for stream in (0, 1):
                medians = []
                for p in processes:
                    samples = [s for s in p['samples'] if (s['rows'], s['pack'], s['stream']) == (rows, pack, stream)]
                    need(len(samples) == 5 and [s['rep'] for s in samples] == list(range(5)), 'five unique samples')
                    medians.append(statistics.median(s['wall_s'] * 1e9 / s['expert_calls'] for s in samples))
                ratios = []
                for start in (0, 2, 4):
                    pair = {ORDER[i]:medians[i] for i in (start, start + 1)}
                    ratios.append(pair[1] / pair[0])
                cells.append(dict(rows=rows, pack=pack, stream=stream, process_median_ns_per_expert=medians,
                    paired_candidate_over_reference=ratios, geometric_mean_ratio=math.exp(statistics.mean(map(math.log, ratios))),
                    component_gain_qualified=all(x < 1 for x in ratios)))
    return dict(cells=cells, complete=True, inference_adopted=False, model_trial_qualified=False,
        next='Only winning production-relevant cells may motivate an explicit decode-only production-dispatch plan;this direct component does not qualify model gain.')


def execute(stage):
    row = load(); frozen = copy.deepcopy(row)
    need(not row['active'] and row['ownership_closed'] and row['stage'] != 'failed', 'closed healthy prior stage required')
    index = int(stage) if stage.isdigit() else None
    need(stage in ('build', 'check') or index in range(6), 'use prepare|build|check|0..5')
    need((stage == 'build' and row['stage'] == 'prepared') or
         (stage == 'check' and row['stage'] == 'built') or
         (index is not None and row['stage'] == 'checked' and len(row['processes']) == index), 'no stage retry/replacement')
    root = ROOT / ('owned-' + stage)
    need(not root.exists(), 'inspect existing stage receipt/handle;never duplicate')
    def persist():
        write(ROOT / 'execution.json', dict(stage=stage, active=bool(owner.active), commands=owner.commands))
    owner = owned.Owner(root, persist=persist)
    deadline = time.monotonic() + (300 if stage == 'build' else 240)
    def run(label, argv, wall, *, cpu=180, rss=512 << 20):
        need(time.monotonic() < deadline, 'whole stage deadline')
        entry, out, err = owner.run(label, argv, row['environment'], ROOT,
            wall=min(wall, deadline - time.monotonic()), cpu=cpu,
            rss_cap=rss, text_cap=row['budgets']['text_cap'], file_cap=row['budgets']['file_cap'], total_cap=row['budgets']['total_cap'])
        need(owned.closed(entry) and entry['exit_code'] == 0, label + ' failed: ' + err.read_text(errors='replace')[-1500:])
        return entry, out, err
    try:
        if stage == 'build':
            allowed = input_map(row)
            run('configure', row['configure'], 60, cpu=120, rss=2 << 30)
            run('compile', row['build'], 240, cpu=1200, rss=8 << 30)
            run('deps', ['/usr/bin/ninja', '-C', str(ROOT / 'build'), '-t', 'deps'], 20)
            row['build_evidence'] = build_check(row, root)
            row['binaries'] = {name:dict(path=str(ROOT / 'build' / name), sha256=digest(ROOT / 'build' / name)) for name in TARGETS}
            row['runtime_inputs'] = {}
            for name in TARGETS:
                _, out, _ = run('links-' + name, ['/usr/bin/ldd', str(ROOT / 'build' / name)], 10)
                need(not any(v in out.read_text().lower() for v in ('sycl', 'libze_', 'cuda', 'amdhip', 'libur_')), 'GPU runtime linked')
                for path in re.findall(r'(/[^\s]+)', out.read_text()):
                    resolved = str(Path(path).resolve())
                    need(allowed.get(resolved) == digest(resolved), 'unfrozen CPU shared library: ' + resolved)
                    row['runtime_inputs'][resolved] = digest(resolved)
            row['same_build_library'] = dict(path=str(ROOT / 'build/libexpert_cpu.a'),
                sha256=digest(ROOT / 'build/libexpert_cpu.a'))
            row['stage'] = 'built'
        elif stage == 'check':
            entry, out, err = run('correctness', [row['binaries'][TARGETS[0]]['path']], 60, cpu=60)
            need(not err.stat().st_size and out.read_text().strip() ==
                'IQ single exact PASS: 384 single and 80 paired cases, bitwise GGML parity and row canaries',
                'complete GGML/paired fixture required;skip77 is not a pass')
            row.update(stage='checked', correctness=dict(passed=True, single_cases=384, paired_cases=80,
                source='tests/core/iq_single_exact_test.cpp', receipt=receipt(entry)))
        else:
            f = row['fixture']
            need(fixture_hash(f) == f['sha256'], 'actual model range drift')
            argv = [row['binaries']['cpu_iq3_pair']['path'], f['shard'], *[str(f[k]) for k in
                ('gate', 'up', 'down', 'blob_bytes', 'total_experts')], str(ORDER[index]), str(index)]
            entry, out, err = run('measure', argv, row['budgets']['measure_wall'],
                rss=row['budgets']['measure_rss'])
            need(not err.stat().st_size, 'CPU component stderr')
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            samples = [v for v in rows if v['kind'] == 'sample']
            validation = [v for v in rows if v['kind'] == 'validation']
            inputs = [v for v in rows if v['kind'] == 'input']
            need(len(inputs) == 1 and len(validation) == 2 and all(v['cases'] == 2016 and v['bitwise'] and
                v['finite'] and v['inputs_unchanged'] for v in validation) and len(samples) == 40 and
                rows[-1] == dict(kind='closed', passed=True), 'complete actual-model component output required')
            need(inputs[0]['arm'] == ORDER[index] and inputs[0]['process'] == index and
                inputs[0]['experts'] == 252 and inputs[0]['group'] == 6 and
                inputs[0]['blob_bytes'] == f['blob_bytes'], 'actual native input identity')
            if row['processes']:
                first = row['processes'][0]['input']
                need(all(first[k] == inputs[0][k] for k in ('CPU', 'experts', 'blob_bytes',
                    'weight_input_fingerprint_fnv1a64', 'activation_input_fingerprint_fnv1a64')),
                    'stop first actual input/core/fingerprint mismatch')
            need(all(s['arm'] == ORDER[index] and s['process'] == index and s['wall_s'] >= .25 and
                s['passes'] < 1000000 and s['timer_s'] < .001 * s['fastest_warm_pass_s'] and
                s['outputs_match_untimed_reference'] for s in samples), 'sample identity/duration/output gate')
            need(fixture_hash(f) == f['sha256'], 'post-run model range drift')
            row['processes'].append(dict(index=index, arm=ORDER[index], input=inputs[0], samples=samples,
                validation_cases=2016, validation_before_and_after=True, passed=True, receipt=receipt(entry)))
            if index == 5:
                row.update(stage='closed', decision=decision(row['processes']), complete=True, passed=True)
        need(pins(P, frozen['source_commit'], PROJECT) == frozen['project_inputs'] and
             pins(GGML, GGML_REV, ('ggml',)) == frozen['ggml_inputs'] and machine() == frozen['machine'], 'post-stage source/machine drift')
        need(time.monotonic() < deadline, 'whole stage deadline')
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, error=repr(error))
        raise
    finally:
        row.update(active=bool(owner.active), ownership_closed=owner.active is None and all(owned.closed(e) for e in owner.commands),
            commands=row['commands'] + owner.commands)
        save(row)
    print('CPU pair stage ' + stage + ' closed;commit before next stage.', flush=True)


if __name__ == '__main__':
    need(len(sys.argv) == 2, 'use prepare|build|check|0..5')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        else: execute(sys.argv[1])
