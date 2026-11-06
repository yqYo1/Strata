#!/usr/bin/env python3
"""Main-owned IQ3_S direct-component comparison; repository LICENSE.

Commit source, prepare/commit, build/commit, check/commit, then 0..5/commit.
Reuse the existing CPU owner/input/link checks privately, without selecting the
candidate in production. Preparation waits for the caller cohort to close.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('strata_iq3s_component_support',
    Path(__file__).with_name('iq3_pair.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.ROOT = base.B / 'cpu-iq3s-pair-component-v1'
base.KEY = 'cpu_iq3s_pair_component_v1'
base.TARGETS = ('cpu_iq_single_exact_test', 'cpu_iq3s_pair', 'cpu_service_pool_test')
ROOT, KEY, need, digest = base.ROOT, base.KEY, base.need, base.digest


def fixture_hash(fixture):
    before = base.identity(fixture['shard'])
    result = hashlib.sha256()
    with Path(fixture['shard']).open('rb', buffering=0) as stream:
        for expert in range(252):
            for key, size in (('gate', 704000), ('up', 704000), ('down', 921600)):
                offset = fixture[key] + expert * size
                need(0 <= offset <= before['bytes'] - size, 'IQ3_S fixture extent')
                stream.seek(offset); raw = stream.read(size)
                need(len(raw) == size, 'IQ3_S fixture short read'); result.update(raw)
    need(base.identity(fixture['shard']) == before, 'fixture changed during hash')
    return result.hexdigest()


def prepare():
    need(not ROOT.exists() and KEY not in json.loads(base.RESULT.read_text()), 'fresh component plan required')
    revision = base.git(P, 'rev-parse', 'HEAD').decode().strip()
    data = json.loads(base.git(P, 'show', 'HEAD:' + str(base.RESULT.relative_to(P))))
    caller = data['decode_service_overhead_v3']
    need(caller['stage'] == 'closed' and caller['complete'] and caller['passed'] and
        not caller['active'] and caller['ownership_closed'] and len(caller['processes']) == 6,
        'close the frozen six-process caller cohort first')
    audit_path = P / 'bench/results/2026-10-15-pr6-recovery-audit/research-index.json'
    need(audit_path.read_bytes() == base.git(P, 'show', 'HEAD:' + str(audit_path.relative_to(P))),
        'committed source audit required')
    audit = json.loads(audit_path.read_text())['iq3s_pair_contract_20261106']
    need(audit['verdict'] == 'accepted_as_conditional_source_decision_support' and
        all(digest(P / name) == sha for name, sha in audit['source_hashes'].items()),
        'unchanged conditionally reviewed IQ3_S source required')
    need(base.git(base.GGML, 'rev-parse', 'HEAD').decode().strip() == base.GGML_REV,
        'official GGML revision')
    project = base.pins(P, revision, base.PROJECT)
    dependency = base.pins(base.GGML, base.GGML_REV, ('ggml',))
    need(digest(base.SDK) == base.SDK_SHA, 'official SDK inventory changed')
    allowed = json.loads(base.SDK.read_text())
    need(len(allowed) == 10945 and all(digest(p) == sha for p, sha in allowed.items()), 'SDK drift')
    env = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
        LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.2/lib', OMP_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', STRATA_IQ_SINGLE_EXACT='0',
        STRATA_IQ3_PAIR='0', STRATA_IQ3_PAIR_AUDIT='0', STRATA_CPU_SERVICE='0',
        TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    host = base.host_inventory(env); allowed.update(host['files'])
    tools = {str(Path(p).resolve()):digest(p) for p in
        (base.CC, base.CXX, '/usr/bin/cmake', '/usr/bin/ninja', '/usr/bin/git',
         '/usr/bin/ldd', '/usr/bin/nm', '/usr/bin/objdump', sys.executable)}
    allowed.update(tools)
    for directory in Path('/usr/share').glob('cmake-*/Modules'):
        for path in directory.rglob('*'):
            if path.is_file(): allowed[str(path.resolve())] = digest(path)
    lines = base.METADATA.read_text().splitlines()
    default = lines[0].split('absolute offsets in ')[1].split(',')[0]
    fields = next(line.split() for line in lines if line and not line.startswith('#')
        and line.split()[1:3] == ['21', '20'])
    layer, gu, down, offset, size, gate, up, dn = map(int, fields[:8])
    need(size == 2329600 and gu == 21 and down == 20, 'IQ3_S/IQ4_NL layout')
    fixture = dict(layer=layer, shard=str(base.MODEL / (fields[8] if len(fields) == 9 else default)),
        gate=gate, up=up, down=dn, blob_bytes=size, total_experts=512, measured_experts=252)
    fixture.update(file_identity=base.identity(fixture['shard']), sha256=fixture_hash(fixture))
    ROOT.mkdir(mode=0o700); (ROOT / 'tmp').mkdir(); (ROOT / 'cache').mkdir()
    base.write(ROOT / 'external-inputs.json', allowed)
    short = base.git(base.GGML, 'rev-parse', '--short', 'HEAD').decode().strip()
    header = '#pragma once\n\n#define GGML_VERSION "0.24.0"\n#define GGML_COMMIT  "' + short + '"\n'
    exact = 'IQ single exact PASS: 384 single and 80 paired cases, bitwise GGML parity and row canaries'
    checks = [dict(label='exact-default', target='cpu_iq_single_exact_test', args=[], arm=0,
        audit=0, expected=exact), dict(label='exact-iq3s', target='cpu_iq_single_exact_test',
        args=['--iq3s-pair'], arm=0, audit=0, expected=exact +
        '\nIQ3_S pair exact PASS: 80 cases,512 grids,256 high/sign bytes,16 scales')]
    for arm in (0, 1):
        checks.extend([dict(label='dispatch-' + str(arm), target='cpu_iq_single_exact_test',
            args=['--dispatch'], arm=arm, audit=1, expected=exact +
            f'\nIQ3 production dispatch PASS arm={arm} calls=80 nt1_rows=20480 pair_rows={10240 * arm} prefill_pair_rows=0'),
            dict(label='pool-' + str(arm), target='cpu_service_pool_test', args=['0'], arm=arm, audit=1)])
    row = dict(stage='prepared', source_commit=revision, project_inputs=project, ggml_inputs=dependency,
        dependency=dict(origin='https://github.com/ggml-org/llama.cpp', revision=base.GGML_REV),
        source_audit=dict(path=str(audit_path.relative_to(P)), source_revision=audit['source_revision'],
            source_hashes=audit['source_hashes'], verdict=audit['verdict']),
        official_sdk=dict(path=str(base.SDK), sha256=base.SDK_SHA, files=10945), host_link_inputs=host,
        external_inputs=dict(path=str(ROOT / 'external-inputs.json'),
            sha256=digest(ROOT / 'external-inputs.json'), files=len(allowed)),
        tools=tools, machine=base.machine(), environment=env, metadata_sha256=digest(base.METADATA),
        fixture=fixture, generated_expected={str(ROOT / 'build/ggml/src/ggml-version.h'):
            hashlib.sha256(header.encode()).hexdigest()},
        configure=['/usr/bin/cmake', '-S', str(P / 'bench/sycl/cpu_roofline'), '-B', str(ROOT / 'build'),
            '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
            '-DCMAKE_C_COMPILER=' + base.CC, '-DCMAKE_CXX_COMPILER=' + base.CXX,
            '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-t', '-DGGML_CCACHE=OFF', '-DSTRATA_GGML_DIR=' + str(base.GGML)],
        build=['/usr/bin/cmake', '--build', str(ROOT / 'build'), '--target', *base.TARGETS, '--parallel', '4'],
        check_plan=checks, order=list(base.ORDER), processes=[], commands=[], active=False,
        ownership_closed=True, passed=False, complete=False, GPU_executed=False, production_selected=False,
        protocol=dict(question='Does sharing activation loads/masks improve IQ3_S NT1 GU while keeping packed weights and the GGML per-output float tree?',
            correctness='Before timing:one same-library default exact fixture,one --iq3s-pair fixture,old --dispatch off/on and real-pool off/on in six immutable pre-exec children. Exact384single+80XXS+80IQ3_S cases with finite bit parity,edge/grid/sign/high/scale coverage,canaries and layout/alias/overflow refusals. Both existing pool checksums must match. Skip77 is failure. Every timing process checks2016 real-model GU cases before/after and all timed outputs.',
            comparison='Direct component only:linked GGML IQ3_S two-dot GU versus iq3s_pair_gu_rows. No production selection/audit in either timed arm. Eight cells:640/128rows,prequant/one production quantization per six experts,reuse6/stream252. Same benchmark ELF,core,model slices,activations and flags;explicit arm argv0/1 in six independent0/1,1/0,0/1 processes. Direct-call/row counters and helper success bind the actual candidate. This does not measure production dispatch or pool scheduling.',
            timing='Four warm passes;five samples/cell,target0.35s,floor0.25s,cap1e6passes,timer<0.1percent fastest warm pass. Setup/load/hash/reference outside timing;pack cells include activation quantization. Keep all40samples/process.',
            ISA='CPU-only icpx2026.2 O3,NDEBUG,znver3,AVX2/FMA/F16C;no fast-math,SYCL or GPU runtime. Inspect actual IQ3_S pair symbol disassembly and require AVX2 integer multiply-add,FMA and vzeroupper,with no ZMM/mask registers in those symbols. Successful fixtures exercise the actual Zen3 path.',
            decision='Per cell:all three candidate/reference process-median ns/expert ratios<1 supports direct-component gain;no arbitrary minimum. Any input/core mismatch fails. No production-entry/pool/model trial follows from a cell failing its component performance gate.',
            stopping='Stop first source,provider,ELF,ownership,ISA,numerical,nonfinite,canary,timer,duration,budget or input failure. No retries,replacements,hash refresh,exclusions or revised criteria.',
            limits='Logical weight coverage is not DRAM/PCIe traffic or saturation. Direct-component gain is not model speed,PP preservation,70TG,quality,262144/RESTORE or adoption.'),
        retention='One temporary CPU-only auxiliary build,not another inference generation. Keep its three ELFs/archive/objects only through this immediate fixture/comparison;then recover committed exact inputs/recipe,curate useful samples/outcome and retire build/raw logs. Preserve both named inference generations and their14ELFs.',
        budgets=dict(build_wall=300, build_cpu=1200, build_rss=8 << 30, check_wall=60,
            check_text_cap=256 << 10, measure_wall=180, measure_rss=1 << 30,
            text_cap=1 << 20, file_cap=16 << 20, total_cap=384 << 20))
    base.save(row); print('Prepared IQ3_S component;commit before CPU-only build.', flush=True)


def isa_check(row, run):
    binary = row['binaries']['cpu_iq3s_pair']['path']
    _, out, _ = run('symbols', ['/usr/bin/nm', '--defined-only', binary], 10)
    symbols = [line.split()[-1] for line in out.read_text().splitlines()
        if 'iq3s_pair_' in line and '.cold' not in line]
    need(symbols and len(symbols) <= 8 and any('iq3s_pair_gu_rows' in s for s in symbols),
        'actual IQ3_S pair symbols required')
    text = ''
    for index, symbol in enumerate(symbols):
        _, out, _ = run('isa-' + str(index), ['/usr/bin/objdump', '-d',
            '--no-show-raw-insn', '--disassemble=' + symbol, binary], 10)
        text += out.read_text()
    need(not re.search(r'%zmm\d|%k[0-7]\b', text) and
        all(re.search(pattern, text) for pattern in
            (r'\bvpmaddubsw\b', r'\bvpmaddwd\b', r'\bvfmadd\w*\b', r'\bvzeroupper\b')),
        'admitted IQ3_S AVX2/FMA/upper-lane cleanup ISA required')
    return dict(passed=True, symbols=symbols, disassembly_sha256=hashlib.sha256(text.encode()).hexdigest(),
        AVX2_integer_madd=True, FMA=True, vzeroupper=True, pair_symbols_have_ZMM_or_mask=False,
        scope='Actual IQ3_S pair symbols only;other unused AVX512 library functions are outside this claim.')


def execute(stage):
    row = base.load(); frozen = copy.deepcopy(row)
    need(not row['active'] and row['ownership_closed'] and row['stage'] != 'failed', 'closed healthy stage required')
    index = int(stage) if stage.isdigit() else None
    need((stage == 'build' and row['stage'] == 'prepared') or
        (stage == 'check' and row['stage'] == 'built') or
        (index in range(6) and row['stage'] == 'checked' and len(row['processes']) == index),
        'ordered fresh stage only;no retry/replacement')
    root = ROOT / ('owned-' + stage); need(not root.exists(), 'inspect existing owner before retry')
    def persist():
        base.write(ROOT / 'execution.json', dict(stage=stage, active=bool(owner.active), commands=owner.commands))
    owner = base.owned.Owner(root, persist=persist)
    deadline = time.monotonic() + (row['budgets']['build_wall'] if stage == 'build' else 240)
    def run(label, argv, wall, *, cpu=180, rss=512 << 20, text=None, environment=None):
        need(time.monotonic() < deadline, 'stage deadline')
        entry, out, err = owner.run(label, argv, row['environment'] if environment is None else environment,
            ROOT, wall=min(wall, deadline - time.monotonic()), cpu=cpu, rss_cap=rss,
            text_cap=text if text is not None else row['budgets']['text_cap'],
            file_cap=row['budgets']['file_cap'], total_cap=row['budgets']['total_cap'])
        need(base.owned.closed(entry) and entry['exit_code'] == 0,
            label + ' failed: ' + err.read_text(errors='replace')[-1500:])
        return entry, out, err
    try:
        if stage == 'build':
            need(not (base.B / 'cpu-iq3-pair-dispatch-v3/build').exists(), 'retire closed auxiliary comparison before build')
            allowed = base.input_map(row)
            run('configure', row['configure'], 60, cpu=120, rss=2 << 30)
            run('compile', row['build'], 240, cpu=row['budgets']['build_cpu'], rss=row['budgets']['build_rss'])
            run('deps', ['/usr/bin/ninja', '-C', str(ROOT / 'build'), '-t', 'deps'], 20)
            row['build_evidence'] = base.build_check(row, root, iq3s=True)
            row['binaries'] = {name:dict(path=str(ROOT / 'build' / name),
                sha256=digest(ROOT / 'build' / name)) for name in base.TARGETS}
            row['runtime_inputs'] = {}
            for name, binary in row['binaries'].items():
                _, out, _ = run('links-' + name, ['/usr/bin/ldd', binary['path']], 10)
                need(not any(word in out.read_text().lower() for word in
                    ('sycl', 'libze_', 'cuda', 'amdhip', 'libur_')), 'GPU runtime linked')
                for path in re.findall(r'(/[^\s]+)', out.read_text()):
                    resolved = str(Path(path).resolve())
                    need(allowed.get(resolved) == digest(resolved), 'unfrozen runtime: ' + resolved)
                    row['runtime_inputs'][resolved] = digest(resolved)
            row['same_build_library'] = dict(path=str(ROOT / 'build/libexpert_cpu.a'),
                sha256=digest(ROOT / 'build/libexpert_cpu.a'))
            row['stage'] = 'built'
        elif stage == 'check':
            row['ISA'] = isa_check(row, run)
            checks, pool = [], []
            for case in row['check_plan']:
                env = dict(row['environment'], STRATA_IQ3_PAIR=str(case['arm']),
                    STRATA_IQ3_PAIR_AUDIT=str(case['audit']))
                entry, out, err = run(case['label'], [row['binaries'][case['target']]['path'], *case['args']],
                    row['budgets']['check_wall'], cpu=60, text=row['budgets']['check_text_cap'], environment=env)
                need(not err.stat().st_size, 'fixture stderr')
                text = out.read_text().strip()
                if 'expected' in case: need(text == case['expected'], 'complete exact fixture;skip77 is not a pass')
                else:
                    match = re.fullmatch(r'cpu_service_pool PASS observer=0 formats=7 nt=1\.\.4 cases=28 checksum_fnv1a64_le=([0-9a-f]{16})', text)
                    need(match is not None, 'complete pool fixture'); pool.append(match[1])
                checks.append(dict(label=case['label'], passed=True, receipt=base.receipt(entry)))
            need(len(pool) == 2 and pool[0] == pool[1], 'existing pool selector output changed')
            row.update(stage='checked', correctness=dict(passed=True, IQ3_S_pair_cases=80,
                default_single_cases=384, default_IQ3_XXS_pair_cases=80, pool_checksum=pool[0], checks=checks))
        else:
            f = row['fixture']; need(fixture_hash(f) == f['sha256'], 'model slice drift')
            argv = [row['binaries']['cpu_iq3s_pair']['path'], f['shard'], *[str(f[k]) for k in
                ('gate', 'up', 'down', 'blob_bytes', 'total_experts')], str(base.ORDER[index]), str(index)]
            entry, out, err = run('measure', argv, row['budgets']['measure_wall'], rss=row['budgets']['measure_rss'])
            need(not err.stat().st_size, 'component stderr')
            output = [json.loads(line) for line in out.read_text().splitlines()]
            inputs = [v for v in output if v['kind'] == 'input']; validation = [v for v in output if v['kind'] == 'validation']
            samples = [v for v in output if v['kind'] == 'sample']
            need(len(inputs) == 1 and len(validation) == 2 and all(v['cases'] == 2016 and
                v['bitwise'] and v['finite'] and v['inputs_unchanged'] for v in validation) and
                len(samples) == 40 and output[-1] == dict(kind='closed', passed=True), 'complete component evidence')
            actual = inputs[0]
            need(actual['arm'] == base.ORDER[index] and actual['process'] == index and actual['gu_type'] == 21 and
                actual['experts'] == 252 and actual['group'] == 6 and actual['blob_bytes'] == f['blob_bytes'] and
                actual['production_entry'] == 0 and actual['reference'] == 'GGML IQ3_S two-dot GU' and
                actual['candidate'] == 'iq3s_pair_gu_rows/direct', 'actual direct-component input identity')
            if row['processes']:
                first = row['processes'][0]['input']
                need(all(first[k] == actual[k] for k in ('CPU', 'experts', 'blob_bytes',
                    'weight_input_fingerprint_fnv1a64', 'activation_input_fingerprint_fnv1a64')), 'actual input mismatch')
            need(all(s['arm'] == base.ORDER[index] and s['process'] == index and s['wall_s'] >= .25 and
                s['passes'] < 1000000 and s['timer_s'] < .001 * s['fastest_warm_pass_s'] and
                s['outputs_match_untimed_reference'] and s['timed_direct_calls'] == s['expert_calls'] and
                s['timed_nt1_rows'] == s['row_ops'] and s['timed_pair_rows'] == base.ORDER[index] * s['row_ops']
                for s in samples), 'actual direct-call/duration/output gate')
            need({(s['rows'], s['pack'], s['stream'], s['rep']) for s in samples} ==
                {(n, q, t, k) for n in (640, 128) for q in (0, 1) for t in (0, 1) for k in range(5)}, 'complete eight-cell matrix')
            need(fixture_hash(f) == f['sha256'], 'post-run model slice drift')
            row['processes'].append(dict(index=index, arm=base.ORDER[index], input=actual, samples=samples,
                validation_cases=2016, validation_before_and_after=True, passed=True, receipt=base.receipt(entry)))
            if index == 5:
                decision = base.decision(row['processes'])
                decision['next'] = 'Only winning direct-component cells may motivate a new production-entry/pool comparison;no model gain or adoption follows.'
                row.update(stage='closed', decision=decision, complete=True, passed=True)
        need(base.pins(P, frozen['source_commit'], base.PROJECT) == frozen['project_inputs'] and
            base.pins(base.GGML, base.GGML_REV, ('ggml',)) == frozen['ggml_inputs'] and
            base.machine() == frozen['machine'], 'post-stage source/machine drift')
        need(time.monotonic() < deadline, 'stage deadline')
    except BaseException as error:
        row.update(stage='failed', passed=False, complete=False, error=repr(error)); raise
    finally:
        row.update(active=bool(owner.active), ownership_closed=owner.active is None and
            all(base.owned.closed(e) for e in owner.commands), commands=row['commands'] + owner.commands)
        base.save(row)
    print('IQ3_S component stage ' + stage + ' closed;commit before next stage.', flush=True)


if __name__ == '__main__':
    need(len(sys.argv) == 2, 'use prepare|build|check|0..5')
    with (base.B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); base.guards.no_model_owners([])
        if sys.argv[1] == 'prepare': prepare()
        else: execute(sys.argv[1])
