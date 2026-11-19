"""Qualify the two isolated QSA prefix operands in the retained coherent build."""

import fcntl
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import prefill_resident_input_build as bank

P, B, RESULT = bank.P, bank.B, bank.RESULT
ROOT = B / 'qsa-prefill-probability-v1'
KEY = 'qsa_prefill_probability_v1'
FIXTURES = ('sycl_qsa_prefill_pv_test', 'sycl_qsa_prefill_pv_bench')
STAGES = ('build', 'smoke', 'numeric', '0', '1', '2')
CASES = ((33, 'gather'), (2051, 'gather'), (2051, 'clustered'))
ARMS = ('ordered_exp', 'reuse_probability')
require, sha, signature = bank.require, bank.sha, bank.signature
read, write = bank.previous.read, bank.stage.write
validation, ownership = bank.basis.validation, bank.builder.ownership


def parent():
    row = read(bank.ROOT / 'record.json')
    foundation = bank.check(row)
    require(row['stage'] == 'built' and row['passed'] and row['complete']
            and row['ownership_closed'] and not row['active'] and row['host_control']['passed']
            and row['qsa_fixture_commit'] == bank.PV_SOURCE, 'verified same-build reference required')
    return row, foundation


def prepare():
    require(not ROOT.exists() and KEY not in read(RESULT), 'inspect any previous component trial')
    native, _ = parent()
    program = str(Path(__file__).resolve().relative_to(P))
    require(native['controller_sources'][program] == sha(P / program), 'controller must predate the build')
    outputs = bank.ROOT / 'build'
    require(all(not (outputs / name).exists() for name in FIXTURES), 'fresh isolated fixture build required')
    row = dict(stage='prepared', source_commit=native['source_commit'], fixture_commit=bank.PV_SOURCE,
               build_sha256=signature(bank.builder.compact(native)), reference=native['binaries']['strata'],
               generated_sha256=native['source_files'][bank.PV_KERNEL], processes=[], passed=False,
               active=False, complete=False, ownership_closed=True, stages=list(STAGES),
               boot=read(bank.basis.reference.ROOT / 'record.json')['boot'], whole_bytes=900 << 20,
               build_command=['/usr/bin/cmake', '--build', str(outputs), '--target', *FIXTURES, '-j', '2'],
               plan=dict(unit='Hoist two BF16 P fragments across four ordered-exp PV output tiles. '
                              'Keep V staging, FP32 alpha scaling, per-output MAD order and both barriers.',
                         build='Only the separate component targets. Preserve and recheck every parent binary.',
                         smoke='Traced cap33 gather on both actual exports. Parameter validation, checked '
                               'closure, positional guards and exact output bits precede full numerical coverage.',
                         numeric='Eleven cases including cap1/17/31/32/33/63/65/2051, full 262144 addresses, '
                                 'padded heads, invalid scales and scratch refusal. Match complete BF16 output '
                                 'and FP32 state against the original ordered-exp kernel.',
                         timing='Three fresh processes, three shapes with 78453 live cells and 8192 queries. '
                                'Six alternated pairs per shape, including admission, pack, compute and drain. '
                                'Reset output outside timing. Floors 0.5 ms cap33 and 10 ms cap2051. '
                                'Native steady_clock overhead median <1 us and p99 <5 us.',
                         decision='Require lower median time in all three processes per shape. Both cap2051 '
                                  'patterns must win to try that cap in a separate model plan. Retain the '
                                  'reference at cap33 unless that shape independently passes. Other caps, '
                                  'model speed, physical traffic and register spills remain unqualified.',
                         retention='Keep the two fixtures through the component decision and machine-code '
                                   'inspection, then retire them under the parent build policy.'),
               budgets=dict(build=bank.builder.LIMITS, smoke=bank.device.SMALL, gpu=bank.device.LARGE))
    ROOT.mkdir(mode=0o700)
    save(row)


def compact(row):
    result = dict(row)
    if 'actual_inputs' in result:
        actual = result.pop('actual_inputs')
        result['actual_input_groups'] = {k: dict(files=len(v), sha256=signature(v)) for k, v in actual.items()}
    return result


def save(row):
    write(ROOT / 'record.json', row)
    data = read(RESULT)
    data[KEY] = compact(row)
    write(RESULT, data)


def data_fields(line):
    return dict(field.split('=', 1) for field in line.split()[1:])


def samples(text, process):
    rows = [data_fields(line) for line in text.splitlines() if line.startswith('sample ')]
    markers = [data_fields(line) for line in text.splitlines() if line.startswith('case ')]
    require(len(rows) == len(CASES) * 12 and len(markers) == len(CASES), 'incomplete timing series')
    points = []
    for i, (cap, pattern) in enumerate(CASES):
        series = {name: [] for name in ARMS}
        common = dict(cells='78453', queries='8192', cap=str(cap), pattern=pattern)
        require(markers[i] == dict(common, samples='12', full_bits='PASS', guards='PASS',
                                   inputs='PASS', status='PASS'), 'timed case completion differs')
        for j, r in enumerate(rows[i*12:(i+1)*12]):
            pair, position = divmod(j, 2)
            variant = (process + pair + position) % 2
            arm = ARMS[variant]
            ms = float(r.pop('milliseconds'))
            require(r == dict(common, process=str(process), round=str(pair), order=str(position),
                              variant=str(variant), implementation=arm, compared=str(8192*24*256),
                              exact_bits='PASS', guards='PASS', status='PASS')
                    and (0.5 if cap == 33 else 10) <= ms < 10000, 'timed arm, order, result or duration differs')
            series[arm].append(ms)
        ratio = statistics.median(series[ARMS[1]]) / statistics.median(series[ARMS[0]])
        points.append(dict(cap=cap, pattern=pattern, samples_ms=series, median_time_ratio=ratio))
    return points


def numerical(text):
    shapes = [(97,49,33,1), (97,71,33,3), *((2113,2113,n,3) for n in (1,17,31,32,63,65,2051)),
              (262144,262144,2051,1), (262144,13,33,1)]
    complete = [data_fields(line) for line in text.splitlines() if line.startswith('case cells=')]
    comparisons = [data_fields(line) for line in text.splitlines() if line.startswith('case exact_ordered_exp_bits=')]
    require(len(complete) == len(comparisons) == len(shapes), 'incomplete numerical checks')
    for (cells, live, cap, queries), case, comparison in zip(shapes, complete, comparisons, strict=True):
        require([int(case[k]) for k in ('cells','live_cells','cap','queries')] == [cells,live,cap,queries]
                and all(case[k] == 'PASS' for k in ('guards','packed_guards','inputs','packed_logical_prefix','status'))
                and int(case['packed_guard_elements']) == 17 and int(case['packed_k_mod8']) != 0
                and comparison == dict(exact_ordered_exp_bits='PASS', all16_head_states='PASS', cells=str(cells),
                                       cap=str(cap), queries=str(queries)), 'numerical, state or guard check missing')
    require(text.count('case state_admission_rejections=9 status=PASS') == 1, 'probe refusal cases missing')
    return [list(s) for s in shapes]


def build(row, native, foundation, child):
    build_root = bank.ROOT / 'build'
    require(all(not (build_root / name).exists() for name in FIXTURES), 'fixture output already present')
    command, out, err = child.run('build', row['build_command'], native['recipe']['environment'], P,
                                   **row['budgets']['build'])
    require(ownership.closed(command) and command['exit_code'] == 0, 'fixture build failed: ' + err.read_text()[-1000:])
    bank.builder.ROOT = bank.ROOT
    actual = bank.builder.census(native, foundation, out)
    require(actual['uncovered'] == 0 and not actual['partial_archive_reuse'], 'incomplete build provenance')
    paths = (bank.PV_KERNEL, *bank.PV_FIXTURES, 'src/kernels/sycl/qsa_bf16_prefix.cpp')
    require(all(actual['compiled'].get(str(bank.ROOT / 'source' / p)) == native['source_files'][p] for p in paths),
            'reference, candidate or fixture input differs')
    for name, expected in native['binaries'].items():
        require(sha(build_root / name) == expected['sha256'], 'pre-existing executable changed')
    previous = native['actual_inputs']
    row['actual_inputs'] = {g:{p:h for p,h in actual[g].items() if p not in previous[g]}
                            for g in ('compiled','generated','links','objects','produced')}
    units = read(build_root / 'compile_commands.json')
    source_root = bank.ROOT / 'source'
    extra_objects = set(row['actual_inputs']['objects'])
    inputs = {}
    ref = 'src/kernels/sycl/qsa_bf16_prefix.cpp'
    for unit in units:
        args = shlex.split(unit['command'])
        obj = str((Path(unit['directory']) / args[args.index('-o')+1]).resolve())
        if obj in extra_objects or unit['file'] == str(source_root / ref):
            inputs[str(Path(unit['file']).relative_to(source_root))] = args
    require(len(extra_objects) == 3 and set(inputs) == {ref, bank.PV_KERNEL, *bank.PV_FIXTURES[:2]},
            'expected only the candidate and the two fixture translation units')
    # Compare effective optimization, SYCL target, floating point mode and
    # preprocessor conditions. Duplicate identical flags carry no new setting.
    def modes(args):
        return set(a for a in args if a.startswith(('-D','-O','-f','-std=','-m','--gcc-install-dir=')))
    expected = modes(inputs[ref])
    require(inputs[ref][0] == inputs[bank.PV_KERNEL][0] and expected == modes(inputs[bank.PV_KERNEL])
            and {'-O3','-fsycl','-ffp-contract=off','-fno-fast-math'} <= expected
            and all('-DSTRATA_QSA_PREFILL_PV_COMPONENT=1' in inputs[p] for p in bank.PV_FIXTURES[:2]),
            'QSA arms or fixture dispatch were compiled under different conditions')
    row['kernel_compilation'] = dict(reference=signature(inputs[ref]), candidate=signature(inputs[bank.PV_KERNEL]),
                                     effective_flags=sorted(expected))
    row['binaries'] = {f:dict(path=str(build_root / f), sha256=sha(build_root / f)) for f in FIXTURES}


def execute(action):
    row = read(ROOT / 'record.json')
    require(validation.committed(RESULT, (KEY,)) == compact(row) and row['stage'] in ('prepared', 'running')
            and not row['active'] and row['ownership_closed'] and action == STAGES[len(row['processes'])],
            'commit the previous outcome before the next stage')
    native, foundation = parent()
    require(signature(bank.builder.compact(native)) == row['build_sha256']
            and native['binaries']['strata'] == row['reference']
            and native['source_files'][bank.PV_KERNEL] == row['generated_sha256']
            and all(sha(p) == h for group in row.get('actual_inputs', {}).values() for p,h in group.items())
            and all(sha(v['path']) == v['sha256'] for v in row.get('binaries', {}).values())
            and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot']
            and ownership.artifact_bytes(ROOT) < row['whole_bytes'],
            'build inputs or exact executables changed')
    directory = ROOT / action
    directory.mkdir(mode=0o700)
    # During compilation supervise the native build outputs as well as the log.
    owner_root = bank.ROOT / 'qsa-probability-build-owners' if action == 'build' else directory / 'owned'
    child = ownership.Owner(owner_root, lambda: write(directory / 'execution.json', child.commands))
    cursor = None
    result = dict(stage=action, passed=False, ownership_closed=False)
    row.update(active=True, ownership_closed=False)
    save(row)
    try:
        if action == 'build':
            build(row, native, foundation, child)
        else:
            cursor, = (s.removeprefix('-- cursor: ') for s in subprocess.check_output(
                ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'], text=True, timeout=15).splitlines()
                if s.startswith('-- cursor: '))
            run_gpu(action, row, native, directory, child, result)
        result['host_probe_accepted'] = True
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        if cursor:
            validation.finish(result, child, cursor, row['boot'])
            result['passed'] = result['passed'] and result.get('driver_after') == native['driver']
        else:
            result['ownership_closed'] = child.active is None and all(ownership.closed(c) for c in child.commands)
            result['passed'] = bool(result.get('host_probe_accepted') and result['ownership_closed'])
        row['processes'].append(result)
        row.update(stage='running' if result['passed'] else 'failed', active=bool(child.active),
                   ownership_closed=result['ownership_closed'])
        if action == STAGES[-1] and result['passed']:
            timing = [p['points'] for p in row['processes'] if p['stage'].isdecimal()]
            ratios = [[run[i]['median_time_ratio'] for run in timing] for i in range(len(CASES))]
            eligible = [all(r < 1 for r in values) for values in ratios]
            row.update(stage='closed', complete=True, passed=True, decision=dict(per_shape=eligible,
                process_ratios=ratios, cap2051_model_plan_eligible=all(eligible[1:]),
                cap33_gather_plan_eligible=eligible[0], other_caps_timed=False, model_tested=False))
        save(row)
    require(result['passed'], result.get('error', 'validation, fault, or closure gate failed'))


def run_gpu(action, row, native, directory, child, result):
    speed = read(bank.basis.reference.ROOT / 'record.json')
    probe = speed['native_validation_probe']
    seed = speed['environment_template']
    env = {k:v.replace(seed['STRATA_MEASUREMENT_ROOT'], str(directory)) for k,v in seed.items()}
    for name in ('tmp','cache'):
        (directory / name).mkdir(mode=0o700)
    env.pop('UR_LOG_TRACING', None)
    env.pop('STRATA_TRACE', None)
    env.update(STRATA_TEST_MAIN_GATES='0', LD_PRELOAD=probe['library']['path'],
               STRATA_UR_PROBE_EXPECTED_LOADER=probe['loader']['path'], UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
               UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(directory / 'validation.log'))
    target = FIXTURES[0] if action == 'numeric' else FIXTURES[1]
    args = [row['binaries'][target]['path'], '--smoke' if action == 'smoke' else '--component']
    if action == 'smoke':
        env.update(LD_DEBUG='libs', UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                   UR_LOG_TRACING='level:info;flush:info;output:stderr')
    if action.isdecimal():
        args.append(action)
    result.update(binary=row['binaries'][target], command=args, environment_sha256=signature(env))
    cmd, out, err = child.run(action, args, env, P, **row['budgets']['smoke' if action == 'smoke' else 'gpu'])
    require(ownership.closed(cmd) and cmd['exit_code'] == 0, 'fixture failed: ' + err.read_text()[-1200:])
    text, log = out.read_text(), err.read_text()
    layer_log = directory / 'validation.log'
    proof = validation.preload_evidence(log, cmd['pid'])
    require(proof['loader'] == probe['loader']['path'] and (not layer_log.exists() or layer_log.stat().st_size == 0)
            and 'fixture_status=PASS' in text, 'incomplete fixture or validation evidence')
    bank.device.diagnostics.diagnostics(log, cmd['pid'])
    result.update(validation=proof, native_exit_code=0)
    if action != 'numeric':
        timer, = [data_fields(line) for line in text.splitlines() if line.startswith('timer ')]
        require(int(timer['calls']) == 10000 and 0 <= float(timer['median_ns']) <= int(timer['p99_ns']) < 5000
                and float(timer['median_ns']) < 1000, 'native timer overhead exceeded')
        result.update(clock_median_ns=float(timer['median_ns']), clock_p99_ns=int(timer['p99_ns']))
    if action == 'smoke':
        require('urEnqueue' in log and 'implementation=reuse_probability' in text
                and 'implementation=ordered_exp' in text and 'full_bits=PASS guards=PASS inputs=PASS' in text,
                'traced smoke did not exercise both arms')
        result['providers'] = bank.device.loaded_providers(log, dict(environment=env, native_validation_probe=probe,
            runtime_manifest=native['foundation']['runtime_manifest']))
    elif action == 'numeric':
        result['cases'] = numerical(text)
    else:
        result['points'] = samples(text, int(action))


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('prepare', *STAGES), 'unknown component stage')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare':
            prepare()
        else:
            execute(action)
