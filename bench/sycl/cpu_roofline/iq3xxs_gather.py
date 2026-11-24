"""Compare scalar and AVX2 grid loads in the decode NT1 IQ3_XXS Gate/Up pair."""
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import iq3_pair as base

KEY = 'cpu_iq3xxs_gather_dual_layout_v1'
ROOT = base.B / KEY.replace('_', '-')
base.KEY, base.ROOT = KEY, ROOT
base.PROJECT += ('AGENTS.md', 'docs/ARTIFACT_RETENTION.md')
TARGETS = ('cpu_iq_single_exact_test', 'cpu_iq3xxs_gather', 'cpu_service_pool_test')
LAYOUTS = {20: (0, 2176000), 42: (30, 1715200)}
LEAVES = '6bbd3e3e1cab1415a4bba7df675c729bf4213d59'
ROUTING = 'a52280ef7362ef9fca23861501a8c4aa23fe6816'
PROTOCOL = {
    'question': 'Does AVX2 grid gathering reduce direct paired IQ3_XXS GU time versus the existing scalar grid lookup?',
    'shapes': 'Real IQ3_XXS weights from layers 0 and 30, including both observed expert-blob layouts. '
              'The timing includes Gate/Up and the selected activation packing. Down is loaded as layout data. '
              'Reuse six and stream 252 experts at 640/128 rows to bracket cache reuse.',
    'order': 'Run both arms consecutively for each layout in a pair. Reverse the layout order in the middle pair. '
             'Use separate processes and the same source, flags, input, core and environment in both arms.',
    'timing': 'Require five distinct samples per cell, four warm passes, at least 0.25 s per sample and '
              'timer overhead below 0.1 percent of the fastest warm pass.',
    'correctness': 'Require the same-archive bitwise exact fixture and existing pool control. Each native timing '
                   'process must pass all 2016 real-model cases twice, preserve inputs and match its timed outputs.',
    'decision': 'Require all three paired process-median ratios below 1 per cell. Advance only winning cells '
                'that cover the observed production geometry to a separate decode-dispatch comparison.',
    'stop': 'Stop after the first source, compiler input, route, numerical, timing or ownership mismatch. '
            'Keep terminal outcomes with all eligible samples.',
    'limits': 'This is direct single-core Gate/Up time. Model throughput, actual DRAM traffic and hardware '
              'saturation require their own measurements.',
}
need, digest = base.need, base.digest


def model_hash(fixture):
    path = Path(fixture['shard'])
    identity = base.identity(path)
    checksum = hashlib.sha256()
    gate_bytes = 627200
    down_bytes = fixture['blob_bytes'] - 2 * gate_bytes
    extents = ((fixture['gate'], gate_bytes), (fixture['up'], gate_bytes), (fixture['down'], down_bytes))
    with path.open('rb', buffering=0) as stream:
        for expert in range(252):
            for offset, size in extents:
                start = offset + expert * size
                need(0 <= start <= identity['bytes'] - size, 'model range outside the shard')
                stream.seek(start)
                data = stream.read(size)
                need(len(data) == size, 'short model range')
                checksum.update(data)
    need(identity == base.identity(path) == fixture['file_identity'], 'model identity changed')
    return checksum.hexdigest()


def fixtures():
    lines = base.METADATA.read_text().splitlines()
    default = lines[0].split('absolute offsets in ', 1)[1].split(',', 1)[0]
    records = {}
    layers = {layer for layer, _ in LAYOUTS.values()}
    for line in lines:
        if not line or line.startswith('#'):
            continue
        fields = line.split()
        layer, gu, down_type, _, size, gate, up, down = map(int, fields[:8])
        if layer in layers:
            need(gu == 18 and LAYOUTS.get(down_type) == (layer, size) and
                 str(down_type) not in records and len(fields) in (8, 9), 'unexpected actual expert layout')
            shard = str(base.MODEL / (fields[8] if len(fields) == 9 else default))
            model = dict(layer=layer, gu_type=gu, down_type=down_type, shard=shard,
                         gate=gate, up=up, down=down, blob_bytes=size,
                         file_identity=base.identity(shard))
            model['sha256'] = model_hash(model)
            records[str(down_type)] = model
    need(set(records) == set(map(str, LAYOUTS)), 'both actual expert layouts required')
    return records


def evidence():
    path = str(base.RESULT.relative_to(base.P))
    leaves = json.loads(base.git(base.P, 'show', LEAVES + ':' + path))['dflash2_decode_cpu_leaves_v1']
    routing = json.loads(base.git(base.P, 'show', ROUTING + ':' + path))['dflash2_long_routing_train_replay_v1']
    need(leaves['stage'] == routing['stage'] == 'closed' and leaves['complete'] and leaves['passed']
         and leaves['ownership_closed'] and not leaves['active'] and routing['passed']
         and routing['routes']['cpu_jobs'] == 1449976 and routing['routes']['jobs_NT'][0] == 1154781
         and routing['plan']['binary'] == leaves['binary'], 'recover the matched long code route')
    return dict(leaves_commit=LEAVES, routing_commit=ROUTING, binary_sha256=leaves['binary']['sha256'],
                cpu_jobs=1449976, nt1_cpu_jobs=1154781)


def prepare():
    need(not ROOT.exists() and KEY not in json.loads(base.RESULT.read_text()), 'fresh CPU trial required')
    source = base.git(base.P, 'rev-parse', 'HEAD').decode().strip()
    route = evidence()
    models = fixtures()
    project = base.pins(base.P, source, base.PROJECT)
    ggml = base.pins(base.GGML, base.GGML_REV, ('ggml',))
    need(base.git(base.GGML, 'rev-parse', 'HEAD').decode().strip() == base.GGML_REV and
         digest(base.SDK) == base.SDK_SHA, 'pinned GGML or SDK changed')
    allowed = json.loads(base.SDK.read_bytes())
    need(len(allowed) == 10945 and all(digest(p) == h for p, h in allowed.items()), 'SDK input drift')
    env = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8',
               LD_LIBRARY_PATH='/opt/intel/oneapi/compiler/2026.2/lib', OMP_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', STRATA_IQ_SINGLE_EXACT='0',
               STRATA_IQ3_PAIR='0', STRATA_IQ3_PAIR_AUDIT='0', STRATA_CPU_SERVICE='0',
               TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'))
    host = base.host_inventory(env)
    tools = {str(Path(p).resolve()): digest(p) for p in
             (base.CC, base.CXX, '/usr/bin/cmake', '/usr/bin/ninja', '/usr/bin/git', sys.executable)}
    allowed.update(host['files'])
    allowed.update(tools)
    for directory in Path('/usr/share').glob('cmake-*/Modules'):
        allowed.update({str(p.resolve()): digest(p) for p in directory.rglob('*') if p.is_file()})
    build = ROOT / 'build'
    configure = ['/usr/bin/cmake', '-S', str(base.P / 'bench/sycl/cpu_roofline'), '-B', str(build),
                 '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                 '-DCMAKE_C_COMPILER=' + base.CC, '-DCMAKE_CXX_COMPILER=' + base.CXX,
                 '-DCMAKE_EXE_LINKER_FLAGS=-Wl,-t', '-DGGML_CCACHE=OFF', '-DSTRATA_GGML_DIR=' + str(base.GGML)]
    short = base.git(base.GGML, 'rev-parse', '--short', 'HEAD').decode().strip()
    header = f'#pragma once\n\n#define GGML_VERSION "0.24.0"\n#define GGML_COMMIT  "{short}"\n'
    schedule = []
    for pair in range(3):
        layouts = (42, 20) if pair == 1 else (20, 42)
        for layout in layouts:
            f = models[str(layout)]
            for position in range(2 * pair, 2 * pair + 2):
                arm = base.ORDER[position]
                args = [str(build / 'cpu_iq3xxs_gather'), f['shard'],
                        *(str(f[field]) for field in ('gate', 'up', 'down', 'blob_bytes')),
                        '512', str(arm), str(position), str(layout)]
                schedule.append(dict(layout=layout, position=position, arm=arm, argv=args))
    ROOT.mkdir(mode=0o700)
    for child in ('tmp', 'cache'):
        (ROOT / child).mkdir(mode=0o700)
    inputs = ROOT / 'external-inputs.json'
    base.write(inputs, allowed)
    base.save(dict(stage='prepared', source_commit=source, project_inputs=project, ggml_inputs=ggml,
        route_evidence=route, protocol=PROTOCOL, fixtures=models, metadata_sha256=digest(base.METADATA),
        host_link_inputs=host, tools=tools, machine=base.machine(), environment=env,
        external_inputs=dict(path=str(inputs), sha256=digest(inputs), files=len(allowed)),
        generated_expected={str(build / 'ggml/src/ggml-version.h'): hashlib.sha256(header.encode()).hexdigest()},
        configure=configure, compile=['/usr/bin/cmake', '--build', str(build), '--target', *TARGETS, '--parallel', '4'],
        check_plan=[['exact', str(build / 'cpu_iq_single_exact_test'), '--iq3xxs-gather'],
                    ['pool', str(build / 'cpu_service_pool_test'), '0']],
        order=schedule, processes=[], active=False, ownership_closed=True, passed=False, complete=False,
        budgets=dict(build_wall=300, case_wall=240, build_cpu=1200, check_cpu=60, bench_cpu=180,
                     text_cap=1 << 20, file_cap=16 << 20, total_cap=384 << 20),
        retention='Temporary CPU-only build. Keep its three executables and archive through the named comparison. '
                  'After selecting a terminal outcome, recover the committed source and recipe and retire its build.'))


def current(row):
    saved = json.loads(base.git(base.P, 'show', 'HEAD:' + str(base.RESULT.relative_to(base.P))))[KEY]
    need(saved == base.compact(row) and row['protocol'] == PROTOCOL, 'commit the frozen trial state')
    need(base.pins(base.P, row['source_commit'], base.PROJECT) == row['project_inputs'] and
         base.pins(base.GGML, base.GGML_REV, ('ggml',)) == row['ggml_inputs'] and
         base.machine() == row['machine'] and digest(base.METADATA) == row['metadata_sha256'],
         'source, machine or model metadata changed')
    need(all(digest(p) == h for p, h in row['tools'].items()), 'planned tool changed')
    for artifact in row.get('artifacts', {}).values():
        path = Path(artifact['path'])
        need(path.stat().st_size == artifact['bytes'] and digest(path) == artifact['sha256'],
             'actual executable or archive changed')
    base.input_map(row)


def sample_result(path, case, receipt, previous):
    data = [json.loads(line) for line in path.read_text().splitlines()]
    input_row, = (v for v in data if v['kind'] == 'input')
    samples = [v for v in data if v['kind'] == 'sample']
    checks = [v for v in data if v['kind'] == 'validation']
    arm, position = case['arm'], case['position']
    need(data[-1] == dict(kind='closed', passed=True) and len(checks) == 2 and
         all(v['cases'] == 2016 and v['bitwise'] and v['finite'] and v['inputs_unchanged'] for v in checks),
         'require checked route with unchanged model inputs and output bits')
    need(input_row['gu_type'] == 18 and input_row['down_type'] == case['layout'] and
         input_row['arm'] == arm and input_row['process'] == position and not input_row['production_entry']
         and input_row['experts'] == 252 and input_row['group'] == 6 and
         input_row['blob_bytes'] == LAYOUTS[case['layout']][1], 'selected geometry or direct arm differs')
    keys = ('CPU', 'experts', 'blob_bytes', 'weight_input_fingerprint_fnv1a64',
            'activation_input_fingerprint_fnv1a64')
    need(all(old['input']['CPU'] == input_row['CPU'] for old in previous), 'pinned core drift')
    need(all(old['layout'] != case['layout'] or all(old['input'][key] == input_row[key] for key in keys)
             for old in previous), 'same-layout input fingerprint changed')
    matrix = {(rows, pack, stream, rep) for rows in (640, 128) for pack in (0, 1)
              for stream in (0, 1) for rep in range(5)}
    need(len(samples) == 40 and {(v['rows'], v['pack'], v['stream'], v['rep']) for v in samples} == matrix,
         'require all distinct timing cells')
    need(all(v['arm'] == arm and v['process'] == position and .25 <= v['wall_s'] < 180
             and 0 <= v['cpu_s'] < 180 and 0 < v['passes'] < 1000000
             and 0 <= v['timer_s'] < .001 * v['fastest_warm_pass_s']
             and v['expert_calls'] == v['passes'] * (252 if v['stream'] else 6)
             and v['row_ops'] == v['expert_calls'] * v['rows']
             and v['outputs_match_untimed_reference'] and v['timed_direct_calls'] == v['expert_calls']
             and v['timed_nt1_rows'] == v['row_ops'] and v['timed_pair_rows'] == arm * v['row_ops']
             for v in samples), 'timing floor, overhead, output or actual arm mismatch')
    fields = ('rows', 'pack', 'stream', 'rep', 'wall_s', 'cpu_s', 'passes', 'expert_calls',
              'timer_s', 'fastest_warm_pass_s')
    kept = [{key: v[key] for key in fields} for v in samples]
    return dict(layout=case['layout'], position=position, arm=arm, input=input_row, samples=kept,
                direct_counts_verified=True, validation_cases=2016, passed=True, receipt=base.receipt(receipt))


def execute(action):
    row = json.loads((ROOT / 'record.json').read_bytes())
    current(row)
    need(not row['active'] and row['ownership_closed'], 'close the owned predecessor')
    i = int(action) if action.isdigit() else None
    need((action == 'build' and row['stage'] == 'prepared') or
         (action == 'check' and row['stage'] == 'built') or
         (i == len(row['processes']) and row['stage'] == 'checked' and 0 <= i < len(row['order'])),
         'ordered immutable stage required')
    stage = ROOT / ('owned-' + action)
    need(not stage.exists(), 'inspect the original attempt before any new stage')

    def progress():
        base.write(ROOT / 'execution.json', dict(stage=action, active=bool(owner.active), commands=owner.commands))

    owner = base.owned.Owner(stage, persist=progress)
    budget = row['budgets']
    deadline = time.monotonic() + budget['build_wall' if action == 'build' else 'case_wall']

    def run(label, argv, *, cpu, rss, wall):
        remaining = deadline - time.monotonic()
        need(remaining > 0, 'whole stage deadline')
        entry, out, err = owner.run(label, argv, row['environment'], ROOT, wall=min(wall, remaining),
            cpu=cpu, rss_cap=rss, text_cap=budget['text_cap'], file_cap=budget['file_cap'],
            total_cap=budget['total_cap'])
        need(base.owned.closed(entry) and entry['exit_code'] == 0,
             label + ' failed: ' + err.read_text(errors='replace')[-1400:])
        return entry, out, err

    try:
        if action == 'build':
            run('configure', row['configure'], cpu=120, rss=2 << 30, wall=60)
            run('compile', row['compile'], cpu=budget['build_cpu'], rss=8 << 30, wall=240)
            run('deps', ['/usr/bin/ninja', '-C', str(ROOT / 'build'), '-t', 'deps'],
                cpu=20, rss=512 << 20, wall=20)
            row['build_evidence'] = base.build_check(row, stage, iq3xxs_gather=True)
            row['artifacts'] = {}
            for name in (*TARGETS, 'libexpert_cpu.a'):
                path = ROOT / 'build' / name
                row['artifacts'][name] = dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))
            row['stage'] = 'built'
        elif action == 'check':
            expected = {
                'exact': 'IQ3_XXS gather exact PASS: 80 cases\n'
                         'IQ single exact PASS: 384 single and 80 paired cases, bitwise GGML parity and row canaries',
                'pool': 'cpu_service_pool PASS observer=0 formats=7 nt=1..4 cases=28 '
                        'checksum_fnv1a64_le=e5d7a50a8e7d5ac5',
            }
            checks = []
            for label, *argv in row['check_plan']:
                entry, out, err = run(label, argv, cpu=budget['check_cpu'], rss=1 << 30, wall=70)
                need(not err.stat().st_size and out.read_text().strip() == expected[label],
                     'the same-build arithmetic or pool control differs')
                checks.append(dict(label=label, receipt=base.receipt(entry), passed=True))
            row.update(stage='checked', checks=checks)
        else:
            case = row['order'][i]
            fixture = row['fixtures'][str(case['layout'])]
            need(model_hash(fixture) == fixture['sha256'], 'initial model input drift')
            entry, out, err = run('measure', case['argv'], cpu=budget['bench_cpu'], rss=1 << 30, wall=180)
            need(not err.stat().st_size, 'unexpected component stderr')
            sample = sample_result(out, case, entry, row['processes'])
            need(model_hash(fixture) == fixture['sha256'], 'post-run model input drift')
            row['processes'].append(sample)
            if i + 1 == len(row['order']):
                decisions = {
                    str(layout): base.decision([p for p in row['processes'] if p['layout'] == layout])['cells']
                    for layout in LAYOUTS
                }
                row.update(stage='closed', complete=True, passed=True, cells=decisions, model_trial_qualified=False)
        need(base.pins(base.P, row['source_commit'], base.PROJECT) == row['project_inputs'] and
             base.machine() == row['machine'] and time.monotonic() < deadline, 'post-stage input or deadline')
    except BaseException as error:
        row.update(stage='failed', error=repr(error), complete=False, passed=False)
        raise
    finally:
        row.update(active=bool(owner.active),
                   ownership_closed=owner.active is None and all(base.owned.closed(e) for e in owner.commands))
        base.save(row)


if __name__ == '__main__':
    need(len(sys.argv) == 2, 'use prepare, build, check or one original case index')
    action, = sys.argv[1:]
    with (base.B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        if action == 'prepare':
            prepare()
        else:
            execute(action)
