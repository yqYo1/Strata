#!/usr/bin/env python3
"""Bounded selected6e40 fused-family service; no critical-path or speed claim.

Uses the existing source-emitted census and official bench.sh, without rebuilding
the engine. Independently authored under the repository LICENSE.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import os
import secrets
import select
import socket
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / 'tools/sycl'))
import benchmark_runtime as python_runtime
import cpu_capacity as models
import engine_once
import generation_engine as gen
import measurement_owner as owner_module
import build_indexed_collector as indexed
import profile_aggregate
import runtime_probe as runtime
import supervision_model as shared
from measure_down_scatter import postchecks

B, A = gen.B, gen.A
ROOT = B / 'selected-fused-moe-service-v1'
RESULT = A / 'fused-moe-service-current.json'
REFERENCE = A / 'prefill-runtime-comparison-current.json'
TEMPLATE = A / 'phase-prefill-speed-current.json'
SOURCE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
CASES = {'smoke-profile': (32768, 39260, ((0, 39259),))}
for case in ('long-base0', 'long-profile0', 'long-profile1', 'long-base1', 'long-base2', 'long-profile2'):
    CASES[case] = (98304, 117489, ((0, 81920), (81920, 35568)))
require, digest = runtime.require, runtime.digest


def recover_inputs(record):
    """Recover original measurement inputs, without relabeling old native builds."""
    pins = record['input_sha256']
    for name, raw in gen.blobs(P, record['source_commit'], list(pins)):
        require(hashlib.sha256(raw).hexdigest() == pins[name], 'historical input unrecoverable: ' + name)


def collector_assets():
    """Reuse the exact qualified collector; later Python helpers are not its source.

    Native generators, generated files, actual dependencies, recipe and ELFs
    still match. Recover old host-only controllers from Git instead of requiring
    unrelated later Python files to equal their historical versions.
    """
    container = gen.committed(indexed.RESULT)
    record = container[indexed.KEY]
    host = container['unitrace_aggregation_host_v2']
    graph = container['unitrace_indexed_graph']
    for row in (record, host, graph):
        require(row['passed'] and row['complete'] and row['ownership_closed'] and not row['active'],
            'closed native/host/graph qualifications required')
        recover_inputs(row)
    require(host['performance_gate_passed'] and indexed.canonical(host) == record['host_qualification_sha256'] and
        graph['service_coverage_qualified'] and graph['fresh_faults'] == [], 'historical qualification differs')
    for name in ('tools/sycl/build_indexed_collector.py', 'tools/sycl/collector_aggregation.py'):
        require(gen.git('show', record['source_commit'] + ':' + name) == (P/name).read_bytes(),
            'native collector generator changed: ' + name)
    require(indexed.command(container) == record['command'], 'collector recipe changed')
    for name, raw in indexed.generated().items():
        require((indexed.ROOT/name).read_bytes() == raw and digest(indexed.ROOT/name) == record['generated_sha256'][name],
            'generated collector input changed')
    launcher = container['unitrace_session_owner']
    require(launcher['passed'] and launcher['ownership_closed'] and not launcher['active'], 'launcher build not closed')
    recover_inputs(launcher)
    require(gen.git('show', launcher['source_commit'] + ':tools/sycl/unitrace_launch.py') ==
        (P/'tools/sycl/unitrace_launch.py').read_bytes(), 'launcher generator changed')
    require(hashlib.sha256(indexed.launch.generate()).hexdigest() == launcher['generated_source_sha256'],
        'launcher generated input changed')
    # Native collector preprocessing closure plus the original launcher/link closure.
    for manifest in (record['compiler_inputs'], container['unitrace_build']['dependencies']):
        require(digest(manifest['path']) == manifest['sha256'], 'collector dependency manifest changed')
        rows = json.loads(Path(manifest['path']).read_text())
        require(len(rows) == manifest['files'], 'collector dependency count changed')
        for row in rows: require(digest(row['path']) == row['sha256'], 'collector compiler/input changed: ' + row['path'])
    assets = dict(launcher=record['artifacts']['unitrace'], collector=record['artifacts']['libunitrace_tool.so'])
    require(all(assets[k] == graph['assets'][k] for k in assets) and
        assets['launcher']['sha256'] == launcher['binary']['sha256'], 'qualified collector/launcher differ')
    for item in assets.values(): require(digest(item['path']) == item['sha256'], 'collector/launcher ELF changed')
    return dict(assets=assets, native_source=record['source_commit'], dependencies=record['compiler_inputs'],
        original_launcher_source=launcher['source_commit'], host_source=host['source_commit'], graph_source=graph['source_commit'],
        reuse='Historical native inputs recovered; unchanged generators/generated dependencies/recipe/ELFs. No rebuild or new GPU qualification.')


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def continue_benchmark(io):
    """Match the actual strata_bench.profile_request_gate line protocol."""
    io.send(b'continue\n')


def finish_benchmark(io):
    """Drain the official CLI's human summary after its final request gate."""
    io.proc.stdin.close()
    deadline = time.monotonic() + 90
    while not io.exit_watch.poll(0) or not io.eof:
        io.poll()
        require(time.monotonic() < deadline, 'benchmark teardown deadline')
        if select.select([io.proc.stdout], [], [], .05)[0]:
            io.read()  # Interactive retains its existing finite pending/output caps.
    require(b'STRATA_PROFILE_GATE' not in io.pending, 'request gate after final boundary')
    io.pending.clear()
    io.exit(1)


def python_contract():
    public = runtime.validation.committed(REFERENCE,
        ('historical_regression_audit', python_runtime.SCOPE))
    original = json.loads((python_runtime.HOST / 'record.json').read_text())
    require(public == python_runtime.compact(original) and public['passed'] and
        public['complete'] and public['ownership_closed'] and not public['active'] and
        all(owner_module.closed(c) for c in original['commands']), 'closed Python qualification required')
    # Reuse provider qualification only for unchanged child scripts. The new
    # Main-only reader is not imported by bench/server/engine_once children.
    # Standalone tools/test_*.py are not imported by the benchmark/server. A
    # new unrelated unit test must not silently enlarge the qualified runtime.
    def child_source(n):
        return n in ('bench.sh', 'tools/sycl/engine_once.py') or n.startswith('serve/') or (
            n.startswith('tools/') and n.count('/') == 1 and not n.startswith('tools/test_'))
    names = [n for n in original['benchmark_inputs'] if child_source(n)]
    current_names = [n for n in gen.benchmark_pins(gen.git('rev-parse', 'HEAD').decode().strip()) if child_source(n)]
    require(set(names) == set(current_names), 'child script inventory changed')
    for name, raw in gen.blobs(P, public['source_commit'], names):
        require(hashlib.sha256(raw).hexdigest() == original['benchmark_inputs'][name] == digest(P / name),
            'qualified child source changed: ' + name)
    manifest = python_runtime.read_manifest(public['plan']['manifest'])
    python_runtime.validate_manifest(manifest)
    return dict(qualification_source=public['source_commit'], manifest=public['plan']['manifest'],
        unchanged_child_sources={n:original['benchmark_inputs'][n] for n in names}), manifest


def native():
    row = runtime.validation.committed(runtime.RESULT)
    full = runtime.current(row)
    require(full.get('binary_origin_source_commit', full['source_commit']) == SOURCE,
        'selected original native source required')
    return full, row


def request_hash(minimum):
    sys.path.insert(0, str(P / 'tools'))
    import strata_bench as client
    suite = client.Suite('', '', 131072, 1, True, log=lambda *a: None,
        prefill_only=True, prefill_tokens=minimum, workload_id='native_gdn_20261018')
    return hashlib.sha256(json.dumps(client.chat_body(suite.prefill_prompt(0), 24, True)).encode()).hexdigest()


def prepared_environment(root, neo):
    template = gen.committed(TEMPLATE)['environment_template']
    old = template['STRATA_MEASUREMENT_ROOT']
    env = {k:v.replace(old, str(root)) for k,v in template.items()}
    require(not env.get('LD_PRELOAD') and env['CUDA_MODULE_LOADING'] == 'EAGER', 'private preload required')
    env.update(STRATA_TRACE='1', STRATA_PREFILL_LAYER_MAJOR_PROFILE='1',
        STRATA_SYCL_PREFILL_COMMAND_PROFILE='0', STRATA_PREFILL_CPU_WAIT_PROFILE='0',
        STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE='0', STRATA_PREFILL_LAYER_MAJOR_GDN_PROFILE='0',
        UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION', ZE_ENABLE_ALT_DRIVERS=neo,
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(root / 'validation.log'),
        PYTHONNOUSERSITE='1', PYTHONPYCACHEPREFIX=str(root / 'cache/python'))
    require(env['EnableDirectSubmission'] == '0' and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and
        env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1', 'qualified safety settings required')
    python_runtime.require_environment(env)
    return env


def admission(row, full_models=False):
    full, rt = native()
    require(full['binary'] == row['binary'] and digest(runtime.RESULT) == row['runtime_sha256'] and
        digest(TEMPLATE) == row['template_sha256'] and rt['boot'] == row['boot'], 'native/runtime/template changed')
    require(gen.benchmark_pins(row['source_commit']) == row['controller_inputs'], 'measurement source changed')
    require(collector_assets() == row['collector_reuse'], 'collector qualification/assets changed')
    contract, manifest = python_contract()
    require(contract == row['python_contract'], 'benchmark runtime qualification changed')
    shared.profile_pin(row['configuration'], row['source_commit'], row['profile'])
    models.model_pins(row['configuration'], row['model_assets'], full=full_models)
    for mode, plan in row['plans'].items():
        root = Path(plan['root'])
        require(plan['environment'] == prepared_environment(root, row['NEO']) and
            python_runtime.selection(plan['environment']) == manifest['selected'] and
            request_hash(CASES[mode][0]) == plan['request_sha256'], 'workload/environment changed')
        for name, sha in plan['inputs'].items(): require(digest(root / name) == sha, 'generated input changed')
    return rt, manifest


def prepare():
    require(not ROOT.exists() and not RESULT.exists(), 'fresh diagnostic only; inspect ownership before retry')
    reusable = collector_assets()
    full, rt = native()
    reference = runtime.validation.committed(REFERENCE, ('historical_regression_audit', 'supervision_model_control_v6'))
    require(reference['complete'] and reference['passed'] and reference['ownership_closed'] and
        reference['binary'] == full['binary'], 'closed same-native reference required')
    cfg = copy.deepcopy(reference['configuration'])
    source = gen.git('rev-parse', 'HEAD').decode().strip()
    contract, manifest = python_contract()
    old_assets = json.loads(gen.git('show', '1e676d554:' + str(TEMPLATE.relative_to(P))))['model_assets']
    model_assets = models.model_pins(cfg, old_assets, full=True)
    loader = rt['preload']['loader']
    probe = dict(library=rt['preload']['library'], loader=dict(path=loader, sha256=rt['runtime_files'][loader]))
    ROOT.mkdir(mode=0o700)
    row = dict(stage='prepared', source_commit=source, native_source=SOURCE, binary=full['binary'],
        runtime_sha256=digest(runtime.RESULT), template_sha256=digest(TEMPLATE), boot=rt['boot'],
        NEO=next(p for p in rt['runtime_files'] if 'libze_intel_gpu' in p), probe=probe,
        configuration=cfg, profile=shared.profile_pin(cfg, source), python_contract=contract,
        controller_inputs=gen.benchmark_pins(source), model_assets=model_assets, plans={}, checks=[],
        active=False, ownership_closed=True, complete=False, passed=False, model_adopted=False,
        collector_reuse=reusable,
        host_tests=dict(source='aa61e8741', command='isolated Python unittest: tests/core/test_profile_aggregate.py + tools/sycl/test_generation_engine.py', passed=28),
        question='Measure unique GU/SwiGLU and Down/scatter family summed GPU service by actual M geometry on selected6e40.',
        criteria=dict(sequence=list(CASES), overhead='Three fixed independent long baseline/profile pairs: base0/profile0,base1/profile1,base2/profile2; balanced launch order. Median profiled/baseline prompt wall ratio<=1.05. Every diagnostic must first pass work,clock,source,ownership,provider,drain and fault gates. No retries or replacements.',
            scope='Whole-process exact-family device service. Long-phase ownership remains conditional; source request counts and histograms must reconcile separately for both families; other phase work is not inferred absent from count equality. Not clean speed or adoption.',
            output='Aggregate --device-timing --verbose only; no timeline. Sum every geometry, require table conservation and full footer. No new UR subscriber.',
            reference='Baseline keeps the same validation,trace and coarse source census as profile; the ratio measures added PTI overhead only,not combined observer overhead or clean production timing.'),
        limits='No complete model coverage,critical path,physical bus traffic,hardware saturation,clean speed,decode or physical262144 qualification.',
        retention='Keep individual prompt-wall samples,per-family calls/rows/service/projection volume and <=46 M-geometry bins for the imminent tile/epilogue choice. No full table,census histogram or repeated source/environment copies after closure. Retire raw logs/private caches after replay;no new engine build.')
    for mode, (minimum, actual, pairs) in CASES.items():
        root = ROOT / mode; root.mkdir()
        for folder in ('tmp', 'cache', 'trace'): (root / folder).mkdir()
        config = copy.deepcopy(cfg); config['log'] = str(root / 'engine.stderr')
        write(root / 'config.json', config)
        native_plan = dict(binary=row['binary'], native_validation_probe=probe)
        if 'profile' in mode:
            native_plan['native_profile'] = dict(mode='aggregate', **reusable['assets'],
                session='StrataFused' + secrets.token_hex(12), environment=dict(UNITRACE_SystemTime='1',
                    UNITRACE_TraceOutputDir=str(root/'trace'),UNITRACE_FollowChildProcess='1',LD_DEBUG='libs'))
        write(root / 'plan.json', native_plan)
        with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        row['plans'][mode] = dict(root=str(root), config=config, native_profile=native_plan.get('native_profile'), expected_tokens=actual, expected_window_pairs=pairs,
            request_sha256=request_hash(minimum), environment=prepared_environment(root, row['NEO']),
            inputs={n:digest(root / n) for n in ('config.json', 'plan.json')},
            command=['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(port),
                '--prefill-only', '--prefill-tokens', str(minimum), '--think', '--runs', '1', '--profile-request-gates',
                '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root / 'bench')],
            budgets=dict(wall=1200, cpu=7200, rss_cap=96 << 30, text_cap=16 << 20, file_cap=32 << 20, total_cap=64 << 20))
    require(row['plans']['long-base0']['request_sha256'] == reference['request_hashes'][0], 'official long request changed')
    admission(row)
    write(ROOT / 'record.json', row); write(RESULT, row)
    print('Prepared fused-family service and fixed overhead comparison;commit exact plans before smoke.', flush=True)


def completed_request(row, plan, entry):
    """Validate closed official output; reusable for host-only schema replay.

    This reader alone never marks a run passed or substitutes for launch/admission,
    owner supervision and postchecks.
    """
    require(owner_module.closed(entry) and entry['exit_code'] == 0, 'closed benchmark required')
    root = Path(plan['root'])
    end = gen.native_end(root, dict(binary=row['binary'], native_validation_probe=row['probe'], native_profile=plan['native_profile']), dict(config=plan['config']), row['boot'], entry['pid'])
    path, = (root/'bench').glob('*/results.json'); data = json.loads(path.read_text())
    cohort = data['results']['prefill_only']; require(data['complete'] and not data['results']['failures'] and cohort['complete'] and len(cohort['requests']) == 1, 'one complete request required')
    sample = cohort['requests'][0]
    require(sample['eligible'] is True and sample['ok'] is True and sample['stream_complete'] is True and
        sample['thinking_requested'] is True and sample['eligibility_errors'] == [] and
        sample['request_sha256'] == plan['request_sha256'] and
        sample['actual_prompt_tokens'] == sample['actual_read_tokens'] == plan['expected_tokens'] and
        sample['actual_reused_tokens'] == 0 and 1 <= sample['generated_tokens'] <= 24, 'request work differs')
    sys.path.insert(0, str(P/'tools')); from strata_bench import verify_request_journal
    journal = verify_request_journal(path.parent)
    require(journal['valid'] and journal['complete'], 'request journal invalid')
    events = [json.loads(line) for line in (path.parent/'requests.jsonl').read_text().splitlines()]
    require([dict(v['sample'], run=v['run']) for v in events if v['kind'] == 'result' and
        v['phase'] == 'prefill-only'] == cohort['requests'], 'cohort differs from durable request journal')
    text = (root/'engine.stderr').read_text()
    validation = runtime.validation.preload_evidence(text, end['native']['pid'])
    require(validation['loader'] == row['probe']['loader']['path'], 'validation provider differs')
    require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain|retirement failed', text), 'native runtime error')
    footers = list(re.finditer(r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*', text, re.M))
    require(len(footers) == 2 and int(footers[0][1]) == 74 and int(footers[0][2]) == 0 and
        tuple(map(int,footers[1].groups())) == (plan['expected_tokens'],0,plan['expected_tokens'],sample['generated_tokens']), 'native request interval differs')
    part = text[footers[0].end():footers[1].start()]
    maximum_rows = max(rows for _, rows in plan['expected_window_pairs'])
    require(part.count('strata prefill: layer-major candidate, tokens='+str(plan['expected_tokens']-1)+' chunk=8192 residual_dtype=bf16') == 1 and
        part.count('strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows='+str(maximum_rows)) == 1, 'selected prefill route differs')
    work = gen.common.extract_planned_moe_work(part, plan['expected_window_pairs'])
    census = gen.common.compact_window_census(work['planned_moe_windows'])
    histogram = census['aggregate_job_rows_histogram']
    expected_calls = sum(n for m, n in histogram if m >= 2334)
    expected_rows = sum(m*n for m, n in histogram if m >= 2334)
    family_counts = {}
    for family, prefix in (('gu','strata MoE GU fusion:'), ('down','strata MoE Down fusion:')):
        counters = gen.common.parse_counter_lines(part, prefix, gen.common.FUSION_FIELDS)
        require(len(counters) == 1, 'one request-local successful family footer required')
        values = {k:int(v) for k,v in counters[0].items()}
        require(values == dict(fused_jobs=expected_calls, fused_rows=expected_rows,
            separate_jobs=census['jobs']-expected_calls, min_rows=2334), 'successful family and source histogram differ')
        require(len(gen.common.parse_counter_lines(text, prefix, gen.common.FUSION_FIELDS)) == 1,
            'family footer outside long request; phase attribution remains conditional')
        family_counts[family] = values
    service = None
    if plan['native_profile']:
        profiling = plan['native_profile']
        positions = []
        for phase, asset in [('init',row['probe']['library']),('init',profiling['collector']),('fini',profiling['collector'])]:
            matches = [m.start() for m in re.finditer(r'calling '+phase+r':\s+([^\s]+)',text)
                if Path(m[1]).resolve() == Path(asset['path']).resolve()]
            require(len(matches) == 1, 'missing/ambiguous validation/collector loader phase')
            positions.extend(matches)
        require(positions[0] < text.index('phase=before-main') < positions[1] < positions[2], 'collector/validation lifetime differs')
        lookup = [line for line in text.splitlines() if 'symbol lookup error' in line]
        require(all(re.search(r'undefined symbol: (zelLoaderTranslateHandle|zelSetDriverTeardown) \(fatal\)$',line)
            for line in lookup), 'new loader lookup failure')
        require(not list((root/'trace').rglob('*')), 'aggregate unexpectedly emitted timeline/output')
        require(not re.search(r'(?i)(lost (?:event|record)|buffer overflow|dropped (?:event|record)|collection failed)',text),
            'collector loss/error marker')
        table = profile_aggregate.device_summary(text)
        service = profile_aggregate.fused_moe_service(table, histogram)
        service['whole_process_total_device_ns'] = table['total_device_ns']
        service['whole_process_table_rows'] = len(table['rows'])
    return dict(request={k:sample[k] for k in ('request_sha256','actual_read_tokens','actual_reused_tokens','generated_tokens','prompt_ms')},
        successful_fusion_counts=family_counts, service=service,
        native_identity=end['native'], validation=validation,
        evidence_sha256={name:digest(root/name) for name in ('engine.stderr','engine-start.json','engine-exit.json')})


def run(mode):
    row = gen.committed(RESULT); plan = row['plans'][mode]; root = Path(plan['root'])
    require(row == json.loads((ROOT / 'record.json').read_text()), 'external diagnostic record differs')
    position = list(CASES).index(mode)
    require(len(row['checks']) == position and all(c['passed'] and c['ownership_closed'] for c in row['checks']) and
        not (root / 'execution.json').exists(), 'ordered single-use diagnostic;stop on failure')
    rt, manifest = admission(row)
    cursor = next(x.split(': ', 1)[1] for x in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor'], text=True).splitlines() if x.startswith('-- cursor: '))
    result = dict(mode=mode, passed=False, complete=False, ownership_closed=False, native_launched=None)
    owner = None
    def persist():
        write(root / 'execution.json', dict(result, active=bool(owner and owner.active), commands=owner.commands if owner else []))
    owner = owner_module.Owner(root / 'owned', persist)
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
    def gate(io, leader):
        for stage in ('begin', 'end'):
            while True:
                line = io.line(900)
                if line.startswith('STRATA_PROFILE_GATE '): break
            require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                dict(stage=stage, phase='prefill-only', run=0), 'request gate differs')
            delta = [a-b for a,b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()), initial)]
            require(min(delta) >= 0 and abs(delta[0]-delta[1]) <= 50000000, 'suspend/clock discontinuity')
            claim = json.loads((root / 'engine-start.json').read_text()); pid = claim['native']['pid']
            result['native_launched'] = True
            current = owner_module.identity(pid)
            require(current and current['start_ticks'] == claim['native']['start_ticks'] and
                current['sid'] == leader['pid'] and str(Path('/proc', str(pid), 'exe').resolve()) == str(Path(row['binary']['path']).resolve()), 'native ownership differs')
            expected = engine_once.native_environment(dict(native_validation_probe=row['probe']), plan['environment'])
            profiling = plan['native_profile']
            if profiling:
                expected.update(profiling['environment'])
                expected['STRATA_UNITRACE_TARGET_PRELOAD'] = row['probe']['library']['path']
                expected['LD_PRELOAD'] = profiling['collector']['path'] + ':' + row['probe']['library']['path']
            actual = dict(x.split(b'=', 1) for x in Path('/proc', str(pid), 'environ').read_bytes().split(b'\0') if b'=' in x)
            require(all(actual.get(k.encode()) == v.encode() for k,v in expected.items()) and
                all(k in {b'PWD',b'OLDPWD',b'SHLVL',b'_'} or (profiling and k.startswith(b'UNITRACE_'))
                    for k in set(actual)-{k.encode() for k in expected}), 'native environment differs')
            maps = runtime.executable_mappings(Path('/proc', str(pid), 'maps').read_text())
            require(maps <= set(rt['runtime_files']) | {row['binary']['path']} |
                ({profiling['collector']['path']} if profiling else set()) and
                {row['NEO'],row['probe']['loader']['path'],row['probe']['library']['path']} <= maps, 'native provider differs')
            if profiling: require(profiling['collector']['path'] in maps, 'declared collector is not mapped')
            allowed = {p['realpath'] for p in manifest['inventory']['files'].values()}
            wrapper = owner_module.identity(claim['wrapper']['pid']); require(wrapper and wrapper['sid'] == leader['pid'], 'wrapper ownership differs')
            server = owner_module.identity(wrapper['ppid']); require(server and server['sid'] == leader['pid'], 'server ownership differs')
            port = plan['command'][plan['command'].index('--port')+1]
            python_runtime.check_running(leader['pid'], P/'tools/strata_bench.py', manifest['selected'], plan['environment'], allowed, plan['command'][2:])
            python_runtime.check_running(server['pid'], P/'serve/server.py', manifest['selected'], plan['environment'], allowed,
                ['--engine','strata','--config',str(root/'config.json'),'--port',port])
            require(not (root/'validation.log').exists() or (root/'validation.log').stat().st_size == 0, 'parameter warning/error')
            continue_benchmark(io)
        finish_benchmark(io)
    try:
        entry, _, _ = owner.run('bench', plan['command'], plan['environment'], P, interaction=gate, **plan['budgets'])
        require(owner_module.closed(entry) and entry['exit_code'] == 0, 'benchmark owner/protocol failed')
        result.update(completed_request(row, plan, entry))
        admission(row, full_models=(mode == list(CASES)[-1]))
        result.update(passed=True, complete=True)
    except BaseException as error:
        result['error'] = repr(error)
        if (root/'engine.stderr').exists(): result['error_excerpt'] = (root/'engine.stderr').read_text()[-4096:]
    finally:
        result['ownership_closed'] = owner.active is None and all(c.get('direct_child_reaped') is True and
            c.get('session_empty') is True and c.get('survivors') == [] and
            c.get('ownership_status') == 'observed' and type(c.get('exit_code')) is int for c in owner.commands)
        result['protocol_qualified'] = bool(owner.commands) and result['passed'] and all(owner_module.closed(c) for c in owner.commands)
        result['passed'] = result['passed'] and result['protocol_qualified']
        profiling = plan['native_profile']
        if profiling and result['ownership_closed']:
            session_path = Path('/dev/shm') / ('uctrl' + str(os.getuid()) + '_' + profiling['session'])
            try:
                if session_path.exists():
                    stop,_,_ = owner.run('stop-session', [profiling['launcher']['path'],'--stop',profiling['session']],
                        plan['environment'], P, wall=15, cpu=15, rss_cap=2<<30,text_cap=1<<20,file_cap=32<<20,total_cap=64<<20)
                    require(owner_module.closed(stop) and stop['exit_code'] == 0, 'exact session stop failed')
                result['session_removed'] = not session_path.exists()
                require(result['session_removed'], 'profile session remains')
            except Exception as error: result.update(passed=False, session_cleanup_error=repr(error))
        result['ownership_closed'] = owner.active is None and all(owner_module.closed(c) for c in owner.commands)
        result['passed'] = bool(result['passed'] and result['ownership_closed'])
        postchecks(result, row['boot'], cursor)
        result['complete'] = bool(result['complete'] and result['passed'])
        persist()
        row['checks'].append(result); row['active'] = bool(owner and owner.active)
        row['ownership_closed'] = result.get('ownership_closed', False)
        row['stage'] = 'running' if result['passed'] else 'failed'
        if mode == list(CASES)[-1] and result['passed']:
            by_mode = {c['mode']:c for c in row['checks']}
            ratios = [by_mode[f'long-profile{i}']['request']['prompt_ms']/by_mode[f'long-base{i}']['request']['prompt_ms'] for i in range(3)]
            row.update(stage='closed', complete=True, passed=True,
                overhead=dict(profile_base_prompt_wall_ratios=ratios, median_ratio=statistics.median(ratios),
                    cap=1.05, passed=statistics.median(ratios)<=1.05),
                attribution_eligible=statistics.median(ratios)<=1.05)
        write(ROOT/'record.json',row);write(RESULT,row)
    print(json.dumps({k:result.get(k) for k in ('mode','passed','complete','ownership_closed','fresh_faults','error')}), flush=True)
    return int(not result['passed'])


if __name__ == '__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);runtime.guards.no_model_owners([]);runtime.guards.driver_guard()
        if sys.argv[1:] == ['prepare']: prepare()
        elif len(sys.argv) == 3 and sys.argv[1] == 'run' and sys.argv[2] in CASES: sys.exit(run(sys.argv[2]))
        else: raise RuntimeError('prepare|run CASE; fixed sequence: ' + ','.join(CASES))
