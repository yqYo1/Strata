#!/usr/bin/env python3
"""Discover the selected long-prefill route, without a new engine or profiler.

Main-authored; repository LICENSE. Existing successful DRAFT_PREFILL markers
establish callback dispatch only. They do not qualify device coverage or speed.
"""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import select
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import full_context_model as models
import generation_engine as gen
import runtime_probe as runtime

P, B, A = gen.P, gen.B, gen.A
ROOT = B / 'prefill-route-discovery-v1'
RESULT = A / 'prefill-resources-current.json'
KEY = 'route_discovery'
TEMPLATE = A / 'phase-prefill-speed-current.json'
CONFIG = A / 'restored-phase-bench-current.json'
HOST_TEST = P / 'tests/core/test_prefill_route.py'
require, digest, write = runtime.require, runtime.digest, runtime.write
FOOTER = r'^strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read[^\n]*, (\d+) generated[^\n]*$'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def compact(row):
    return {k: v for k, v in row.items() if k not in ('environment', 'models', 'config', 'commands')}


def qualification():
    rt = runtime.validation.committed(runtime.RESULT)
    native = runtime.current(rt)
    require(native.get('binary_origin_source_commit', native['source_commit']) ==
            '6e40afc1beddd059ee518617f6c1efce1da14aca', 'original selected engine required')
    require(rt['passed'] and rt['ownership_closed'] and not rt['active'] and rt['fresh_faults'] == [],
            'closed current-runtime qualification required')
    base = models.setup()
    return rt, native, base


def prepare():
    container = runtime.validation.committed(RESULT)
    require(KEY not in container['event_reconciliation'] and not ROOT.exists(), 'fresh single-use route plan only')
    rt, native, base = qualification()
    revision = gen.git('rev-parse', 'HEAD').decode().strip()
    pins = gen.benchmark_pins(revision)
    require(gen.git('show', revision + ':' + str(HOST_TEST.relative_to(P))) == HOST_TEST.read_bytes(),
            'commit exact host controls')
    template = runtime.validation.committed(TEMPLATE)['environment_template']
    config = copy.deepcopy(runtime.validation.committed(CONFIG)['configuration'])
    require(config['exe'] == str(P / 'tools/sycl/engine_once.py'), 'qualified native wrapper required')
    args = config['args']
    require(all(args[args.index(k) + 1] == v for k, v in (
        ('--expert-cache', '1256'), ('--kv-resident', '131072'), ('--max-context', '131072'),
        ('--kv', 'int8'), ('--prefill', '8192'), ('--spec', '4'), ('--pool-workers', '5'),
        ('--pcie-frac', '0'), ('--adapt-swaps', '0'), ('--prompt-cache', '0'),
        ('--conversation-cache-mib', '0'), ('--suffix-draft', '0'))) and
        '--no-token-graph' in args and '--no-coupled-draft' in args, 'selected decode layout/configuration differs')
    require(not any(k in template for k in ('LD_PRELOAD', 'STRATA_TRACE', 'STRATA_SNAPSHOT_VERIFY',
        'STRATA_DRAFT_TIMING', 'STRATA_STATE_HASH', 'STRATA_CPU_WORK_CENSUS')), 'unexpected template observer')
    env = {k: v.replace(template['STRATA_MEASUREMENT_ROOT'], str(ROOT)) for k, v in template.items()}
    env.update(STRATA_TRACE='1', STRATA_SNAPSHOT_VERIFY='1',
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(ROOT / 'parameter.validation'))
    require(env.get('UR_ENABLE_LAYERS') == 'UR_LAYER_PARAMETER_VALIDATION', 'silent parameter validation required')
    for k in ('UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD', 'NEOReadDebugKeys', 'EnableDirectSubmission',
              'STRATA_SYCL_PREFILL_SAME_QUEUE', 'SYCL_CACHE_PERSISTENT', 'ZES_ENABLE_SYSMAN'):
        require(env[k] == rt['environment'][k], 'qualified safety setting differs: ' + k)
    loader = rt['preload']['loader']
    probe = dict(library=rt['preload']['library'], loader=dict(path=loader, sha256=rt['runtime_files'][loader]),
        retained_reference_until_exit=True, ordinary_runtime_teardown_qualified=False)
    models.model_pins(config, base['models'], full=True)
    ROOT.mkdir(mode=0o700)
    for name in ('tmp', 'cache'): (ROOT / name).mkdir()
    config.update(log=str(ROOT / 'engine.stderr'), env={})
    native_plan = dict(binary=native['binary'], native_validation_probe=probe)
    write(ROOT / 'config.json', config); write(ROOT / 'plan.json', native_plan)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    row = dict(stage='prepared', source_commit=revision, benchmark_fingerprint=fingerprint(pins),
        benchmark_files=len(pins), native_source='6e40afc1beddd059ee518617f6c1efce1da14aca', binary=native['binary'],
        runtime_sha256=digest(runtime.RESULT), template_sha256=digest(TEMPLATE), config_reference_sha256=digest(CONFIG),
        model_reference=base['reference'], model_fingerprint=fingerprint(base['models']), models=base['models'],
        native_validation_probe=probe, environment=env, environment_sha256=fingerprint(env), config=config,
        generated_inputs={str(ROOT / n): digest(ROOT / n) for n in ('config.json', 'plan.json')},
        command=['/bin/sh', str(P / 'bench.sh'), '--config', str(ROOT / 'config.json'), '--port', str(port),
            '--prefill-only', '--prefill-tokens', '98304', '--think', '--runs', '1', '--profile-request-gates',
            '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600',
            '--out', str(ROOT / 'bench')],
        question='Does every final-layer callback of the actual selected117K long request use Prefill::draft_kv '
            'successfully,or does any callback fall back to MtpDrafter::prefill and its graph launches?',
        criteria='One short official warmup then one117489-token fresh thinking-on request,24-output cap. '
            'Require15 successful batched mode2 markers,14x8192+2800 cells=117488;actual GPU BF16 residual,MoE10, '
            'Full48 dense restoration,complete native/journal extents,matching sources/models/runtime and actual '
            'providers at both request gates,silent parameter validation,normal owners/exit and no fresh fault. '
            'Stop first failure;no retry. Existing SNAPSHOT_VERIFY marker adds no synchronization here; '
            'snapshot restore/readback is excluded by both disabled caches. STRATA_TRACE is charged.',
        limits='Dispatch discovery only. No collector/XPTI,new runtime startup override or native rebuild. '
            'No timing comparison,observer-cost,complete device ledger,library-internal graph absence, '
            'hardware bottleneck/saturation,decode preservation,task,physical262144 or adoption qualification.',
        retention='Keep compact callback conservation,source/config/runtime identity and result. '
            'Retire successful raw output/JIT after committed replay. Selected original engine and12 fixtures stay unchanged.',
        host_controls=dict(command=['/usr/bin/python3', '-B', str(HOST_TEST)], source_sha256=digest(HOST_TEST),
            python_sha256=digest('/usr/bin/python3')),
        budgets=dict(wall=1200, cpu=7200, rss_cap=104 << 30, text_cap=16 << 20,
                     file_cap=64 << 20, total_cap=128 << 20),
        active=False, ownership_closed=True, passed=False, production_changed=False,
        boot=rt['boot'], driver=rt['driver'])
    write(ROOT / 'record.json', row)
    container['event_reconciliation'][KEY] = compact(row); write(RESULT, container)
    print('Prepared single-use117K callback-route discovery;commit plan before execution.', flush=True)


def current(row):
    require(runtime.validation.committed(RESULT, ('event_reconciliation', KEY)) == compact(row), 'commit exact route plan')
    rt, native, base = qualification()
    require(fingerprint(gen.benchmark_pins(row['source_commit'])) == row['benchmark_fingerprint'] and
        digest(runtime.RESULT) == row['runtime_sha256'] and digest(TEMPLATE) == row['template_sha256'] and
        digest(CONFIG) == row['config_reference_sha256'] and native['binary'] == row['binary'] and
        base['reference'] == row['model_reference'] and fingerprint(base['models']) == row['model_fingerprint'] and
        base['models'] == row['models'] and fingerprint(row['environment']) == row['environment_sha256'] and
        all(digest(p) == sha for p, sha in row['generated_inputs'].items()), 'route inputs changed')
    require(digest(HOST_TEST) == row['host_controls']['source_sha256'] and
        digest('/usr/bin/python3') == row['host_controls']['python_sha256'] and
        gen.git('show', row['source_commit'] + ':' + str(HOST_TEST.relative_to(P))) == HOST_TEST.read_bytes(),
        'host controls changed')
    require(json.loads((ROOT / 'config.json').read_text()) == row['config'] and
        json.loads((ROOT / 'plan.json').read_text()) == dict(binary=row['binary'],
            native_validation_probe=row['native_validation_probe']), 'generated native inputs differ')
    models.model_pins(row['config'], row['models'])
    return rt


def analyze(text, sample):
    headers = list(re.finditer(FOOTER, text, re.M))
    require(len(headers) == 2 and int(headers[0][1]) < 32768, 'warmup/long footer extent differs')
    require(tuple(map(int, headers[1].groups())) == (117489, 0, 117489, sample['generated_tokens']),
            'actual long request differs')
    part = text[headers[0].end():headers[1].end()]
    evidence, = gen.request_evidence(part, [sample])
    require(evidence == dict(read_tokens=117489, draft_cells=117488, batch_count=15, path='batched', mode=2),
            'unexpected callback dispatch conservation')
    require(not gen.draft_rows(text[headers[1].end():]), 'unbound trailing draft callback')
    placement = re.findall(r'^strata prefill: layer-major candidate, tokens=(\d+) chunk=8192 '
        r'residual_dtype=bf16 residual_element_bytes=2 residual_total=(\d+) residual_gpu_rows=(\d+) '
        r'residual_gpu=(\d+) residual_host=0 residual_staging=0 ', part, re.M)
    require(placement == [('117488', '2406154240', '117488', '2406154240')] and
        'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows=81920' in part and
        len(re.findall(r'^strata prefill dense: phase=restored residency=Full activations=48 ', part, re.M)) == 1,
        'actual BF16/GPU/MoE10/dense restoration differs')
    require(not re.search(r'^strata (?:draft kv|serve: SNAPSHOT_VERIFY|serve: STATE_HASH)', text, re.M),
            'unexpected synchronization/readback observer')
    return dict(**evidence, final_layer_callback_graph_fallbacks=0, full_dense_layers_restored=48,
        GPU_BF16_residual_rows=117488, MoE_mode=10, library_internal_graphs_qualified=False)


def run(row):
    rt = current(row); require(row['stage'] == 'prepared' and not (ROOT / 'owned').exists(), 'single-use route owner')
    initial = copy.deepcopy(row); owner = None
    before = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15)
    cursor = next(v.split(': ', 1)[1] for v in before.splitlines() if v.startswith('-- cursor: '))

    def persist():
        write(ROOT / 'execution.json', dict(active=bool(owner and owner.active is not None),
            commands=owner.commands if owner else []))

    def interact(io, leader):
        for stage in ('begin', 'end'):
            while True:
                line = io.line(900)
                if line.startswith('STRATA_PROFILE_GATE '): break
            require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                dict(stage=stage, phase='prefill-only', run=0), 'request gate differs')
            claim = json.loads((ROOT / 'engine-start.json').read_text())
            ident = runtime.supervision.identity(claim['native']['pid'])
            require(ident and ident['start_ticks'] == claim['native']['start_ticks'] and ident['sid'] == leader['pid'] and
                Path('/proc', str(ident['pid']), 'exe').resolve() == Path(row['binary']['path']).resolve(), 'native ownership differs')
            actual = dict(v.split(b'=', 1) for v in Path('/proc', str(ident['pid']), 'environ').read_bytes().split(b'\0') if b'=' in v)
            expected = dict(row['environment'], LD_PRELOAD=row['native_validation_probe']['library']['path'],
                STRATA_UR_PROBE_EXPECTED_LOADER=row['native_validation_probe']['loader']['path'])
            require(all(actual.get(k.encode()) == v.encode() for k, v in expected.items()) and
                set(actual) - {k.encode() for k in expected} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'}, 'native environment differs')
            mapped = runtime.executable_mappings(Path('/proc', str(ident['pid']), 'maps').read_text())
            require(all(p in rt['runtime_files'] or p == str(Path(row['binary']['path']).resolve()) for p in mapped) and
                row['native_validation_probe']['library']['path'] in mapped and
                any('libze_intel_gpu.so.1.17.39758' in p for p in mapped), 'actual runtime provider differs')
            require(not (ROOT / 'parameter.validation').exists() or (ROOT / 'parameter.validation').stat().st_size == 0,
                    'parameter validation warning')
            row[stage + '_verified'] = True
            io.send(b'continue\n')
        io.proc.stdin.close(); deadline = time.monotonic() + 180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic() < deadline, 'benchmark teardown deadline')
            if select.select([io.proc.stdout], [], [], .05)[0]: io.read(); io.pending.clear()
        io.exit(1)

    try:
        owner = runtime.supervision.Owner(ROOT / 'owned', persist)
        entry, _, _ = owner.run('host-controls', row['host_controls']['command'],
            dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', PYTHONDONTWRITEBYTECODE='1'), P,
            wall=45, cpu=45, rss_cap=2 << 30, text_cap=1 << 20, file_cap=1 << 20, total_cap=128 << 20)
        require(runtime.supervision.closed(entry) and entry['exit_code'] == 0, 'host route controls failed;no GPU launch')
        row['host_controls_passed'] = True
        entry, _, _ = owner.run('bench', row['command'], row['environment'], P, interaction=interact, **row['budgets'])
        require(runtime.supervision.closed(entry) and entry['exit_code'] == 0, 'benchmark owner/exit failed')
        result_file, = (ROOT / 'bench').glob('*/results.json'); result = json.loads(result_file.read_text())
        sample, = result['results']['prefill_only']['requests']
        require(result['complete'] and not result['results']['failures'] and result['params']['mode'] == 'prefill-only' and
            result['params']['runs'] == 1 and result['params']['think'] is True and
            all(sample.get(k) is True for k in ('ok', 'eligible', 'stream_complete', 'thinking_requested', 'reasoning_observed')),
            'official benchmark request incomplete/ineligible')
        sys.path.insert(0, str(P / 'tools')); from strata_bench import verify_request_journal
        journal, = (ROOT / 'bench').glob('*/requests.jsonl'); checked = verify_request_journal(journal.parent)
        require(checked['valid'] and checked['complete'], 'request journal incomplete')
        events = [json.loads(v) for v in journal.read_text().splitlines()]
        results = [v for v in events if v['kind'] == 'result']
        require(len(results) == 2 and results[0]['phase'] == 'warm-up/chat' and results[1]['phase'] == 'prefill-only' and
            dict(results[1]['sample'], run=results[1]['run']) == sample, 'journal request binding differs')
        native = gen.native_end(ROOT, row, dict(config=row['config']), row['boot'], entry['pid'])
        text = (ROOT / 'engine.stderr').read_text()
        row['activation'] = runtime.validation.preload_evidence(text, native['native']['pid'])
        boundary = re.search(r'^strata UR validation: .*phase=before-main[^\n]*\n', text, re.M)
        require(boundary and not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|teardown uncertain|cleanup failed',
            text[boundary.end():]), 'runtime error after canary')
        row['route'] = analyze(text, sample)
        current(initial); models.model_pins(row['config'], row['models'], full=True)
        row.update(stage='route-verified', passed=True, host_probe_accepted=True,
            request_sha256=sample['request_sha256'], generated_tokens=sample['generated_tokens'],
            native=native['native'], full_model_hashes_matched=15,
            raw_evidence=[dict(path=str(p), sha256=digest(p), bytes=p.stat().st_size)
                for p in (ROOT / 'engine.stderr', result_file, journal)])
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
        if (ROOT / 'engine.stderr').exists(): row['error_excerpt'] = (ROOT / 'engine.stderr').read_text()[-4096:]
    finally:
        if owner: runtime.validation.finish(row, owner, cursor, row['boot'])
        row['passed'] = bool(row['passed'] and row.get('driver_after') == row['driver'])
        if not row['passed']: row['stage'] = 'failed'
        row['active'] = bool(owner and owner.active is not None)
        write(ROOT / 'record.json', row)
        container = json.loads(RESULT.read_text()); container['event_reconciliation'][KEY] = compact(row); write(RESULT, container)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'ownership_closed', 'fresh_faults', 'route', 'error')}), flush=True)
    return int(not row['passed'])


if __name__ == '__main__':
    require(sys.argv[1:] in (['prepare'], ['run']), 'prepare|run')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime.guards.no_model_owners([]); runtime.guards.driver_guard()
        if sys.argv[1:] == ['prepare']: prepare()
        else: sys.exit(run(json.loads((ROOT / 'record.json').read_text())))
