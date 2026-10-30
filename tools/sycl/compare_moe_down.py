#!/usr/bin/env python3
"""Fresh same-build Down policy/model evidence; repository LICENSE applies.

Main-authored adaptation of this fork's 203c6fa:compare_moe_split.py. No
XeStrata implementation input. Device gates, frontend, diagnostics and clean
speed remain distinct; neither this comparison nor its screen adopts a route.
"""
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import select
import socket
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import check_moe_split as device
import prefill_exact_checks as diagnostic
import engine_once

P, B, RESULT = diagnostic.P, diagnostic.B, diagnostic.RESULT
NATIVE, KEY = device.build.context('down')
ROOT = NATIVE / 'model-v2'
HOST = ROOT / 'frontend'
POLICIES = ('original', 'row128', 'row64')
ORDER = ('original', 'row128', 'row64', 'row128', 'row64', 'original',
         'row64', 'original', 'row128')
EXACT, RUNS = 117489, 4
require, digest, write = diagnostic.require, diagnostic.digest, diagnostic.write


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def qualified():
    record = diagnostic.runtime.validation.committed(RESULT)
    scope = record[KEY]
    checks = scope['device_checks']
    require([v['mode'] for v in checks] == device.DOWN_ORDER and
            all(v['passed'] and v['complete'] and v['ownership_closed'] and
                not v['active'] and not v['fresh_faults'] for v in checks),
            'all eight fresh same-build device checks required')
    built = device.native(scope, NATIVE)
    rt = diagnostic.runtime.validation.committed(diagnostic.runtime.RESULT)
    diagnostic.runtime.guards.no_model_owners([])
    return record, scope, built, rt


def compact(row):
    return {k: v for k, v in row.items() if k not in ('sources', 'commands_executed')}


def environment(root, rt, kernel, observed=False):
    env = diagnostic.environment(root, rt)
    env.update(STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR='1',
               STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL=kernel,
               STRATA_SYCL_PREFILL_DOWN_SCATTER='large')
    if observed:
        env['STRATA_TRACE'] = '1'
    require(all(v == '0' for k, v in env.items() if '_PROFILE' in k or '_AUDIT' in k),
            'heavy observer enabled')
    return env


def configuration(root):
    config = copy.deepcopy(diagnostic.runtime.validation.committed(diagnostic.TEMPLATE)['configuration'])
    args = config['args']
    for flag, value in (('--expert-profile', str(P/'data/expert-profile.bin')),
                        ('--expert-cache', '1256'), ('--pool-workers', '5')):
        require(args.count(flag) == 1, 'ambiguous configuration: ' + flag)
        args[args.index(flag) + 1] = value
    for flag, value in (('--max-context', '131072'), ('--kv', 'int8'),
                        ('--kv-resident', '131072'), ('--prefill', '8192'),
                        ('--spec', '4'), ('--adapt-swaps', '0'), ('--pcie-frac', '0'),
                        ('--conversation-cache-mib', '0'), ('--prompt-cache', '0')):
        require(args.count(flag) == 1 and args[args.index(flag)+1] == value,
                'selected placement/configuration changed: ' + flag)
    require('--no-prefill-borrow' not in args and args.count('--expert-cache-per-layer') == 1,
            'selected borrowed/static placement required')
    config.update(log=str(root/'engine.stderr'), env={})
    return config


def native_probe(rt):
    """Bind unchanged preload code to the separately requalified runtime.

    The original full-runtime preload gate is historical evidence, not this
    workstation's current provider set. Keep its refused plan unchanged.
    runtime.current(), called by qualified(), checks this whole current closure.
    """
    preload = diagnostic.runtime.components()
    require(preload == rt['preload'] and rt['passed'] and rt['ownership_closed'] and
            not rt['active'] and not rt['fresh_faults'], 'current preload/runtime qualification differs')
    loader = preload['loader']
    require(digest(loader) == rt['runtime_files'][loader] and
            digest(preload['library']['path']) == preload['library']['sha256'],
            'current preload/provider bytes changed')
    return dict(library=preload['library'], loader=dict(path=loader, sha256=rt['runtime_files'][loader]),
        component_source_commit=preload['source_commit'], component_original_plan=preload['original_plan'],
        current_runtime_qualification_sha256=digest(diagnostic.runtime.RESULT),
        retained_reference_until_exit=True, runtime_teardown_qualified=False)


def prepare_frontend():
    record, scope, _, rt = qualified()
    require('model_frontend' not in scope and not ROOT.exists(), 'fresh frontend root required')
    previous = diagnostic.runtime.validation.committed(diagnostic.REFERENCE,
        ('historical_regression_audit', 'benchmark_python_runtime_v4'))
    manifest = diagnostic.python_runtime.read_manifest(previous['plan']['manifest'])
    diagnostic.python_runtime.validate_manifest(manifest)
    revision = diagnostic.gen.git('rev-parse', 'HEAD').decode().strip()
    pins = diagnostic.pins(revision)
    env = environment(HOST, rt, 'original')
    require(diagnostic.python_runtime.selection(env) == manifest['selected'], 'Python provider changed')
    config = configuration(HOST)
    template = Path(config['tokenizer'])/'chat_template.jinja'
    ROOT.mkdir(mode=0o700); HOST.mkdir(mode=0o700)
    for part in ('tmp', 'cache'):
        (HOST/part).mkdir()
    row = dict(stage='prepared', source_commit=revision, sources=pins,
        selected=manifest['selected'], manifest=previous['plan']['manifest'],
        template=dict(path=str(template), sha256=digest(template)), environment=env,
        commands=[[role, [manifest['selected']['path'], str(diagnostic.python_runtime.SOURCE),
                   'probe', str(P/entry), str(template)]] for role, entry in
                  [('benchmark', 'tools/strata_bench.py'), ('server', 'serve/server.py')]],
        budgets=dict(wall=45, cpu=30, rss_cap=512<<20, text_cap=512<<10,
                     file_cap=2<<20, total_cap=8<<20),
        passed=False, active=False, ownership_closed=True, GPU_executed=False,
        criterion='Two fresh current-source definition/mock-loopback probes;late tokenizer/telemetry '
                  'and exact Python/project/provider identity. No native engine or GPU initialization.')
    write(HOST/'record.json', row); scope['model_frontend'] = compact(row); write(RESULT, record)
    print('Prepared fresh frontend;commit before host probes.', flush=True)


def frontend(admitted=True):
    public = diagnostic.runtime.validation.committed(RESULT, (KEY, 'model_frontend'))
    row = json.loads((HOST/'record.json').read_text())
    require(compact(row) == public, 'frontend receipt differs')
    if admitted:
        require(row['stage'] == 'closed' and row['passed'] and row['ownership_closed'] and not row['active'],
                'fresh closed frontend required')
    require(diagnostic.pins(row['source_commit']) == row['sources'], 'frontend/controller source changed')
    require(digest(row['template']['path']) == row['template']['sha256'], 'tokenizer template changed')
    manifest = diagnostic.python_runtime.read_manifest(row['manifest'])
    diagnostic.python_runtime.validate_manifest(manifest)
    return row, manifest


def run_frontend():
    record, scope, _, _ = qualified()
    row, manifest = frontend(False)
    require(row['stage'] == 'prepared' and 'commands_executed' not in row, 'fresh committed frontend only')
    owner = None
    try:
        allowed = {v['realpath'] for v in manifest['inventory']['files'].values()}
        owner = diagnostic.runtime.supervision.Owner(HOST/'owned')
        for role, argv in row['commands']:
            require(diagnostic.python_runtime.selection(row['environment']) == row['selected'], 'Python changed')
            entry, out, err = owner.run(role, argv, row['environment'], P, **row['budgets'])
            require(diagnostic.runtime.supervision.closed(entry) and entry['exit_code'] == 0 and
                    err.stat().st_size == 0, 'frontend probe failed')
            meta = json.loads(out.read_text())
            require(str(Path(meta['executable']).resolve()) == row['selected']['file']['realpath'],
                    'host executable changed')
            for value in [*meta['modules'], *meta['code_maps']]:
                path = Path(value).resolve()
                if path.is_relative_to(P):
                    require(row['sources'].get(str(path.relative_to(P))) == digest(path), 'unfrozen project import')
                else:
                    require(str(path) in allowed, 'unfrozen Python provider: ' + str(path))
            if role == 'server':
                x = meta['exercised']
                require(x['late_tokenizer_import'] and x['production_telemetry_initialized'] and
                        x['native_engine'] is False, 'mock coverage missing')
            diagnostic.python_runtime.validate_manifest(manifest)
        require(diagnostic.pins(row['source_commit']) == row['sources'], 'source changed during host probes')
        row.update(stage='closed', passed=True)
    except BaseException as error:
        row.update(stage='failed', passed=False, error=repr(error))
    finally:
        row.update(active=bool(owner and owner.active), ownership_closed=bool(owner and owner.active is None and
            all(diagnostic.runtime.supervision.closed(c) for c in owner.commands)),
            commands_executed=owner.commands if owner else [])
        row['passed'] = bool(row['passed'] and row['ownership_closed'])
        write(HOST/'record.json', row); scope['model_frontend'] = compact(row); write(RESULT, record)
    print(json.dumps({k: row.get(k) for k in ('stage', 'passed', 'ownership_closed', 'error')}), flush=True)
    require(row['passed'], 'frontend did not qualify')


def basis():
    _, _, built, rt = qualified()
    host, manifest = frontend()
    row = json.loads((ROOT/'basis.json').read_text())
    require(row['source_commit'] == host['source_commit'] and row['sources'] == host['sources'] and
            row['binary'] == built['binaries']['strata'] and row['native_source_commit'] == built['source_commit'] and
            row['runtime_sha256'] == digest(diagnostic.runtime.RESULT) and row['boot'] == rt['boot'] and
            row['native_validation_probe'] == native_probe(rt),
            'model basis changed')
    require(diagnostic.models.file_identity(row['profile']['path']) == row['profile']['identity'] and
            digest(row['profile']['path']) == row['profile']['sha256'], 'profile changed')
    diagnostic.models.model_pins(configuration(ROOT), row['model_assets'])
    return row, rt, manifest


def route_samples(log, count):
    prefixes = ['strata prefill: layer-major candidate, ', 'strata MoE expert bank: ',
        'strata MoE window configuration: ', 'strata MoE accumulator window: ',
        'strata MoE expert bank complete: ', 'strata MoE split admission: ', 'strata MoE split complete: ']
    groups = [[line for line in log.splitlines() if line.startswith(prefix)] for prefix in prefixes]
    require(all(len(rows) == count for rows in groups), 'missing/duplicate request route records')
    fused = re.findall(r'^strata MoE Down fusion: fused_jobs=(\d+) fused_rows=(\d+) separate_jobs=(\d+) min_rows=2334$', log, re.M)
    require(len(fused) == count, 'missing/duplicate Down dispatch counters')
    result = []
    for i in range(count):
        route = diagnostic.route('\n'.join(rows[i] for rows in groups), dict(rows=EXACT-1, read=EXACT, split=True))
        jobs, rows, separate = map(int, fused[i])
        require(route['residual_host_bytes'] == 0 and route['work']['windows'] == 48 and
                jobs > 0 and rows >= jobs*2334 and jobs+separate == route['work']['jobs'],
                'actual split/Down route differs')
        route.update(down_fused_jobs=jobs, down_fused_rows=rows, down_separate_jobs=separate,
            limitation='Source-bound same-build policy plus actual success counters;not a per-kernel model trace. '
                       'Requested work/bytes are not physical traffic,critical-path service or saturation.')
        result.append(route)
    return result


def comparison_plan():
    return dict(order=list(ORDER), runs_per_process=RUNS, input_tokens=EXACT, output_cap=24,
        question='Do row-cached Down epilogues improve whole117K prefill over the same binary original split policy?',
        constants='Same9a79cd352 model binary,split117488-row accumulator,128effective prefill bank slots,'
                  '1256unchanged decode slots,BF16/vLLM/mode10,8192chunks,5workers,spec4,INT8 resident131072context.'
                  'Only immutable pre-exec Down kernel choice differs.',
        samples='Three Latin-order blocks,three independent fresh processes per policy,four exact-count '
                'requests/process. First long use separate;median of subsequent three is the process unit. '
                'Count-only preflight warms tokenizer caches;not cold-disk latency. Retain all PP/time/TTFT samples.',
        acceptance='Complete all9processes,no speed-based early stop. For each candidate versus original, '
                   'all3within-block median-time ratios<1 and their geometric mean<=0.995 permit broader '
                   'qualification. Select row64 over qualifying row128 only if all3paired ratios<1 and '
                   'geomean<=0.9975;otherwise keep row128. No adoption from this screen. '
                   'Target1000 requires all3selected-policy warmed process medians>=1000tok/s.',
        stopping='Stop first source,runtime,ownership,validation,fault,budget,route or sample failure. '
                 'Never retry/replace/exclude a process or change criteria after readings.',
        limits='No sustained decode/task/agent/full262144/RESTORE qualification. The original split policy is '
               'a same-build control,not the selected6e40fallback;matched fallback and independent decode '
               'preservation remain necessary before adoption. Placement/window/actual positive Down counters '
               'must match policy;record dynamic jobs and requested bytes rather than assume identical routing.')


def prepare(kind, kernel=None):
    record, scope, built, rt = qualified()
    host, _ = frontend()
    history = scope.get('model_diagnostics', [])
    if kind == 'diagnostic':
        require(kernel in POLICIES and [v['kernel'] for v in history] == list(POLICIES[:POLICIES.index(kernel)]) and
                all(v['passed'] and v['ownership_closed'] and not v['fresh_faults'] for v in history),
                'ordered successful model diagnostics required;no retry')
        if kernel == 'original':
            require(not (ROOT/'basis.json').exists(), 'fresh model basis required')
            config = configuration(ROOT)
            profile = P/'data/expert-profile.bin'
            shared = dict(source_commit=host['source_commit'], sources=host['sources'],
                binary=built['binaries']['strata'], native_source_commit=built['source_commit'],
                runtime_sha256=digest(diagnostic.runtime.RESULT), boot=rt['boot'],
                native_validation_probe=native_probe(rt),
                profile=dict(path=str(profile), identity=diagnostic.models.file_identity(profile), sha256=digest(profile)),
                model_assets=diagnostic.models.model_pins(config, full=True))
            write(ROOT/'basis.json', shared)
        shared, _, _ = basis()
        plans = [(kernel, ROOT/('diagnostic-'+kernel), 1, True)]
        key = 'model_plan'
        require(key not in scope or scope[key]['stage'] == 'closed', 'preceding model plan still open')
    else:
        require(kind == 'clean' and [v['kernel'] for v in history] == list(POLICIES) and
                all(v['passed'] and v['ownership_closed'] and not v['fresh_faults'] for v in history),
                'all three model diagnostics required before clean speed')
        require('clean_down_comparison' not in scope, 'fresh comparison only')
        shared, _, _ = basis()
        plans = [(policy, ROOT/'clean'/str(i), RUNS, False) for i, policy in enumerate(ORDER)]
        key = 'clean_down_comparison'
    values = []
    for policy, root, count, observed in plans:
        require(not root.exists(), 'fresh model process root required')
        root.mkdir(mode=0o700, parents=True)
        for part in ('tmp', 'cache'):
            (root/part).mkdir()
        config = configuration(root)
        write(root/'config.json', config)
        write(root/'plan.json', {k: shared[k] for k in ('binary', 'native_validation_probe')})
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        command = ['/bin/sh', str(P/'bench.sh'), '--config', str(root/'config.json'), '--port', str(port),
            '--prefill-exact-tokens', str(EXACT), '--think', '--runs', str(count), '--profile-request-gates',
            '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600',
            '--out', str(root/'bench')]
        values.append(dict(kernel=policy, root=str(root), runs=count, observed=observed, command=command,
            generated={n: digest(root/n) for n in ('plan.json', 'config.json')},
            environment_sha256=fingerprint(environment(root, rt, policy, observed))))
    row = dict(stage='prepared', kind=kind, source_commit=host['source_commit'],
        basis_sha256=digest(ROOT/'basis.json'), native_source_commit=shared['native_source_commit'],
        binary=shared['binary'], plans=values, processes=[], plan=comparison_plan() if kind == 'clean' else
        dict(order=list(POLICIES), criterion='One exact117K thinking-on diagnostic per immutable policy;'
             'full GPU residual/split guard/drain,actual positive Down dispatch and matching startup policy.'
             'Observer-charged times are not clean speed;no model-quality or decode adoption.'),
        budgets=dict(wall=1800, cpu=5400, rss_cap=104<<30, text_cap=64<<20,
                     file_cap=128<<20, total_cap=512<<20),
        complete=False, passed=False, active=False, ownership_closed=True)
    write(ROOT/('comparison.json' if kind == 'clean' else 'diagnostic.json'), row)
    scope[key] = row; write(RESULT, record)
    print('Prepared '+kind+' '+str(kernel)+';commit before native execution.', flush=True)


def decision(processes):
    require(len(processes) == len(ORDER) and [v['kernel'] for v in processes] == list(ORDER), 'incomplete comparison')
    require(all(v['passed'] and v['complete'] and v['ownership_closed'] and not v['fresh_faults'] and
                v['clean_speed'] for v in processes), 'failed,incomplete or observed process')
    require(len({(v['native_identity']['pid'], v['native_identity']['start_ticks']) for v in processes}) == len(ORDER),
            'independent native processes required')
    require(all(math.isfinite(v['subsequent_median_ms']) and v['subsequent_median_ms'] > 0 for v in processes),
            'invalid process median')
    medians = [v['subsequent_median_ms'] for v in processes]
    ratios = {}
    for policy, control in (('row128', 'original'), ('row64', 'original'), ('row64', 'row128')):
        values = []
        for start in (0, 3, 6):
            block = {processes[i]['kernel']: medians[i] for i in range(start, start+3)}
            values.append(block[policy]/block[control])
        ratios[policy+'/'+control] = dict(paired_time_ratios=values,
            geometric_mean=math.exp(statistics.mean(math.log(v) for v in values)))
    def better(name, limit):
        row = ratios[name]
        return all(v < 1 for v in row['paired_time_ratios']) and row['geometric_mean'] <= limit
    selected = 'row128' if better('row128/original', .995) else None
    if better('row64/original', .995) and (selected is None or better('row64/row128', .9975)):
        selected = 'row64'
    return dict(ratios=ratios, eligible_policy_for_broader_qualification=selected,
        all_three_selected_warm_medians_ge1000=bool(selected and all(EXACT*1000/medians[i] >= 1000
            for i, v in enumerate(processes) if v['kernel'] == selected)),
        inference_adopted=False, sustained_decode_qualified=False, physical262144_qualified=False)


def run(kind, index):
    record, scope, _, _ = qualified()
    shared, rt, manifest = basis()
    key = 'clean_down_comparison' if kind == 'clean' else 'model_plan'
    path = ROOT/('comparison.json' if kind == 'clean' else 'diagnostic.json')
    row = json.loads(path.read_text())
    require(row == scope[key] and row['kind'] == kind and row['stage'] in ('prepared', 'running') and
            len(row['processes']) == index < len(row['plans']) and
            all(v['passed'] for v in row['processes']) and row['basis_sha256'] == digest(ROOT/'basis.json'),
            'fresh committed ordered process required;no retry')
    plan = row['plans'][index]; root = Path(plan['root']); count = plan['runs']
    env = environment(root, rt, plan['kernel'], plan['observed'])
    require(not (root/'execution.json').exists() and fingerprint(env) == plan['environment_sha256'] and
            all(digest(root/n) == sha for n, sha in plan['generated'].items()), 'model plan changed')
    cursors = re.findall(r'^-- cursor: (.+)$', subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15), re.M)
    require(len(cursors) == 1, 'fresh fault cursor required')
    outcome = dict(index=index, kernel=plan['kernel'], complete=False, passed=False,
        host_probe_accepted=False, model_executed=None,
        clean_speed=not plan['observed'], adoption_qualified=False)
    owner = None
    def persist():
        write(root/'execution.json', dict(outcome, active=bool(owner and owner.active),
              commands=owner.commands if owner else []))
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
    def interact(io, entry):
        identity = None
        for run_index in range(count):
            for stage in ('begin', 'end'):
                while True:
                    line = io.line(900)
                    if line.startswith('STRATA_PROFILE_GATE '):
                        break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                        dict(stage=stage, phase='prefill-only', run=run_index), 'unexpected request gate')
                delta = [a-b for a, b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()), initial)]
                require(min(delta) >= 0 and abs(delta[0]-delta[1]) <= 50000000, 'suspend/clock discontinuity')
                if stage == 'begin':
                    actual = json.loads((root/'engine-start.json').read_text())['native']; pid = actual['pid']
                    live = diagnostic.runtime.supervision.identity(pid)
                    require(live and live['start_ticks'] == actual['start_ticks'] and live['sid'] == entry['pid'] and
                            str(Path('/proc', str(pid), 'exe').resolve()) == str(Path(shared['binary']['path']).resolve()) and
                            (identity is None or identity == actual), 'owned native identity differs')
                    identity = actual
                    expected = engine_once.native_environment(shared, env)
                    actual_env = dict(v.split(b'=', 1) for v in Path('/proc', str(pid), 'environ').read_bytes().split(b'\0') if b'=' in v)
                    require(all(actual_env.get(k.encode()) == v.encode() for k, v in expected.items()) and
                            set(actual_env)-{k.encode() for k in expected} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'},
                            'native pre-exec environment differs')
                    mapped = diagnostic.runtime.executable_mappings(Path('/proc', str(pid), 'maps').read_text())
                    require(mapped <= set(rt['runtime_files']) | {str(Path(shared['binary']['path']).resolve())} and
                            {shared['native_validation_probe']['loader']['path'], shared['native_validation_probe']['library']['path'],
                             env['ZE_ENABLE_ALT_DRIVERS']} <= mapped, 'native provider differs')
                    outcome['physical_device'] = device.device.live_device(pid, 'Intel(R) Arc(TM) B570 Graphics',
                        device.device.device_inventory(rt), rt)
                    allowed = {v['realpath'] for v in manifest['inventory']['files'].values()}
                    diagnostic.python_runtime.check_running(entry['pid'], P/'tools/strata_bench.py',
                        manifest['selected'], env, allowed, plan['command'][2:])
                    outcome['native_identity'] = actual; outcome['model_executed'] = True
                require(not (root/'validation.log').exists() or (root/'validation.log').stat().st_size == 0,
                        'parameter-validation warning/error')
                print(f'{kind} {index} {plan["kernel"]} request{run_index} {stage}', flush=True)
                io.send(b'continue\n')
        io.proc.stdin.close(); deadline = time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic() < deadline, 'teardown deadline')
            if select.select([io.proc.stdout], [], [], .05)[0]:
                io.read(); io.pending.clear()
        io.exit(1)
    try:
        owner = diagnostic.runtime.supervision.Owner(root/'owned', persist)
        entry, _, _ = owner.run('bench', plan['command'], env, P, interaction=interact, **row['budgets'])
        require(diagnostic.runtime.supervision.closed(entry) and entry['exit_code'] == 0, 'model process failed')
        files = list((root/'bench').glob('*/results.json')); require(len(files) == 1, 'missing/duplicate official result')
        result = json.loads(files[0].read_text()); cohort = result['results']['prefill_only']
        samples, preflight = cohort['requests'], cohort['count_preflight']
        require(result['complete'] and not result['results']['failures'] and len(samples) == len(preflight) == count,
                'official cohort incomplete')
        for i, sample in enumerate(samples):
            require(all(sample.get(k) is True for k in ('ok', 'eligible', 'stream_complete', 'thinking_requested', 'reasoning_observed')) and
                    sample['request_sha256'] == preflight[i]['body_sha256'] and preflight[i]['counted_input_tokens'] == EXACT and
                    sample['actual_prompt_tokens'] == sample['actual_read_tokens'] == EXACT and sample['actual_reused_tokens'] == 0 and
                    1 <= sample['generated_tokens'] <= 24 and sample['prompt_tps_source'] == 'engine' and
                    math.isfinite(sample['prompt_ms']) and sample['prompt_ms'] > 10000, 'workload/clock differs')
            prior = row['processes'][0]['samples'][i] if row['processes'] else None
            if prior:
                require(sample['request_sha256'] == prior['request_sha256'], 'cross-process body differs')
        require(diagnostic.client.verify_request_journal(files[0].parent)['valid'], 'request journal invalid')
        basis()
        config = json.loads((root/'config.json').read_text())
        end = diagnostic.gen.native_end(root, shared, dict(config=config), shared['boot'], entry['pid'])
        log = (root/'engine.stderr').read_text()
        proof = diagnostic.runtime.validation.preload_evidence(log, end['native']['pid'])
        require(proof['loader'] == shared['native_validation_probe']['loader']['path'] and
                [s for s in log.splitlines() if s.startswith('[native16-down] ')] ==
                ['[native16-down] fusion=large min_rows=2334 split_kernel='+plan['kernel']] and
                not any(t in log for t in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called', 'teardown uncertain')),
                'actual policy/runtime differs')
        if not plan['observed']:
            require(not any(t in log for t in ('strata trace:', 'strata prefill dense:', 'strata loan audit:',
                        'strata MoE mixed owners:', 'SYCL_EVENT_QUERY', 'strata decode cpu:', 'strata decode timing:')),
                    'clean process emitted observer records')
        routes = route_samples(log, count)
        require(all(digest(root/n) == sha for n, sha in plan['generated'].items()),
                'generated process inputs changed during execution')
        fields = ('request_sha256', 'actual_prompt_tokens', 'actual_read_tokens', 'actual_reused_tokens',
                  'generated_tokens', 'prompt_ms', 'prompt_tps', 'prompt_tps_source', 'ttft_s', 'total_s',
                  'thinking_requested', 'reasoning_observed', 'ok', 'eligible', 'stream_complete')
        outcome.update(complete=True, host_probe_accepted=True, native_exit_code=0, validation=proof, routes=routes,
            samples=[{k: s[k] for k in fields if k in s} for s in samples],
            subsequent_median_ms=statistics.median(s['prompt_ms'] for s in samples[1:]) if count > 1 else None)
        if kind == 'clean' and index == len(ORDER)-1:
            diagnostic.models.model_pins(config, shared['model_assets'], full=True)
    except BaseException as error:
        outcome.update(error=repr(error), complete=False, host_probe_accepted=False)
    finally:
        if owner:
            diagnostic.runtime.validation.finish(outcome, owner, cursors[0], shared['boot'])
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete']); persist()
        row['processes'].append(outcome)
        row.update(stage='running' if outcome['passed'] else 'failed', ownership_closed=outcome.get('ownership_closed', False),
                   active=bool(owner and owner.active))
        if outcome['passed'] and len(row['processes']) == len(row['plans']):
            row.update(stage='closed', complete=True, passed=True)
            if kind == 'clean':
                row['decision'] = decision(row['processes'])
        write(path, row); scope[key] = row
        if kind == 'diagnostic':
            scope.setdefault('model_diagnostics', []).append(dict(outcome, kernel=plan['kernel']))
        write(RESULT, record)
    print(json.dumps({k: outcome.get(k) for k in ('kernel', 'passed', 'complete', 'ownership_closed', 'fresh_faults', 'error')}), flush=True)
    require(outcome['passed'], 'model process did not qualify')


if __name__ == '__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args = sys.argv[1:]
        require(args, 'prepare-frontend|frontend|prepare-diagnostic POLICY|diagnostic|prepare|run INDEX')
        if args == ['prepare-frontend']:
            prepare_frontend()
        elif args == ['frontend']:
            run_frontend()
        elif len(args) == 2 and args[0] == 'prepare-diagnostic':
            prepare('diagnostic', args[1])
        elif args == ['diagnostic']:
            run('diagnostic', 0)
        elif args == ['prepare']:
            prepare('clean')
        elif len(args) == 2 and args[0] == 'run':
            run('clean', int(args[1]))
        else:
            raise RuntimeError('unknown action')
