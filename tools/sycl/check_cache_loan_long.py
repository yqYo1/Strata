#!/usr/bin/env python3
"""Main's official bench.sh long-input admission for checked DFlash2 loans.

Repository LICENSE. Compare fixed owned256/borrowed1536 residency in separate
same-build processes. First/subsequent requests are diagnostics,not adoption.
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
import sys
import time

sys.dont_write_bytecode = True
import check_cache_loan_model as small
import drafter_benchmark as bench
import compare_drafters as identity

P, B, STATUS = small.P, small.B, small.STATUS
ROOT, KEY = B / 'dflash2-cache-loan-long-v1', 'dflash2_cache_loan_long_v1'
SELF = Path(__file__).resolve()
require, digest, write, signature = small.require, small.digest, small.write, small.signature
READ, RUNS = 32769, 2


def compact(row):
    out = {k: v for k, v in row.items() if k not in ('sources', 'models', 'runtime_files', 'native_build')}
    out['source_inputs'] = dict(files=len(row['sources']), sha256=signature(row['sources']))
    out['native_build'] = {k: row['native_build'][k] for k in ('source_commit', 'project_source_sha256', 'record')}
    out['plans'] = [{k: v for k, v in p.items() if k != 'environment' or row['stage'] in ('prepared', 'ready')} |
        dict(environment_sha256=signature(p['environment'])) for p in row['plans']]
    out['processes'] = [{k: v for k, v in p.items() if k != 'commands'} | dict(commands=[
        dict(label=c['label'], exit_code=c.get('exit_code'), error=c.get('error'),
            ownership_closed=small.loan.ownership.closed(c), errors=c.get('errors', []), cleanup=c.get('cleanup', []))
        for c in p.get('commands', [])]) for p in row['processes']]
    if 'host_commands' in out:
        out['host_commands'] = [dict(label=c['label'], exit_code=c.get('exit_code'),
            ownership_closed=small.loan.ownership.closed(c), error=c.get('error')) for c in out['host_commands']]
    return out


def save(row):
    write(ROOT / 'record.json', row)
    public = json.loads(STATUS.read_text()); public['current_sync'][KEY] = compact(row); write(STATUS, public)


def prior(native):
    proof = small.loan.validation.committed(STATUS, ('current_sync', small.KEY))
    require(proof['passed'] and proof['ownership_closed'] and not proof['fresh_faults'] and
        len(proof['results']) == 2 and all(p['passed'] and len(p['requests']) == 4 for p in proof['results']) and
        proof['results'][1]['lease_totals']['pairs'] == 2 and proof['native_build']['record'] == native['record'],
        'closed same-build bounded model handoff required')
    return dict(model_result_sha256=signature(proof), same_build=small.prerequisites(native))


def current(row, full=False):
    require(small.loan.validation.committed(STATUS, ('current_sync', KEY)) == compact(
        json.loads((ROOT / 'record.json').read_text())), 'commit exact prospective long plan/result')
    require(bench.sources(row['source_commit']) == row['sources'], 'benchmark/controller input changed')
    n, native = small.loan.native(); require(native == row['native_build'] and prior(native) == row['prior'], 'native/model proof changed')
    require(digest(row['recipe']['path']) == row['recipe']['sha256'], 'original recipe changed')
    probe, reuse = small.loan.current_probe()
    require(probe == row['native_validation_probe'] and reuse == row['runtime_reuse'], 'current-runtime proof changed')
    require(digest(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256'] and
        json.loads(Path(row['runtime_manifest']['path']).read_text()) == row['runtime_files'], 'runtime map changed')
    for p, sha in row['runtime_files'].items(): require(digest(p) == sha, 'runtime ELF changed: ' + p)
    require(small.loan.guards.driver_guard() == row['driver'] and
        Path('/proc/sys/kernel/random/boot_id').read_text().strip() == row['boot'], 'driver/boot changed')
    small.model.model_pins(row, full=full)
    manifest = bench.python_runtime.read_manifest(row['python_contract']); bench.python_runtime.validate_manifest(manifest)
    for plan in row['plans']:
        bench.python_runtime.require_environment(plan['environment'])
        require(bench.python_runtime.selection(plan['environment']) == row['python_selected'] and
            all(digest(Path(plan['root']) / p) == sha for p, sha in plan['generated'].items()), 'Python/config/native plan changed')
    small.loan.guards.no_model_owners([])
    return manifest


def prepare():
    require(not ROOT.exists(), 'fresh finite cohort only;no automatic retry')
    n, native = small.loan.native(); qualification = prior(native)
    probe, reuse = small.loan.current_probe()
    recipe = bench.ROOT / 'record.json'; old = json.loads(recipe.read_text())
    require(old['passed'] and old['ownership_closed'], 'old owned recipe was not closed')
    # Reuse only the recipe,not its retired private qualification chain. All
    # current qualification comes from the same90f native and new closed model.
    oldplan = old['plans'][0]; cfg = json.loads((Path(oldplan['root']) / 'config.json').read_text())
    oldnative = json.loads((B / 'upstream-features-146df3c4/record.json').read_text())
    a, b = oldnative['project']['source_sha256'], n['project']['source_sha256']
    recovered = {p: hashlib.sha256(raw).hexdigest() for p, raw in small.loan.source.blobs(P,
        oldnative['project']['revision'], a)}
    changed = sorted(p for p in a if a[p] != b.get(p))
    require(oldnative['project']['revision'] == '5ecb4435f8ac3bc5067554f368d0fa7424ec193f' and recovered == a and
        oldnative['toolchain'] == n['toolchain'] and oldnative['dependencies'] == n['dependencies'] and
        a.keys() == b.keys() and changed == sorted(['include/strata/core/cache_scratch_lease.hpp',
        'src/program/generate.cpp', 'tests/sycl/prefill_head_scope.cpp', 'tools/sycl/build_upstream_features.py']) and
        all(oldnative['binaries'][k]['sha256'] == v['sha256'] for k, v in n['binaries'].items() if k != 'strata'),
        'native math/fixture inputs differ beyond reviewed optional-owner policy')
    reference = json.loads((small.model.REFERENCE / 'record.json').read_text())
    revision = small.loan.source.git(P, 'rev-parse', 'HEAD').decode().strip()
    row = dict(stage='prepared', source_commit=revision, sources=bench.sources(revision),
        native_build=native, binary=native['binaries']['strata'], native_source_commit=native['source_commit'], prior=qualification,
        native_math_reuse=dict(reference_source=oldnative['project']['revision'], project_files=len(a), changed=changed,
            original_source_recovered=True, same_toolchain_and_dependencies=True, fixtures_same_bytes=5,
            scope='All math/kernels/prefill inputs equal the previous f6338192/384bank admission;'
                'policy change is independently host/CLI/five-fixture/4Kmodel checked. Not a new long-model result.'),
        native_validation_probe=probe, runtime_reuse=reuse,
        runtime_manifest=reference['runtime_manifest'],
        runtime_files=json.loads(Path(reference['runtime_manifest']['path']).read_text()),
        models=reference['model_assets'], models_sha256=small.model.fingerprint(reference['model_assets']),
        recipe=dict(path=str(recipe), sha256=digest(recipe), scope='configuration/environment template only'),
        python_contract=reference['python_contract'], python_selected=reference['python_selected'], template=reference['template'],
        boot=reference['boot'], driver=reference['driver'], plans=[], host=[], processes=[],
        passed=False, complete=False, ownership_closed=True,
        previous_prepare_failure=dict(source_commit='5f4c343860d5caf9d758e9a9f516d0f184116a54',
            error="KeyError('runtime_files')", plan_written=False, native_executed=False,
            reason='Reference stores the qualified runtime inventory in its pinned manifest,not an inline runtime_files key.'),
        budgets=dict(wall=1500, cpu=18000, rss_cap=104 << 30, text_cap=32 << 20, file_cap=64 << 20, total_cap=128 << 20),
        comparison=dict(question='Can checked prompt borrowing fund1536decode expert slots while admitting the8192/384bank long route?',
            order='Fresh owned256 first,then checked-borrowed1536;2exact32769-token thinking requests each. Original official warmups unchanged.'
                'Keep first/subsequent distinct;no independent-process repeats or speed adoption here.',
            constants='Same90f engine,65536INT8 resident context,8192chunk,spec4,no-token-graph,5workers,zero caches/adaptation,'
                'throughput BF16/native16/GDN16,mode10/384packed bank,head retirement1,absent-MTP DFlash2. Only expert residency and borrowed policy differ.'
                'No payload readback audit;parameter validation and stage diagnostics stay charged.',
            admission='Require actual8192chunk and actual expert slots256/1536 before warmup;no automatic chunk fallback.'
                'Positive free memory alone is not long-route admission;each actual request must also retain full GPU residuals/384bank/checked drain,'
                'matching optional-head reader and exact read/reuse/counter clocks. Borrowed long windows must show positive balanced mask/ready leases.'
                'Do not impose the old owned-buffer1536MiB headroom guess on the borrowed placement;actual allocations/routes decide.',
            acceptance='Official bench.sh exact-token mode;2complete thinking requests per arm with32769read,input32769,reuse0,1..24outputs,'
                'finite PP/prompt/clientTTFT and journals;actual provider/pre-main/arguments/environment/Python closure;normal native and benchmark0/reap/empty,'
                'unchanged inputs/assets and no fresh GPU fault. Stop first failure;no retry/reset/hash refresh/threshold relaxation.',
            decision='Qualify capacity and first/subsequent observed PP/TTFT only. Record phase differences separately;24output is not sustained TG.'
                'A decode-side capacity gain does not adopt a slower prefill path. Preserve552b/e1b/f633 and choose feature generation after this gate.'
                'Next eligible screen uses>=1Kthinking outputs and matched repeats;agent/task/physical262144/tails/RESTORE stay pending.',
            retention='Keep compact samples,actual route/lease/capacity identities and decisions;normal raw logs/JIT/journals retire after verification.'
                'No new native generation. At long-model decision select the retained feature generation,then recover original source and retire extra build.'))
    ROOT.mkdir(mode=0o700)
    for label, slots, borrowed in [('owned256', 256, False), ('borrowed1536', 1536, True)]:
        root = ROOT / label; root.mkdir(mode=0o700)
        for name in ('tmp', 'cache'): (root / name).mkdir(mode=0o700)
        env = {k: v.replace(oldplan['root'], str(root)) for k, v in oldplan['environment'].items()}
        env.update(STRATA_MEASUREMENT_ROOT=str(root), TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'),
            PYTHONPYCACHEPREFIX=str(root / 'cache/python'), STRATA_SYCL_CACHE_SCRATCH_LEASE=str(int(borrowed)),
            STRATA_SYCL_CACHE_LOAN_AUDIT='0', STRATA_PREFILL_RELEASE_MTP_EXPERTS='0')
        require(not env.get('LD_PRELOAD') and env['CUDA_MODULE_LOADING'] == 'EAGER' and
            env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1' and env['EnableDirectSubmission'] == '0' and
            env['UR_ENABLE_LAYERS'] == 'UR_LAYER_PARAMETER_VALIDATION' and env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == '0' and
            env['ZES_ENABLE_SYSMAN'] == '0', 'qualified immutable safety required')
        config = copy.deepcopy(cfg); config['log'] = str(root / 'engine.stderr')
        config['args'] = [x for x in config['args'] if x != '--no-prefill-borrow']
        config['args'][config['args'].index('--expert-cache') + 1] = str(slots)
        if not borrowed: config['args'].append('--no-prefill-borrow')
        write(root / 'config.json', config)
        write(root / 'plan.json', dict(binary=row['binary'], native_validation_probe=probe))
        with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        row['plans'].append(dict(label=label, root=str(root), slots=slots, borrowed=borrowed, environment=env, configuration=config,
            generated={p: digest(root / p) for p in ('config.json', 'plan.json')}, command=['/bin/sh', str(P / 'bench.sh'),
                '--config', str(root / 'config.json'), '--port', str(port), '--prefill-exact-tokens', str(READ),
                '--think', '--runs', str(RUNS), '--profile-request-gates', '--profile-startup-gate',
                '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root / 'bench')]))
    small.model.model_pins(row, full=True)
    save(row); print('Prepared owned256/borrowed1536 exact32Kx2;commit before host/native.', flush=True)


def route_config(path):
    sys.path.insert(0, str(P)); from serve import server
    cfg = json.loads(Path(path).read_text()); args = server.engine_args(cfg, exe=cfg['exe'], announce=False)
    def value(flag): return args[args.index(flag) + 1] if flag in args else None
    slots = value('--expert-cache')
    require(slots in ('256', '1536') and args.count('--no-prefill-borrow') == int(slots == '256') and
        all(value(k) == v for k, v in {'--prefill': '8192', '--max-context': '65536', '--kv-resident': '65536',
        '--spec': '4', '--draft': 'dflash2', '--draft-fallback': 'none', '--draft-conf': '0', '--suffix-draft': '0',
        '--pool-workers': '5', '--short-read': '0', '--prefill-strategy': 'throughput'}.items()) and
        Path(value('--draft-model')).resolve() == small.state.PACK.resolve() and '--mtp' not in args and
        cfg['family'] == 'qwen' and cfg['model_name'] == 'qwen3.8-flash-next-iq3_s-primary', 'actual named route differs')
    print(json.dumps(dict(args=args, backend=server.drafting.detect_hardware())), flush=True)


def host():
    row = json.loads((ROOT / 'record.json').read_text()); manifest = current(row)
    require(row['stage'] == 'prepared' and not row['host'], 'fresh host stage only')
    owner = small.loan.ownership.Owner(ROOT / 'host-owned')
    try:
        row['identity_host_gate'] = identity.identity_host_gate()
        allowed = {v['realpath'] for v in manifest['inventory']['files'].values()}
        env = row['plans'][0]['environment']
        for script in ('tools/strata_bench.py', 'serve/server.py'):
            entry, out, err = owner.run(Path(script).stem, [row['python_selected']['path'], '-B',
                str(bench.python_runtime.SOURCE), 'probe', str(P / script), row['template']['path']], env, P,
                wall=60, cpu=60, rss_cap=1 << 30, text_cap=1 << 20, file_cap=2 << 20, total_cap=8 << 20)
            require(small.loan.ownership.closed(entry) and entry['exit_code'] == 0 and not err.stat().st_size, 'current host probe failed')
            meta = json.loads(out.read_text()); require(meta['executable'] == row['python_selected']['path'], 'host interpreter differs')
            for name in [*meta['modules'], *meta['code_maps']]:
                p = Path(name).resolve()
                require(row['sources'].get(str(p.relative_to(P))) == digest(p) if p.is_relative_to(P) else str(p) in allowed,
                    'unfrozen host provider: ' + str(p))
            row['host'].append(dict(role=script, passed=True))
        for plan in row['plans']:
            entry, out, _ = owner.run(plan['label'], [row['python_selected']['path'], '-B', str(SELF), 'route-config',
                str(Path(plan['root']) / 'config.json')], plan['environment'], P,
                wall=60, cpu=60, rss_cap=1 << 30, text_cap=1 << 20, file_cap=2 << 20, total_cap=8 << 20)
            require(small.loan.ownership.closed(entry) and entry['exit_code'] == 0, 'named placement host probe failed')
            actual = json.loads(out.read_text()); require(actual['backend'][0] == 'sycl', 'actual backend changed')
            plan['native_args'] = actual['args']; row['host'].append(dict(role=plan['label'], passed=True))
        row['stage'] = 'ready'
    except BaseException as error: row.update(stage='failed', error=repr(error))
    finally:
        row['host_commands'] = owner.commands
        row['ownership_closed'] = owner.active is None and all(small.loan.ownership.closed(c) for c in owner.commands); save(row)
    require(row['stage'] == 'ready' and row['ownership_closed'], row.get('error', 'host closure failed'))
    print('Host/Python/named placement passed;commit before native.', flush=True)


def loans(text, borrowed):
    events = re.findall(r'strata cache loan: phase=(masked|ready) generation=(\d+) chunk=(\d+) first=(\d+) lent=(\d+) decode_ready=([01])', text)
    require('strata loan audit:' not in text and text.count('strata cache loan:') == len(events), 'unexpected/unparsed audit')
    if not borrowed:
        require(not events, 'owned control borrowed')
        return dict(pairs=0)
    require(len(events) >= 2 and len(events) % 2 == 0, 'missing/open long lease')
    generations, lent = [], []
    for a, b in zip(events[::2], events[1::2]):
        require(a[0] == 'masked' and b[0] == 'ready' and a[1:5] == b[1:5] and a[5] == '0' and b[5] == '1' and
            int(a[2]) == 8192 and int(a[3]) >= 128 and int(a[4]) > 0 and
            (not generations or int(a[1]) > generations[-1]), 'long lease state/order/extent differs')
        generations.append(int(a[1])); lent.append(int(a[4]))
    return dict(pairs=len(generations), generations=generations, lent_rows=lent, checked_mask_ready=True)


def run(index):
    row = json.loads((ROOT / 'record.json').read_text()); manifest = current(row, full=index == 0)
    require(row['stage'] in ('ready', 'running') and len(row['processes']) == index < len(row['plans']) and
        all(p['passed'] for p in row['processes']), 'fresh frozen arm order only')
    if index == 0: row['original_plan_commit'] = small.loan.source.git(P, 'rev-parse', 'HEAD').decode().strip()
    plan = row['plans'][index]; root = Path(plan['root']); env = plan['environment']
    require(not (root / 'execution.json').exists(), 'no duplicate process')
    cursor = next(s.split(': ', 1)[1] for s in small.model.mask.ring.device.kernel_log('-n', '0', '--show-cursor').splitlines()
        if s.startswith('-- cursor: '))
    owner, windows = None, []
    observed = dict(label=plan['label'], passed=False, complete=False, model_executed=False)
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
    def persist(): write(root / 'execution.json', dict(observed, active=bool(owner and owner.active), commands=owner.commands if owner else []))
    def live(entry):
        claim = json.loads((root / 'engine-start.json').read_text()); pid = claim['native']['pid']
        require(identity.identity_matches(small.loan.ownership.identity(pid), claim['native'], entry['pid']) and
            str(Path('/proc', str(pid), 'exe').resolve()) == row['binary']['path'] and
            claim['args'] == ['--serve', *plan['native_args']], 'native identity/arguments differ')
        expected = small.loan.engine_once.native_environment(row, env)
        actual = dict(v.split(b'=', 1) for v in Path('/proc', str(pid), 'environ').read_bytes().split(b'\0') if b'=' in v)
        require(all(actual.get(k.encode()) == v.encode() for k, v in expected.items()) and
            set(actual) - {k.encode() for k in expected} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'}, 'native environment differs')
        observed.update(small.model.runtime_evidence(dict(row, probe=row['native_validation_probe'], environment=env),
            pid, (root / 'engine.stderr').read_text(errors='replace')))
        bench.python_runtime.check_running(entry['pid'], P / 'tools/strata_bench.py', manifest['selected'], env,
            {v['realpath'] for v in manifest['inventory']['files'].values()}, plan['command'][2:])
        observed.update(model_executed=True, native_identity=claim['native'])
    def gate(io, entry, stage, phase, run_id):
        while True:
            line = io.line(1000)
            if line.startswith('STRATA_PROFILE_GATE '): break
        require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) == dict(stage=stage, phase=phase, run=run_id), 'unexpected benchmark gate')
        delta = [a - b for a, b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()), initial)]
        require(min(delta) >= 0 and abs(delta[0] - delta[1]) <= 50000000, 'clock/suspend discontinuity')
        live(entry); warning = root / 'validation.log'
        require(not warning.exists() or not warning.stat().st_size, 'validation warning/error')
    def interact(io, entry):
        gate(io, entry, 'ready', 'startup', -1)
        text = (root / 'engine.stderr').read_text(errors='replace')
        chunks = re.findall(r'^strata prefill: FP16 MoE compact .*?chunk=(\d+),', text, re.M)
        free = re.findall(r'^strata serve: (\d+) MiB of VRAM free with everything loaded$', text, re.M)
        require(chunks == ['8192'] and len(free) == 1 and int(free[0]) > 0 and "chunk's prompt buffers need" not in text,
            'startup8192 capacity/fallback differs')
        slots = re.findall(r'^strata generate: expert cache (\d+) slots, [0-9.]+ GiB of VRAM; policy is$', text, re.M)
        require(slots == [str(plan['slots'])], 'actual native expert residency differs')
        observed['startup_admission'] = dict(chunk=8192, expert_slots=plan['slots'], free_MiB=int(free[0]), borrowed=plan['borrowed'])
        print(plan['label'] + ' startup8192/slots admitted', flush=True); io.send(b'continue\n')
        for request in range(RUNS):
            offsets = {}
            for stage in ('begin', 'end'):
                gate(io, entry, stage, 'prefill-only', request)
                offsets[stage] = (root / 'engine.stderr').stat().st_size
                print(f'{plan["label"]} request{request} {stage}', flush=True); io.send(b'continue\n')
            windows.append(offsets)
        io.proc.stdin.close(); deadline = time.monotonic() + 180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic() < deadline, 'native/benchmark closure deadline')
            if select.select([io.proc.stdout], [], [], .02)[0]: io.read(); io.pending.clear()
        io.exit(1)
    try:
        require(not small.state.faults(cursor), 'fault before launch')
        owner = small.loan.ownership.Owner(root / 'owned', persist)
        entry, _, _ = owner.run('bench', plan['command'], env, P, interaction=interact, **row['budgets'])
        require(small.loan.ownership.closed(entry) and entry['exit_code'] == 0, 'native/benchmark closure failed')
        outputs = list((root / 'bench').glob('*/results.json')); require(len(outputs) == 1, 'one result required')
        result = json.loads(outputs[0].read_text()); cohort = result['results']['prefill_only']; samples = cohort['requests']; counts = cohort['count_preflight']
        require(result['complete'] and not result['results']['failures'] and len(samples) == len(counts) == len(windows) == RUNS, 'incomplete cohort')
        sys.path.insert(0, str(P / 'tools')); import strata_bench as client
        require(client.verify_request_journal(outputs[0].parent)['valid'], 'invalid request journal')
        end = bench.evidence.native_end(root, row, dict(config=dict(args=plan['native_args'])), row['boot'], entry['pid'])
        raw = (root / 'engine.stderr').read_bytes(); text = raw.decode()
        require(not any(s in text for s in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called', 'teardown uncertain')), 'native runtime error')
        require(('prompt path borrows' in text) == plan['borrowed'] and
            text.count('strata serve: the prompt path allocates its own buffers ') == int(not plan['borrowed']), 'actual buffer ownership differs')
        routes = []
        for sample, count, window in zip(samples, counts, windows):
            require(all(sample[k] is True for k in ('ok', 'eligible', 'stream_complete', 'thinking_requested', 'reasoning_observed')) and
                sample['actual_prompt_tokens'] == sample['actual_read_tokens'] == count['counted_input_tokens'] == READ and
                sample['actual_reused_tokens'] == 0 and sample['request_sha256'] == count['body_sha256'] and
                1 <= sample['generated_tokens'] <= 24 and sample['prompt_tps_source'] == 'engine' and
                all(type(sample[k]) in (int, float) and math.isfinite(sample[k]) and sample[k] > 0 for k in
                    ('prompt_ms', 'prompt_tps', 'ttft_s', 'total_s')) and sample['ttft_s'] <= sample['total_s'], 'request/thinking/clock differs')
            part = raw[window['begin']:window['end']].decode()
            route = bench.request_route(part, 'dflash2', sample); route['loan'] = loans(part, plan['borrowed']); routes.append(route)
        if index == len(row['plans']) - 1: small.model.model_pins(row, full=True)
        observed.update(passed=True, complete=True, native_exit_code=end['exit_code'], samples=samples, count_preflight=counts,
            routes=routes, speed_adoption_qualified=False)
    except BaseException as error: observed['error'] = repr(error)
    finally:
        observed['commands'] = owner.commands if owner else []
        observed['ownership_closed'] = owner is None or owner.active is None and all(small.loan.ownership.closed(c) for c in owner.commands)
        try:
            observed['fresh_faults'] = small.state.faults(cursor); require(not observed['fresh_faults'], 'fresh GPU fault'); current(row)
        except BaseException as error: observed.update(passed=False, finalization_error=repr(error))
        observed['passed'] = bool(observed['passed'] and observed['ownership_closed'])
        row['processes'].append(observed); row.update(stage='running' if observed['passed'] else 'failed', ownership_closed=observed['ownership_closed'])
        if len(row['processes']) == len(row['plans']) and observed['passed']: row.update(stage='closed', passed=True, complete=True)
        save(row)
    require(observed['passed'], observed.get('error', observed.get('finalization_error', 'long admission failed')))
    print(plan['label'] + ' long model closed;commit before next arm.', flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'route-config': route_config(sys.argv[2])
    else:
        with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); small.loan.guards.no_model_owners([])
            if sys.argv[1:] == ['prepare']: prepare()
            elif sys.argv[1:] == ['host']: host()
            elif len(sys.argv) == 3 and sys.argv[1] == 'run': run(int(sys.argv[2]))
            else: raise RuntimeError('use prepare|host|run INDEX|route-config FILE')
