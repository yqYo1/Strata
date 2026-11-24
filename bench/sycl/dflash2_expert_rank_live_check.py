"""Validate a held-out-qualified static DFlash2 expert layout on the retained engine."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_expert_rank_forecast as forecast

capture, bank, basis, require = forecast.capture, forecast.bank, forecast.basis, forecast.require
confidence, leaves, model = capture.confidence, capture.leaves, capture.model
census = capture.census
KEY = 'dflash2_expert_rank_live_check_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ORDER = ('stock', 'trained')
FLAG = '--expert-profile'
PROMPTS = (82, 257, 903, 1068, 11793, 82)
ROUTES = (('cache-window', 8192, 1256), ('chunked', 256, 1141),
          ('chunked', 1024, 1053), ('chunked', 1280, 1023),
          ('layer-major', 8192, 132), ('cache-window', 8192, 1256))
OUTPUT = 32


def inputs(revision):
    scored = basis.validation.committed(bank.RESULT, (forecast.KEY,))
    closed = basis.validation.committed(bank.RESULT, (capture.experiment('holdout').KEY,))
    require(scored['stage'] == 'scored' and scored['eligible_for_live_comparison']
            and closed['stage'] == 'closed' and closed['passed'] and closed['complete']
            and closed['ownership_closed'] and not closed['active']
            and scored['sources'] == forecast.verify_sources(scored['source_commit']),
            'require the committed successful held-out placement gate')
    process, = closed['processes']
    require(process['native_exit_code'] == 0 and not process['fresh_faults']
            and scored['holdout']['request_sha256'] == closed['routing_workload']['request_sha256'],
            'held-out request or owner evidence differs')
    base = capture.baseline('train')
    require(all(base[k] == closed[k] for k in ('binary', 'native_source_commit', 'native_build_sha256',
                                              'boot', 'driver', 'runtime_manifest', 'routing_metadata')),
            'require the original qualified executable, runtime and stock profile')
    stock_path = Path(base['routing_metadata']['profile'])
    trained_path = forecast.PROFILE
    require(scored['profile']['path'] == str(trained_path)
            and bank.sha(trained_path) == scored['profile']['sha256']
            and bank.sha(stock_path) == base['routing_metadata']['profile_sha256']
            and bank.sha(base['routing_metadata']['manifest']) == scored['geometry']['manifest_sha256'],
            'profile or expert manifest identity differs')
    stock = forecast.make_profile.read_profile(stock_path)
    trained = forecast.apply_delta(stock, scored['profile']['profile_delta'])
    require(forecast.make_profile.read_profile(trained_path) == trained,
            'candidate bytes do not match the ordered committed delta')
    paths = dict(zip(ORDER, (stock_path, trained_path), strict=True))
    membership = {arm: census.expected_membership(path, forecast.CACHE_SLOTS) for arm, path in paths.items()}
    require(membership['stock'] != membership['trained'], 'require genuinely different resident experts')
    pins = capture.experiment('train').pins(revision)
    for file in (THIS, *forecast.SOURCES):
        require((bank.P / file).read_bytes() == basis.validation.git('show', revision + ':' + file),
                'changed profile trial source: ' + file)
        pins[file] = bank.sha(bank.P / file)
    cases, configurations, environments = {}, {}, {}
    requests = None
    for arm, path in paths.items():
        root = ROOT / arm
        choice = dict(root=str(root), arm='threshold')
        baseline = confidence.configuration(base, choice)
        args = list(baseline['args'])
        require(args.count(FLAG) == 1 and args[args.index(FLAG) + 1] == str(stock_path),
                'expected the measured stock profile argument')
        args[args.index(FLAG) + 1] = str(path)
        config = dict(baseline, args=args)
        require(model.server.engine_args(config, exe=config['exe'], announce=False) == args,
                'production resolver changed more than the profile path')
        env = capture.environment(base, choice)
        require(env['EnableDirectSubmission'] == '0' and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1',
                'qualified runtime safety settings required')
        env = model.native_environment(base, dict(env, STRATA_STATE_HASH='1', STRATA_SYCL_CACHE_LOAN_AUDIT='1'))
        short = model.request_ids(Path(config['tokenizer']))['ids']
        require(len(short) == PROMPTS[0], 'original short body changed')
        workload = [([i % 4096 for i in range(n - len(short))] + short, OUTPUT) for n in PROMPTS]
        if requests is None:
            requests = workload
        require(requests == workload, 'wire requests differ by profile')
        cases[arm] = dict(command=[base['binary']['path'], '--serve', *args],
                          environment_sha256=bank.signature(env), profile=str(path), profile_sha256=bank.sha(path))
        configurations[arm], environments[arm] = config, env
    expected_args = [str(paths['stock']) if value == str(paths['trained']) else value
                     for value in cases['trained']['command']]
    stock_root, trained_root = (str(ROOT / arm) for arm in ORDER)
    expected_env = {key: value.replace(trained_root, stock_root)
                    for key, value in environments['trained'].items()}
    require(cases['stock']['command'] == expected_args and environments['stock'] == expected_env,
            'the rank path must be the only effective difference')
    plan = dict(source_commit=revision, sources=dict(files=len(pins), sha256=bank.signature(pins)),
                native_source_commit=base['native_source_commit'], native_build_sha256=base['native_build_sha256'],
                binary=base['binary'], boot=base['boot'], driver=base['driver'],
                runtime_manifest=base['runtime_manifest'], validation=base['native_validation_probe'],
                drafter_assets=base['dflash2_assets'], cases=cases, order=list(ORDER), limits=model.LIMITS,
                profile_forecast_sha256=bank.signature(scored), heldout_sha256=bank.signature(closed),
                manifest=base['routing_metadata']['manifest'],
                manifest_sha256=base['routing_metadata']['manifest_sha256'],
                routes=[list(route) for route in ROUTES], wires=[
                    dict(prompt=len(ids), output=budget, sha256=model.restore.signature_bytes(model.wire(ids, budget)))
                    for ids, budget in requests],
                purpose='Trace native cache-window, short chunk and long return with each fixed profile. '
                        'Require exact cache membership, payload and request-reset evidence before clean timing.')
    return plan, base, configurations, environments, requests, membership


def one(text, pattern):
    found = list(re.finditer(pattern, text, re.M))
    require(len(found) == 1, 'missing or repeated native evidence: ' + pattern)
    return found[0]


def check_route(frame, answer, route, chunk, prefix):
    read = answer['prompt'] - 1
    actual = one(frame, r'^strata prefill policy: requested=throughput route=([a-z-]+) tokens=(\d+) chunk=(\d+)$')
    require((actual[1], int(actual[2]), int(actual[3])) == (route, read, chunk), 'prefill route or shape differs')
    taps, = re.findall(r'^strata DFlash2: eager, prompt_tail_rows=(\d+) cold_prefix=0 vram_bytes=[1-9]\d*$', frame, re.M)
    require(int(taps) == min(read, 2046), 'DFlash2 tail extent differs')
    first_window = one(frame, r'^strata trace: window ' + str(read) + r' 1$')
    draft = frame.index('strata DFlash2: eager,')
    require(actual.end() < draft < first_window.start(), 'prefill, draft activation or verifier order differs')
    if route == 'cache-window':
        require('strata cache loan:' not in frame and 'strata loan audit:' not in frame,
                'cache-window prefill unexpectedly used a loan')
    else:
        mask = one(frame, r'^strata cache loan: phase=masked generation=(\d+) chunk=(\d+) first=(\d+) lent=(\d+) decode_ready=0$')
        ready = one(frame, r'^strata cache loan: phase=ready generation=(\d+) chunk=(\d+) first=(\d+) lent=(\d+) decode_ready=1$')
        lent = forecast.CACHE_SLOTS - prefix
        require(mask.groups() == ready.groups() and tuple(map(int, ready.groups()[1:])) == (chunk, prefix, lent),
                'loan coverage or returned generation differs')
        lend_table, refill_table = [one(frame, '^strata loan audit: phase=' + phase +
            r' table_entries=24576 table_match=1 free_bytes=\d+ total_bytes=\d+$') for phase in ('lend', 'refill')]
        payload = one(frame, r'^strata loan audit: phase=payload rows=1256 bytes=2575206400 '
                      + f'lent={lent} same_slots=1 payload_match=1$')
        require(lend_table.start() < mask.start() < actual.start() < payload.start() < refill_table.start()
                < ready.start() < draft, 'cache reads, audit and refill publication order differs')
    return model.restore.require_store(frame, {'direct': False}, {'read': answer['prompt']})


def inspect(text, plan, arm, pid, requests, membership):
    proof = basis.validation.preload_evidence(text, pid)
    path = re.escape(plan['cases'][arm]['profile'])
    one(text, f'^strata generate: profile {path}: 24576 ranked pairs, built for 24576 slots$')
    require(proof['loader'] == plan['validation']['loader']['path']
            and text.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1
            and not re.search(r'(?m)^strata mtp: .*loaded|UR_RESULT_ERROR_|ZE_RESULT_ERROR_|teardown uncertain|cleanup failed', text)
            and tuple(a['prompt'] for a in requests) == PROMPTS
            and all(1 <= a['generated'] <= OUTPUT and 0 <= a['accepted'] <= a['offered'] for a in requests)
            and all(requests[i]['generated'] == OUTPUT and requests[i]['offered'] > 0 for i in (0, 4, 5)),
            'native runtime, drafter or wire outcome differs')
    markers = list(re.finditer(r'^strata trace: request (\d+) 0$', text, re.M))
    require(tuple(int(m[1]) for m in markers) == PROMPTS, 'missing native request boundaries')
    ends = [m.start() for m in markers[1:]] + [len(text)]
    frames = [text[m.start():end] for m, end in zip(markers, ends, strict=True)]
    stores, fingerprints = [], []
    for frame, request, route in zip(frames, requests, ROUTES, strict=True):
        stores.append(check_route(frame, request, *route))
        state = dict(re.findall(r'(\w+)=([0-9a-f,-]+)', one(frame, r'^strata serve: STATE_HASH L=.*$')[0]))
        fields = model.restore.ACTIVE_STATE
        require(set(fields) <= state.keys() and int(state['L']) == request['prompt'],
                'target state marker differs')
        fingerprints.append(bank.signature({k: state[k] for k in fields}))
    require(fingerprints[0] == fingerprints[-1] and requests[0] == requests[-1],
            'repeated short reset differs within the same profile')
    samples = [dict(prompt_tokens=a['prompt'], read_tokens=a['prompt'], reused_tokens=0,
                    generated_tokens=a['generated'], drafts_accepted=a['accepted'], drafts_offered=a['offered'])
               for a in requests]
    native, _ = leaves.leaf_reader.evidence(text, samples, 4, False)
    bindings, _ = capture.census_inputs(text, samples, native)
    require(bank.sha(plan['manifest']) == plan['manifest_sha256']
            and bank.sha(plan['cases'][arm]['profile']) == plan['cases'][arm]['profile_sha256'],
            'live expert formats or profile changed')
    observed = census.extract(text, bindings, census.manifest_formats(plan['manifest']), membership)
    require(stores[-2]['gpu_rows'] == PROMPTS[-2] - 1 and stores[-2]['host_rows'] == 0,
            'near long prefill did not take the all-device BF16 route')
    keys = ('windows', 'rows', 'cpu_jobs', 'cpu_entries', 'logical_job_blob_bytes', 'membership_sha256')
    return dict(validation=proof, fingerprints=fingerprints, stores=stores,
                work=[{k: result[k] for k in keys} for result in observed])


def save(row):
    bank.stage.write(ROOT / 'record.json', row)
    result = bank.previous.read(bank.RESULT)
    result[KEY] = row
    bank.stage.write(bank.RESULT, result)


def main(action):
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *ORDER),
            'use the pinned Python with prepare, stock, or trained')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare':
            require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT),
                    'inspect the existing diagnostic first')
            revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
            plan, _, _, _, _, _ = inputs(revision)
            for arm in ORDER:
                for name in ('cache', 'tmp'):
                    (ROOT / arm / name).mkdir(mode=0o700, parents=True)
            save(dict(stage='prepared', active=False, complete=False, passed=False,
                      ownership_closed=True, plan=plan, outcomes={}))
            return
        row = bank.previous.read(ROOT / 'record.json')
        plan, base, configs, envs, requests, memberships = inputs(row['plan']['source_commit'])
        require(row == basis.validation.committed(bank.RESULT, (KEY,)) and row['plan'] == plan
                and row['stage'] == 'prepared' and row['ownership_closed'] and not row['active']
                and tuple(row['outcomes']) == ORDER[:ORDER.index(action)]
                and all(r['passed'] for r in row['outcomes'].values()),
                'commit the unused plan or previous successful arm before launch')
        root = ROOT / action
        require(not (root / 'owned').exists() and not (root / 'execution.json').exists(),
                'inspect the existing owner before retrying')
        leaves.speed.verify_model_assets(base, configs[action])

        def verify(text, pid, answers):
            result = inspect(text, plan, action, pid, answers, memberships[action])
            leaves.speed.verify_model_assets(base, configs[action])
            return result

        outcome = confidence.width_check.run_case(plan, root, action, envs[action], requests, verify)
        row['outcomes'][action] = outcome
        passed = all(r['passed'] for r in row['outcomes'].values())
        closed = outcome['ownership_closed']
        done = passed and action == ORDER[-1]
        row.update(stage='closed' if done else 'prepared' if passed else 'failed',
                   complete=done, passed=done, ownership_closed=closed, active=not closed)
        save(row)
        print({k: outcome.get(k) for k in ('passed', 'native_exit_code', 'fresh_faults', 'error')}, flush=True)
        require(outcome['passed'], 'close and preserve the original diagnostic before clean timing')

if __name__ == '__main__':
    action, = sys.argv[1:]
    main(action)
