"""Qualify global decode placement and its long-prefill cache restoration."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_primary_restore as primary
import dflash2_primary_retained as retained
import dflash2_decode_layer_quota_replay as quota
import cpu_work_census as census
import decode_footer as footer
from make_profile import write_profile

bank, basis, require = primary.bank, primary.basis, primary.require
KEY = 'dflash2_decode_stock_global_restore_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
REFERENCE = '3d75f7e9126c74077f2bd1016981e19b9459db9f'
RANKED_KEY = 'dflash2_decode_ranked_global_restore_v1'
RANKED_ROOT = bank.B / RANKED_KEY.replace('_', '-')
REPLAY_RESULT = '6a36f262b'
RANK = '07362e5b275d174eba7a1dcb4511fca13f26825e4c328fde576f2b9ac9b04547'
MANIFEST = 'd9ac2dfa3ee63c55c9c6a6db26f72da0cec0ee41733f007e5cbbbba71617aa5d'
ORDER = ('host', 'short-stock', 'short-global', 'near-stock', 'near-global')
RANKED_ORDER = ('host', 'short-within', 'short-cross', 'near-within', 'near-cross')
RANKED_PROFILES = {
    'within': ('within_layer', 'f731c6a0fd0464b8648bd6520cc29a2bef8c213eb355113f659d5b59630c9851',
               2_013_072, 4_086_496_000_000),
    'cross': ('across_equal_size_layers', '2863acb7b313d1e4a53a88f630842d2a188e41611e4fbeb39af55cb8aab16749',
              1_967_341, 3_999_896_140_800),
}
STOCK = bank.P / quota.PROFILE
GLOBAL = ROOT / 'stock-global.bin'
PLAN = dict(order=list(ORDER), reference=REFERENCE, rank_sha256=RANK,
            scope='Same retained engine, same slot identities, same widths and protected prefix. '
            'Change only admission from per-layer to global. Reuse both primary copy/RESTORE workloads.',
            gates='Bind all 48 measured membership maps to the stock bytes, check all occupied payloads and actual '
            'loan/refill, and compare short handoff, output, active state, expert work and verified RAM restore. '
            'Diagnostic only. Stop on the first failed check.')
RANKED_PLAN = dict(order=list(RANKED_ORDER), stock_gate=KEY, replay=REPLAY_RESULT,
                   scope='Keep the qualified global loader, slot widths, protected prefix, prefill and '
                         'restore protocols. Qualify both reranked decode placements independently.',
                   gates='Reproduce the fixed holdout ranking; bind its actual membership, loaned slots '
                         'and restored bytes. Verify each arm against its own saved state, preserve '
                         'prefill and context extents, and record output and work for each arm. '
                         'Diagnostic only. Stop on the first failed check.')


def geometry(row):
    args = row['draft_arguments']
    manifest = Path(args[args.index('--pack') + 1]) / 'native_experts.txt'
    require(bank.sha(manifest) == MANIFEST, 'the captured packed model manifest changed')
    stock = census.expected_membership(STOCK)
    formats = census.manifest_formats(manifest)
    order, slots, sizes, _ = quota.stock_geometry(STOCK, stock, formats)
    ranking, requested = quota.global_profile(slots, order, sizes)
    require(requested == 968 and quota.rank_sha256(ranking) == RANK, 'the selected stock global ranking changed')
    return ranking, stock, formats


def ranked_profiles():
    result, profiles = quota.profile_study()
    require(result['advance_to_admission'], 'require the original held-out improvement')
    selected = {}
    for arm, (label, digest, jobs, packed_bytes) in RANKED_PROFILES.items():
        case = result['cases'][label]
        require(case['ranking_sha256'] == digest and case['requested_expert_cache'] == 968
                and case['effective_slots'] == quota.EXPECTED_SLOTS
                and (case['holdout']['cpu_jobs'], case['holdout']['logical_job_blob_bytes'])
                == (jobs, packed_bytes), 'the original held-out ranking or results changed')
        selected[arm] = profiles[label]
    return selected


def check_ranked_profile(ranking, stock_ranking, formats, arm):
    slots = quota.EXPECTED_SLOTS
    size = [quota.replay.aligned(f[2]) for f in formats]
    require(quota.rank_sha256(ranking) == RANKED_PROFILES[arm][1]
            and ranking[:quota.PROTECTED] == stock_ranking[:quota.PROTECTED]
            and [size[layer] for layer, _ in ranking[:slots]] == [size[layer] for layer, _ in stock_ranking[:slots]]
            and quota.global_profile(ranking[:slots], quota.read_profile(STOCK), size) == (ranking, 968),
            'ranked file changed the protected prefix, slot geometry or loader extent')


def check_placement(log, requests, membership, formats):
    loans = [tuple(map(int, fields)) for fields in re.findall(
        r'^strata loan audit: phase=layout chunk=(\d+) first=(\d+) slots=(\d+) '
        r'bytes=(\d+) need=(\d+) lent=(\d+) mask_match=1$', log, re.M)]
    lent = quota.EXPECTED_SLOTS - quota.PROTECTED
    phases = re.findall(r'^strata loan audit: phase=(\w+)', log, re.M)
    require(len(loans) == 2 and phases == ['layout', 'lend', 'payload', 'refill'] * 2
            and all(l[:4] == (8192, quota.PROTECTED, quota.EXPECTED_SLOTS, quota.EXPECTED_TAIL)
                    and 0 < l[4] <= quota.EXPECTED_TAIL and l[5] == lent for l in loans)
            and log.splitlines().count(f'strata loan audit: phase=payload rows={quota.EXPECTED_SLOTS} '
                                      f'bytes={quota.EXPECTED_BYTES} lent={lent} same_slots=1 payload_match=1') == 2,
            'the actual occupied loan/refill or restored bytes changed')
    hdr = re.findall(footer.HEADER, log, re.M)
    clocks = re.findall(footer.TIMING, log, re.M)
    copies = re.findall(footer.COPY, log, re.M)
    windows = [tuple(map(int, fields)) for fields in re.findall(r'^strata trace: window (\d+) (\d+)$', log, re.M)]
    require(windows == [(r['prompt'] - 1, 1) for r in requests], 'verifier positions changed after the handoff')
    bindings = []
    for req, h, clock, copy in zip(requests, hdr, clocks, copies, strict=True):
        ms = dict(zip(footer.LABELS, map(float, clock[1:]), strict=True))
        require(tuple(map(int, h)) == (req['prompt'], req['reused'], req['read'], 1, 0, 0)
                and int(clock[0]) == 1 and ms['average_T'] == ms['tokens_per_window'] == 1
                and ms['PCIe'] == 0 and not any(map(int, copy[:6])), 'actual single-row expert work changed')
        bindings.append(dict(prompt_tokens=req['prompt'], windows=1, rows=1, cpu_jobs=int(copy[-2]),
                             cpu_entries=int(copy[-1]), pool_ms=ms['CPU_pool_run']))
    observed = census.extract(log, bindings, formats, membership)
    keys = ('request', 'membership_sha256', 'cpu_jobs', 'cpu_entries', 'vram_entries', 'logical_job_blob_bytes')
    return dict(loans=[list(item) for item in loans], decode_placement=[{k: item[k] for k in keys} for item in observed])


def experiment(*, ranked=False):
    ctx = primary.experiment()
    base_inspect, base_environment = ctx.inspect, ctx.environment
    root, key, order, plan = (RANKED_ROOT, RANKED_KEY, RANKED_ORDER, RANKED_PLAN) if ranked else (ROOT, KEY, ORDER, PLAN)
    paths = {arm: root / (arm + '.bin') for arm in RANKED_PROFILES} if ranked else {'global': GLOBAL}
    ctx.ROOT, ctx.KEY, ctx.ORDER = root, key, order

    def inputs(revision):
        prior = bank.previous.historic(REFERENCE, primary.KEY)
        actual = retained.baseline()
        require(retained.closed(prior, 'closed')
                and all(prior[k] == actual[k] for k in ('binary', 'native_source_commit', 'boot', 'driver',
                                                        'native_validation_probe'))
                and prior['runtime_manifest']['sha256'] == actual['runtime_manifest']['sha256']
                and prior['model_assets_sha256'] == bank.signature(actual['model_assets'])
                and prior['environment_template_sha256'] == bank.signature(actual['environment_template'])
                and prior['drafter_assets'] == actual['dflash2_assets'],
                'require the exact primary restore receipt and retained executable')
        files = (THIS, quota.THIS, 'bench/dflash2_decode_residency_replay.py', quota.PROFILE,
                 primary.THIS, primary.physical.THIS, primary.physical.checks.THIS, primary.generation.THIS,
                 retained.THIS, retained.speed.THIS, 'tools/make_profile.py')
        pins = ctx.sources(revision)
        for name in files:
            path = bank.P / name
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit the exact restore input: ' + name)
            pins[name] = bank.sha(path)
        fields = ('configuration', 'native_validation_probe',
                  'boot', 'driver', 'vocab_size', 'drafter_assets', 'ordinary_control',
                  'python_selected', 'native_source_commit', 'native_build_sha256', 'binary', 'draft_arguments')
        row = {k: prior[k] for k in fields}
        row.update({k: actual[k] for k in ('environment_template', 'model_assets', 'runtime_manifest')})
        if ranked:
            stock = basis.validation.committed(bank.RESULT).get(KEY)
            require(stock is not None and retained.closed(stock, 'closed') and stock['binary'] == row['binary']
                    and stock['native_source_commit'] == row['native_source_commit']
                    and all(stock['outcomes'][name]['passed'] for name in ORDER),
                    'complete the identical stock global control on this build')
            row['stock_control_sha256'] = bank.signature(stock)
        return dict(row, source_commit=revision, sources=pins, plan=plan), prior

    def environment(row, case):
        return dict(base_environment(row, case), STRATA_CPU_WORK_CENSUS='1', STRATA_DECODE_TIMING='1')

    def plans(row, prior):
        previous = {case['name']: case for case in prior['plans']}
        result = []
        for name in order[1:]:
            extent, arm = name.split('-')
            old = previous[extent + '-copy']
            require(ctx.input_plan(old['near']) == old['requests'] and old['command'][0] == row['binary']['path']
                    and bank.signature(base_environment(row, old)) == old['environment_sha256'],
                    'the original restore workload or provider envelope changed')
            command = list(old['command'])
            profile_index = command.index('--expert-profile') + 1
            cache_index = command.index('--expert-cache') + 1
            require(command[profile_index] == str(STOCK) and command[cache_index] == '1256'
                    and command.count('--expert-cache-per-layer') == 1, 'expected the original stock placement')
            if arm in paths:
                command[profile_index] = str(paths[arm])
                command[cache_index] = '968'
                command.remove('--expert-cache-per-layer')
            case = dict(old, name=name, arm=arm, root=str(root / name), command=command)
            case['environment_sha256'] = bank.signature(environment(row, case))
            result.append(case)
        return result

    def inspect(log, row, case, requests, pid):
        extent, arm = case['name'].split('-')
        out = base_inspect(log, row, dict(case, name=extent + '-copy'), requests, pid)
        prefix = 'R4.2g PER-LAYER:'
        path = paths.get(arm, STOCK)
        require(log.splitlines().count(f'strata generate: profile {path}: 24576 ranked pairs, built for 24576 slots') == 1
                and log.count('pre-filled 1256 of 1256 slots from the profile; slot 0 verified') == 1
                and (prefix in log) == (arm == 'stock')
                and (arm == 'stock' or 'PROFILE, ranked by routing frequency, no eviction.' in log),
                'the actual loader or allocated stock profile changed')
        _, stock, formats = geometry(row)
        membership = quota.masks(quota.read_profile(path)[:quota.EXPECTED_SLOTS]) if ranked else stock
        out.update(check_placement(log, requests, membership, formats))
        if arm == 'global':
            control = row['outcomes'][extent + '-stock']
            require(all(out[k] == control[k] for k in ('restore_payload', 'loans', 'decode_placement', 'requests')),
                    'global loading changed fixed-work behavior, state or payload')
        return out

    def prepare():
        require(not root.exists() and key not in basis.validation.committed(bank.RESULT), 'fresh placement control required')
        revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
        row, prior = inputs(revision)
        ranking, _, formats = geometry(row)
        cases = plans(row, prior)
        selected = ranked_profiles() if ranked else {'global': ranking}
        if ranked:
            for arm, profile in selected.items():
                check_ranked_profile(profile, ranking, formats, arm)
        root.mkdir(mode=0o700)
        for arm, profile in selected.items():
            write_profile(paths[arm], profile)
        for case in cases:
            for folder in ('tmp', 'cache'):
                (Path(case['root']) / folder).mkdir(mode=0o700, parents=True)
        (root / 'host').mkdir(mode=0o700)
        row.update(stage='prepared', outcomes={}, plans=cases,
                   complete=False, passed=False, active=False, ownership_closed=True, speed_eligible=False)
        if ranked:
            row['ranked_profile_sha256'] = {arm: bank.sha(path) for arm, path in paths.items()}
        else:
            row['profile_sha256'] = bank.sha(GLOBAL)
        ctx.save(row)

    def current(row):
        require(ctx.compact(row) == basis.validation.committed(bank.RESULT, (key,)), 'commit the preceding control')
        expected, prior = inputs(row['source_commit'])
        ranking, stock, formats = geometry(row)
        require(all(row[k] == v for k, v in expected.items()) and row['plans'] == plans(row, prior),
                'frozen control or plans changed')
        if ranked:
            require({arm: bank.sha(path) for arm, path in paths.items()} == row['ranked_profile_sha256'],
                    'frozen ranked profile bytes changed')
            for arm, path in paths.items():
                check_ranked_profile(quota.read_profile(path), ranking, formats, arm)
        else:
            require(bank.sha(GLOBAL) == row['profile_sha256'] and quota.read_profile(GLOBAL) == ranking
                    and census.expected_membership(GLOBAL) == stock, 'frozen stock profile bytes changed')

    ctx.prepare, ctx.current, ctx.environment, ctx.inspect = prepare, current, environment, inspect
    return ctx


if __name__ == '__main__':
    ranked = sys.argv[1:2] == ['--ranked']
    if ranked:
        sys.argv.pop(1)
    action, = sys.argv[1:]
    allowed = RANKED_ORDER if ranked else ORDER
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', *allowed), 'select a placement control stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        controller = experiment(ranked=ranked)
        if action == 'prepare':
            controller.prepare()
        else:
            controller.run(action)
