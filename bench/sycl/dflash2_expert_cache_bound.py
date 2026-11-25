"""Compute optimistic static expert-cache bounds from two closed traces."""

import fcntl
from fractions import Fraction
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_expert_rank_forecast as forecast

bank, basis, require = forecast.bank, forecast.basis, forecast.require
KEY = 'dflash2_static_expert_cache_bound_v1'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
LINEAR_READ_BYTES = 39303500000
TARGET_TG = 70
ALIGN = 256


def capacity_bytes(formats, pairs):
    return sum((formats[layer][2] + ALIGN - 1) // ALIGN * ALIGN for layer, _ in pairs)


def relaxed_bound(usage, formats, protected, capacity, target_visits):
    """Fractional cache gives a lower bound on logical CPU visits."""
    locked = capacity_bytes(formats, protected)
    require(locked <= capacity, 'cache must include the protected physical slots')
    groups = []
    total = 0
    for layer, expert, counts in usage:
        if (layer, expert) in protected:
            continue
        raw = formats[layer][2]
        visits = sum(counts) * raw
        payload = (raw + ALIGN - 1) // ALIGN * ALIGN
        require(payload > 0 and visits > 0, 'invalid expert size or observed work')
        total += visits
        groups.append((Fraction(visits, payload), visits, payload))
    groups.sort(reverse=True)
    left, saved = capacity - locked, Fraction()
    needed = max(Fraction(), total - target_visits)
    minimum = Fraction(locked)
    for ratio, visits, payload in groups:
        moved = min(left, payload)
        saved += moved * ratio
        left -= moved
        removed = min(needed, visits)
        minimum += removed / ratio
        needed -= removed
    require(not needed and saved <= total, 'invalid relaxed cache bound')
    remain = total - saved
    return dict(logical_bytes_lower_bound=remain.__ceil__(), minimum_payload_for_target=minimum.__ceil__())


def run():
    revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
    tools = forecast.verify_sources(revision)
    require((bank.P / THIS).read_bytes() == basis.validation.git('show', revision + ':' + THIS),
            'commit the bound evaluator before using the closed traces')
    tools[THIS] = bank.sha(bank.P / THIS)
    out = {}
    for role in ('train', 'holdout'):
        usage, formats, geometry, observed = forecast.observed(role)
        profile = forecast.make_profile.read_profile(geometry['profile'])
        by_layer = forecast.by_layer(profile)
        installed = [(layer, e) for layer, row in enumerate(by_layer)
                     for e in row[:forecast.CAPACITY[layer]]]
        protected = set(installed[:forecast.PREFIX])
        require(len(installed) == forecast.CACHE_SLOTS and len(protected) == forecast.PREFIX,
                'physical slot quota or protected prefix changed')
        stock_bytes = capacity_bytes(formats, installed)
        require(stock_bytes == 2575206400, 'aligned stock expert budget differs')
        stock = observed['jobs']['logical_job_blob_bytes']
        n = observed['generated_tokens']
        target = Fraction(LINEAR_READ_BYTES * n, TARGET_TG)
        fixed = forecast.routing.rank_by_verifier_jobs(profile, [usage], cache_slots=forecast.CACHE_SLOTS,
                                                       protected_prefix=forecast.PREFIX)
        fixed_work = forecast.routing.projected_cpu_expert_work(usage, forecast.membership(fixed), formats)
        fixed_bound = fixed_work['logical_job_blob_bytes']
        rows = []
        for budget in (stock_bytes, 4 << 30, 6 << 30, 8 << 30):
            point = relaxed_bound(usage, formats, protected, budget, target)
            lower = point['logical_bytes_lower_bound']
            require(0 <= lower <= fixed_bound <= stock and
                    (not rows or lower <= rows[-1]['logical_bytes_lower_bound']),
                    'the relaxed bound must improve monotonically on the feasible fixed quotas')
            rows.append(dict(payload_budget=budget, **point,
                             logical_GB_per_s_at_target=lower * TARGET_TG / n / 1e9))
        out[role] = dict(inputs=observed, protected_payload_bytes=capacity_bytes(formats, protected),
                         fixed_quota_oracle_logical_bytes=fixed_bound,
                         stock_logical_GB_per_s_at_target=stock * TARGET_TG / n / 1e9, bounds=rows)
    result = bank.previous.read(bank.RESULT)
    require(KEY not in result, 'one archived bound for this evidence')
    result[KEY] = dict(stage='closed', source_commit=revision, tools=tools, target_tg=TARGET_TG,
                      continuous_read_reference_bytes_per_second=LINEAR_READ_BYTES, workloads=out,
                      scope='Static perfect-future ranking. Protected slots stay. Relax per-layer quotas, '
                            'slot count, and whole-expert residency. Packed visits are logical. The read '
                            'reference is illustrative and ignores overlap, computation and GPU memory overhead.',
                      GPU_executed=False, CPU_performance_measured=False)
    bank.stage.write(bank.RESULT, result)
    for role, row in out.items():
        print(role, row['stock_logical_GB_per_s_at_target'], [
            (v['payload_budget'], v['logical_GB_per_s_at_target'], v['minimum_payload_for_target'])
            for v in row['bounds']], flush=True)


if __name__ == '__main__':
    require(sys.executable == basis.reference.model.PYTHON and len(sys.argv) == 1,
            'use the pinned interpreter for the offline analysis')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        run()
