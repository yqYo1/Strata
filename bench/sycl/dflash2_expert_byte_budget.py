"""Replay decode-only expert quotas under the current payload and slot caps."""

import fcntl
from fractions import Fraction
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_expert_rank_forecast as forecast

bank, basis, require = forecast.bank, forecast.basis, forecast.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
EXPERTS = 512
TARGET_TG = 70
LARGE_COPY_GBPS = (6.291627, 6.485465)
LARGE_COPY_EVIDENCE = 'a0404d96e:prefill_copy_engine_v6'
PROTOCOL = (
    'Select from the existing regular-code training capture only. Compare stock, '
    'training rank with fixed quotas, and a training rank with variable quotas on '
    'the separate prose holdout. Keep the first 132 physical slots and hold '
    'the 1,256-slot and original raw expert-payload limits. Require fewer held-out '
    'jobs without increasing logical bytes against either control before proposing a runtime layout. '
    'No host-mirror alignment, real cache traffic or throughput is measured.'
)


def payload(capacities, sizes):
    return sum(cap * size for cap, size in zip(capacities, sizes, strict=True))


def estimate(train, stock, sizes):
    """Keep the original loan prefix and choose remaining slots by visits per byte."""
    orders = forecast.by_layer(stock)
    require(all(len(row) == EXPERTS and set(row) == set(range(EXPERTS)) for row in orders),
            'require a complete ordered expert profile')
    visits = {(layer, expert): sum(counts) for layer, expert, counts in train}
    pools, selected, locked_layers = [], [], 0
    left = forecast.PREFIX
    for layer, (order, width, size) in enumerate(zip(orders, forecast.CAPACITY, sizes, strict=True)):
        keep = min(width, left)
        selected.append(order[:keep])
        left -= keep
        if keep == width:
            # More slots here would move a later layer's protected entries.
            locked_layers += 1
            continue
        for expert in order[keep:]:
            gain = visits.get((layer, expert), 0)
            if gain:
                pools.append((-Fraction(gain, size), layer, expert))

    require(left == 0, 'protected physical prefix is longer than the stock cache')
    available = payload(forecast.CAPACITY, sizes) - payload([len(row) for row in selected], sizes)
    slots = forecast.CACHE_SLOTS - forecast.PREFIX
    for _, layer, expert in sorted(pools):
        if not slots:
            break
        size = sizes[layer]
        if size <= available:
            selected[layer].append(expert)
            available -= size
            slots -= 1

    capacities = [len(row) for row in selected]
    original_prefix = [(l, e) for l, order in enumerate(orders)
                       for e in order[:forecast.CAPACITY[l]]][:forecast.PREFIX]
    selected_prefix = [(l, e) for l, order in enumerate(selected) for e in order][:forecast.PREFIX]
    require(selected_prefix == original_prefix and capacities[:locked_layers] == forecast.CAPACITY[:locked_layers]
            and sum(capacities) <= forecast.CACHE_SLOTS
            and payload(capacities, sizes) <= payload(forecast.CAPACITY, sizes),
            'variable quotas moved protected slots or exceeded either budget')
    return capacities, [sum(1 << expert for expert in order) for order in selected]


def ideal_link_envelope(sample):
    # This charges one raw expert blob for each logged logical job. A real
    # mirror can serve bytes from several levels of reuse, with other costs.
    bytes_per_output = sample['jobs']['logical_job_blob_bytes'] / sample['generated_tokens']
    return dict(logical_job_blob_bytes_per_output=bytes_per_output,
                GBps_if_copied_at_target=TARGET_TG * bytes_per_output / 1e9,
                tok_s_if_full_payload_copied=[g * 1e9 / bytes_per_output for g in LARGE_COPY_GBPS])


def main():
    require(sys.executable == basis.reference.model.PYTHON, 'use the pinned Python')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
        sources = forecast.verify_sources(revision)
        require((bank.P / THIS).read_bytes() == basis.validation.git('show', revision + ':' + THIS),
                'commit the exact byte-budget replay and protocol')
        train, formats, geometry, training = forecast.observed('train')
        heldout, checked_formats, checked_geometry, testing = forecast.observed('holdout')
        original = basis.validation.committed(bank.RESULT, (forecast.KEY,))
        require(formats == checked_formats and geometry == checked_geometry == original['geometry']
                and training == original['train'] and testing == original['holdout']
                and training['request_sha256'] != testing['request_sha256']
                and original['stage'] == 'scored', 'recover the previously checked independent captures')
        sizes = [item[2] for item in formats]
        require(all(size > 0 for size in sizes), 'invalid expert bytes')
        stock = forecast.make_profile.read_profile(geometry['profile'])
        precomputed = forecast.apply_delta(stock, original['profile']['profile_delta'])
        capacities, variable = estimate(train, stock, sizes)
        controls = {'stock': forecast.membership(stock), 'fixed_train_rank': forecast.membership(precomputed)}
        results = {}
        for name, member in dict(controls, variable_train_quotas=variable).items():
            quotas = capacities if name == 'variable_train_quotas' else forecast.CAPACITY
            results[name] = dict(capacities=quotas, slots=sum(quotas), payload_bytes=payload(quotas, sizes),
                                 train=forecast.routing.projected_cpu_expert_work(train, member, formats),
                                 heldout=forecast.routing.projected_cpu_expert_work(heldout, member, formats))
        require(results['stock']['train'] == training['jobs'] and results['stock']['heldout'] == testing['jobs']
                and results['fixed_train_rank']['heldout'] == original['holdout_forecast'],
                'the established baseline or fixed-quota forecast failed to reproduce')
        contender = results['variable_train_quotas']['heldout']
        advance = all(
            contender['cpu_jobs'] < results[name]['heldout']['cpu_jobs']
            and contender['logical_job_blob_bytes'] <= results[name]['heldout']['logical_job_blob_bytes']
            for name in controls
        )
        print(json.dumps(dict(source=revision, source_sha256=bank.signature(sources | {THIS: bank.sha(bank.P / THIS)}),
             protocol=PROTOCOL, request_sha256=[training['request_sha256'], testing['request_sha256']],
             manifest_sha256=geometry['manifest_sha256'], protected_slots=forecast.PREFIX,
             raw_stock_nonresident_blob_bytes=payload([EXPERTS - cap for cap in forecast.CAPACITY], sizes),
             raw_expert_sizes=sizes, results=results, eligible_for_layout_check=advance,
             hypothetical_link_envelope=dict(copy_reference=LARGE_COPY_EVIDENCE,
                                             training=ideal_link_envelope(training),
                                             heldout=ideal_link_envelope(testing))), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
