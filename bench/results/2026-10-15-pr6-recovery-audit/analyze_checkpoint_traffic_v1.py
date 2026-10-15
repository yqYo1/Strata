#!/usr/bin/env python3
"""Offline byte/scenario arithmetic from retained primary evidence; no GPU or timing."""
from pathlib import Path
import hashlib
import json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    primary = here.parent / '2026-10-14-primary-transfer-ceiling/results.json'
    receipt = json.loads(primary.read_text())
    layers = receipt['metadata']['layers']
    assert receipt['passed'] and len(layers) == 48
    experts = receipt['metadata']['n_expert']
    assert experts == 512 and [v['layer'] for v in layers] == list(range(48))
    payload = sum(v['blob_bytes'] * experts for v in layers)
    gdn = 36 * (128 * 48 * 128 + 10240 * 3) * 4
    groups = receipt['groups']
    bulk = next(g for g in groups if g['lane'] == 'bulk' and
                g['allocation'] == 'ordinary-host' and g['direction'] == 'H2D')
    expert_groups = [g for g in groups if g['lane'] == 'expert' and
                     g['allocation'] == 'ordinary-host' and g['direction'] == 'H2D']
    fastest = max(((v, g) for g in expert_groups
                   for v in g['individual_values']['host_group_GBps']), key=lambda pair: pair[0])
    rates = {
        'ordinary_host_bulk_median_of_medians': bulk['summary']['median_GBps']['median'],
        'fastest_observed_ordinary_expert_host_group': fastest[0],
    }
    cases = []
    interval, chunk = 16384, 8192
    for tokens in (32768, 65536, 262144):
        remaining, segments = tokens - 1, []
        while remaining:
            count = min(interval, remaining)
            assert count > chunk  # Every segment in these three cases uses long LM.
            segments.append(count)
            remaining -= count
        extra = (len(segments) - 1) * payload
        positions = list(range(interval, tokens, interval))
        cases.append(dict(input_tokens=tokens, batched_tokens=tokens - 1,
            chunk=chunk, interval=interval, root_turn_boundaries=False,
            segment_tokens=segments, extra_full_expert_payload_bytes=extra,
            extra_transfer_seconds_scenarios={k: extra / (v * 1e9) for k, v in rates.items()},
            target_1000tps_wall_seconds=tokens / 1000,
            actual_periodic_checkpoint_positions=positions,
            checkpoint_gdn_all_boundaries_payload_bytes=len(positions) * gdn,
            checkpoint_gdn_six_retained_payload_bytes=min(6, len(positions)) * gdn))
    result = dict(schema='checkpoint-traffic-scenarios-v1',
        status='OFFLINE_ARITHMETIC_ONLY_NOT_RUNTIME_MEASUREMENT',
        source_receipt=str(primary.relative_to(here.parent.parent.parent)),
        source_sha256=digest(primary), recipe_sha256=digest(Path(__file__)),
        full_lm_logical_expert_payload_bytes=payload,
        checkpoint_gdn_payload_bytes=gdn, checkpoint_gdn_mib=gdn / (1 << 20),
        rates_GBps=rates,
        fastest_observed_group={k: fastest[1][k] for k in ('blob_bytes', 'copies', 'allocation', 'direction')},
        cases=cases, hardware_peak_established=False, actual_model_transfer_measured=False,
        performance_qualified=False,
        limitations=[
            'Measured synthetic rates are scenario inputs, not maximum bandwidth or time lower bounds.',
            'Different batch, allocation, direction, overlap, cache and queue conditions can change rates.',
            'Logical expert bytes are not measured physical PCIe or DRAM traffic.',
            'GDN-only volumes exclude PLE/QSA/IDs, copies of prior checkpoints and admission overhead.',
            'Final n-1, root/turn, resume and chunk changes require separate actual schedules.',
            'This arithmetic does not prove a throughput target impossible or a candidate faster.',
        ])
    (here / 'checkpoint-traffic-scenarios-v1.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(payload_bytes=payload, gdn_bytes=gdn,
                         scenarios=[c['extra_transfer_seconds_scenarios'] for c in cases])))


if __name__ == '__main__':
    main()
