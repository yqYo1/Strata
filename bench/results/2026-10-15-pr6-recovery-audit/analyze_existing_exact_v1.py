#!/usr/bin/env python3
"""Closed setting screen: preserve every sample without conflating changed work."""
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def service(request):
    table = {x['pair']: x for x in request['format_service']['formats']}
    by_gu, roles = {}, dict.fromkeys(['gu', 'quant', 'down'], 0)
    for c in request['format_service']['counters']:
        role = int(c['role']) % 3
        ns = int(c['cpu_ns'])
        roles[['gu', 'quant', 'down'][role]] += ns
        if role == 0:
            fmt = table[c['pair']]['gu']
            by_gu[fmt] = by_gu.get(fmt, 0) + ns
    return dict(cpu_thread_ns=roles, gu_by_type_ns=by_gu,
                existing_ms={x['index']: float(x['delta']) for x in request['ledger']['existing_ms']})


def main():
    paths = [P / x for x in ['decode-format-service-normal-v1.json',
                            'decode-format-existing-exact-normal-v1.json']]
    baseline, candidate = [json.loads(p.read_text()) for p in paths]
    for d in [baseline, candidate]:
        assert d['passed'] and d['complete'] and not d['active'] and not d['fresh_faults']
        assert d['boot_unchanged'] and not d['driver_dump_exists']
    assert baseline['binary_sha256'] == candidate['binary_sha256']
    assert baseline['source_pins'] == candidate['source_pins']
    assert baseline['argv'] == candidate['argv']
    out = dict(GPU_executed=False, complete=True, adopted=False,
               performance_qualified=False, quality_qualified=False,
               receipt_sha256={p.name: sha(p) for p in paths}, samples=[],
               scope='Offline comparison of closed diagnostic setting screens; thread service is concurrent and not additive with wall spans.')
    for a, b in zip(baseline['requests'], candidate['requests'], strict=True):
        sa, sb = service(a), service(b)
        different = next((i for i, (x, y) in enumerate(zip(a['generated_ids'], b['generated_ids'])) if x != y), None)
        out['samples'].append(dict(index=a['index'], baseline_done=a['done'], exact_done=b['done'],
            generated_ids_equal=a['generated_ids'] == b['generated_ids'], first_different_generated_index=different,
            work_equal=b['baseline_comparison']['coverage_equal'],
            job_nt_lines_equal=b['baseline_comparison']['job_nt_lines_equal'],
            baseline_coverage=a['ledger']['coverage'], exact_coverage=b['ledger']['coverage'],
            baseline_service=sa, exact_service=sb,
            gu_type_cpu_time_ratios={t: sb['gu_by_type_ns'][t] / ns for t, ns in sa['gu_by_type_ns'].items() if ns},
            gu_total_cpu_time_ratio=sb['cpu_thread_ns']['gu'] / sa['cpu_thread_ns']['gu'],
            decode_wall_time_ratio=b['done']['decode_ms'] / a['done']['decode_ms'],
            causal_performance_claim=False))
    out['decision'] = ('Keep STRATA_IQ_SINGLE_EXACT default OFF. The first long sample has identical IDs/work and a small wall difference; '
                       'the second has different IDs/work. These are not repeated clean official benchmark or independent quality evidence.')
    (P / 'existing-exact-setting-summary-v1.json').write_text(json.dumps(out, indent=2) + '\n')
    for s in out['samples']:
        print(json.dumps({k: s[k] for k in ['index', 'generated_ids_equal', 'first_different_generated_index',
              'work_equal', 'gu_type_cpu_time_ratios', 'gu_total_cpu_time_ratio', 'decode_wall_time_ratio']}))


if __name__ == '__main__':
    main()
