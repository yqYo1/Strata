import fcntl
import importlib.util
import json
import statistics
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
spec = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'qualified owner')


def stats(values):
    return dict(samples=values, count=len(values), mean=statistics.mean(values),
                median=statistics.median(values),
                sample_standard_deviation=statistics.stdev(values),
                minimum=min(values), maximum=max(values))


with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-native-dequant-group-component-analysis-root-v2'
    m.require(not out.exists(), 'immutable analysis')
    out.mkdir(mode=0o700)
    r = dict(active=True, complete=False, passed=False, main_analysis=True,
             controller_sha256=m.sha(__file__), started_utc=m.utc(),
             commands=[], receipt_pins={}, candidate_adopted=False,
             prefill_caller_batch_applied=False, model_executed=False)

    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')

    def read_closed(name, path):
        v = json.loads(path.read_text())
        m.require(v['passed'] and v['complete'] and not v['active'], 'closed ' + name)
        for c in v['commands']:
            m.require(m.closed(c), 'closed owned command ' + name)
            ident = c.get('direct_identity')
            now = m.identity(ident['pid']) if ident else None
            m.require(not now or now['start_ticks'] != ident['start_ticks'], 'owned process absent ' + name)
        r['receipt_pins'][name] = dict(path=str(path), sha256=m.sha(path))
        return v

    save()
    try:
        byte = read_closed('all18-byte', B / 'xe-native-dequant-group-byte-gpu-root-v4-free-full/record.json')
        m.require(byte['coverage']['group_cases'] == 18432, 'full original byte oracle')
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        m.require(byte['boot_id'] == boot, 'same actual boot')
        r['boot_id'] = boot
        r['source_pins'] = byte['source_pins']
        m.require(all(m.sha(W / f) == h for f, h in r['source_pins'].items()), 'exact source')
        prior_path = B / 'xe-native-dequant-group-component-analysis-v1.json'
        m.require(m.sha(prior_path) == '42699c3cd2c5530faa434cf2fdb23084b953898c9dc07ed7c6bfc3b39d5b8f56', 'unchanged rejected v1 analysis')
        prior = json.loads(prior_path.read_text())
        r['prior_analysis'] = dict(path=str(prior_path), sha256=m.sha(prior_path))
        prior_rows = {(x['phase'], x['gu_type'], x['down_type'], x['n']): x for x in prior['rows']}
        rows = []
        for phase in ['full-host', 'full-tags']:
            for matrix in ['smoke-' + phase.split('-')[1], phase]:
                v = read_closed(matrix, B / ('xe-native-dequant-group-component-gpu-root-v2-free-' + matrix + '/record.json'))
                m.require(v['source_pins'] == r['source_pins'] and v['boot_id'] == boot and v['api_version'] == 2, 'fresh v2 scope')
                m.require(v['binary']['sha256'] == '61cf8d7156cda8b0bc8f72f89b7778a69075cc7b7a8efcb1f83a2fe318db71e6', 'current component binary')
            m.require(v['coverage']['samples'] == 252 and v['coverage']['complete_byte_compares'] == 105, 'all full component work')
            metrics = ['host_submit_ms', 'host_submit_checked_drain_ms', 'host_submit_per_expert_us', 'host_submit_checked_drain_per_expert_us']
            if phase == 'full-tags':
                metrics += ['completed_stream_tag_ms', 'completed_stream_tag_per_expert_us']
            for gu, down in [(18, 20), (18, 42), (21, 20), (21, 42), (22, 20), (22, 42), (23, 20)]:
                for n in [1, 8, 16]:
                    row = dict(phase=phase, gu_type=gu, down_type=down, n=n, arms={})
                    old = prior_rows[phase, gu, down, n]
                    for arm in ['individual', 'group']:
                        selected = [x for x in v['samples'] if (x['gu_type'], x['down_type'], x['n'], x['arm']) == (gu, down, n, arm)]
                        m.require(len(selected) == 6 and [x['index'] for x in selected] == list(range(6)), 'all six ABBA samples')
                        row['arms'][arm] = {metric: stats([x[metric] for x in selected]) for metric in metrics}
                    row['median_individual_over_group'] = {
                        metric: row['arms']['individual'][metric]['median'] / row['arms']['group'][metric]['median'] for metric in metrics}
                    row['prior_v1_group_median_over_v2_group'] = {
                        metric: old['arms']['group'][metric]['median'] / row['arms']['group'][metric]['median'] for metric in metrics}
                    row['prior_v1_individual_median_over_v2_individual'] = {
                        metric: old['arms']['individual'][metric]['median'] / row['arms']['individual'][metric]['median'] for metric in metrics}
                    rows.append(row)
        m.require(len(rows) == 42, 'all 42 pair-n-phase rows')
        r['rows'] = rows
        r['summary_by_n'] = {}
        for n in [1, 8, 16]:
            summary = {}
            for phase, metric in [('full-host', 'host_submit_checked_drain_ms'), ('full-tags', 'completed_stream_tag_ms')]:
                selected = [x for x in rows if x['n'] == n and x['phase'] == phase]
                ratios = [x['median_individual_over_group'][metric] for x in selected]
                prior_ratios = [x['prior_v1_group_median_over_v2_group'][metric] for x in selected]
                summary[phase] = dict(individual_over_group_range=[min(ratios), max(ratios)],
                                      group_faster_pair_count=sum(x > 1 for x in ratios),
                                      prior_v1_group_over_v2_group_range=[min(prior_ratios), max(prior_ratios)])
            r['summary_by_n'][str(n)] = summary
        r['decision'] = 'No production adoption from this component result alone. Record all same-work samples and the prior rejected v1 outcome; subsequent caller qualification requires model correctness, lifetime and actual long-input measurements.'
        r['limitations'] = [
            'Six independently drained ABBA samples per arm within one process per phase, not independent model runs.',
            'Completed queue tag interval includes issuance gaps and end-tag work, not exclusive kernel duration.',
            'Component excludes H2D/GEMM/routing/decode and does not attribute whole-model bottlenecks.',
            'Prior v1 and new v2 are separate runs: cross-version ratios are historical comparisons, not a fresh randomized compiler-version experiment.',
            'No new32K/64K/full262144/RESTORE/global quality or model throughput claim.',
            'Both mode programs were freshly built byte-identical; these GPU runs use free mode only. Mode-specific dense products are excluded.'
        ]
        m.require(all(m.sha(Path(x['path'])) == x['sha256'] for x in r['receipt_pins'].values()), 'receipts stable')
        m.require(all(m.sha(W / f) == h for f, h in r['source_pins'].items()), 'source stable')
        r['complete'] = True
    except BaseException as exc:
        r['error'] = repr(exc)
    finally:
        r['active'] = False
        r['passed'] = bool(r['complete'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k: r.get(k) for k in ['passed', 'error', 'summary_by_n']}))
    raise SystemExit(0 if r['passed'] else 1)
