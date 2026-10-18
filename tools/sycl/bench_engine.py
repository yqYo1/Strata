#!/usr/bin/env python3
"""Prepare/commit/run a matched official bench.sh GDN or QSA comparison; Main only."""
import fcntl
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import re
import shutil
import socket
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import measurement_guards as guards
import measurement_owner as supervision

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
A = P / 'bench/results/2026-10-15-pr6-recovery-audit'
RESULT = A / 'native-gdn-model-current.json'
BUILD = A / 'native-gdn-engine-build.json'
BASE = A / 'qsa-softmax-measurement-inputs-v1.json'
SETTINGS = P / 'tools/sycl/measurement_environment.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compact_window_census(windows):
    """Keep source-shaped work volume after checking every planned window."""
    histogram = Counter()
    first = windows[0]
    for window in windows:
        guards.require(all(window[k] == first[k] for k in
            ('gu_nk', 'down_nk', 'gu_down_storage', 'hidden_storage')),
            'inconsistent census geometry/storage')
        guards.require(len(dict(window['job_rows_histogram'])) == len(window['job_rows_histogram']),
                       'duplicate census histogram bin')
        histogram.update(dict(window['job_rows_histogram']))
    return dict(window_count=len(windows), layer_rows_jobs_columns=['layer', 'rows', 'jobs'],
        layer_rows_jobs=[[w['layer'], w['rows'], w['jobs']] for w in windows],
        aggregate_job_rows_histogram=sorted(histogram.items()),
        jobs=sum(w['jobs'] for w in windows), routed_rows=sum(w['rows'] * 10 for w in windows),
        gu_nk=first['gu_nk'], down_nk=first['down_nk'],
        gu_down_storage=first['gu_down_storage'], hidden_storage=first['hidden_storage'],
        requested_router_D2H_bytes=sum(w['requested_router_D2H_bytes'] for w in windows),
        requested_map_H2D_bytes=sum(w['requested_map_H2D_bytes'] for w in windows),
        all_job_row_and_expert_conservation_checks_passed=True, planned_matches_retired=True,
        limits='Requested/logical metadata; not physical PCIe or VRAM traffic, occupancy or additive wall time.')


def compact_host_resource_profile(profile):
    """Keep aggregate attribution after validating every request/layer/window."""
    census = compact_window_census(profile['planned_moe_windows'])
    histogram = census.pop('aggregate_job_rows_histogram')
    census.pop('layer_rows_jobs')
    census.pop('layer_rows_jobs_columns')
    census['window_rows_counts'] = sorted(Counter(w['rows'] for w in profile['planned_moe_windows']).items())
    census['job_size_groups'] = [dict(min_rows=lo, max_rows=hi,
        jobs=sum(count for rows, count in histogram if lo <= rows <= hi),
        routed_rows=sum(rows * count for rows, count in histogram if lo <= rows <= hi))
        for lo, hi in ((1, 2333), (2334, 8192))]
    guards.require(sum(group['jobs'] for group in census['job_size_groups']) == census['jobs'] and
        sum(group['routed_rows'] for group in census['job_size_groups']) == census['routed_rows'],
        'compact job groups lost work')
    return dict(resource_diagnostics=[line for line in profile['resource_diagnostics']
        if 'layer=' not in line or 'layer=-1 ' in line],
        host_counter_conservation=profile['host_counter_conservation'],
        window_census=census, all_resource_gates_passed=True,
        detail_not_retained='All per-layer/window conservation and ownership checks run before reduction. Full layer arrays and exact job-size histograms are not needed by the current aggregate host-source attribution question.')

def compact_command_service(profile):
    """Preserve every selected interval without repeated per-sample field names."""
    for name in ('sample', 'group'):
        rows = profile[name + 's']
        columns = list(rows[0])
        guards.require(all(set(row) == set(columns) for row in rows),
                       'inconsistent command record fields')
        profile[name + '_columns'] = columns
        profile[name + 's'] = [[row[k] if k == 'kind' else int(row[k]) for k in columns] for row in rows]
    return profile


def selected_service_screen(profile):
    """Descriptive per-stratum sample-mean expansion; never a hardware ceiling."""
    kinds = {}
    windows = set()
    for values in profile['groups']:
        group = dict(zip(profile['group_columns'], values))
        windows.add(group['window'])
        item = kinds.setdefault(group['kind'], dict(eligible_commands=0, sampled_commands=0,
            expanded_sample_mean_service_ns=0.0))
        item['eligible_commands'] += group['eligible']
        item['sampled_commands'] += group['service_samples']
        item['expanded_sample_mean_service_ns'] += (group['eligible'] *
            group['sample_service_ns'] / group['service_samples'])
    return dict(window_ids=sorted(windows), per_kind=kinds,
        method='Within each selected window/kind/row-bin, multiply the sample mean service by eligible command count, then sum by kind.',
        limits='Deterministic sparse samples in four selected windows; descriptive screening assumes within-stratum representativeness. Not full-model service, statistical confidence, GPU utilization, critical path, physical bytes or a hardware ceiling.')


CPU_HOST_BASE = ('source_copy', 'host_group', 'gemm_submit', 'route_exchange',
    'upload_submit', 'residual_input_host', 'residual_input_device',
    'residual_store_host', 'residual_store_device')
CPU_HOST_MOE = ('moe_map_upload', 'moe_job_weights', 'moe_dequant_gu',
    'moe_dequant_down', 'moe_gather', 'moe_swiglu', 'moe_accumulate', 'moe_status')


def split_long_request_logs(log, count):
    starts = [m.start() for m in re.finditer(r'(?m)^strata prefill: GDN columns=', log)]
    guards.require(len(starts) == count, 'wrong number of long-request start markers')
    guards.require(not any(x in log[:starts[0]] for x in
        ('strata lm profile: v=', 'strata command profile:', 'strata prefill cpu wait:')),
        'resource observer before first long-request marker')
    return [log[a:b] for a, b in zip(starts, starts[1:] + [len(log)])]


def extract_mixed_owners(log, window_pairs, tail_offset):
    entries = [x for x in log.splitlines() if x.startswith('strata MoE mixed owners: phase=entry')]
    retired = [x for x in log.splitlines() if x.startswith('strata MoE mixed owners: phase=retired')]
    expected_keys = [(layer, first, rows) for layer in range(48) for first, rows in window_pairs]
    guards.require(len(entries) == len(retired) == len(expected_keys) and all(
        'direct_owners=1 previous_attention=checked_drain' in x for x in entries) and all(
        'route_status=0 shared_HC=checked_drain callbacks=after_return' in x for x in retired),
        'incomplete mixed owner/phase census')
    samples = [dict(field.split('=', 1) for field in x.split()[4:]) for x in entries]
    done = [dict(field.split('=', 1) for field in x.split()[4:]) for x in retired]
    keys = lambda items: [tuple(int(x[k]) for k in ('layer', 'first', 'rows')) for x in items]
    guards.require(keys(samples) == keys(done) == expected_keys,
                   'mixed owner census has duplicate, missing or reordered windows')
    guards.require(all(int(x['primary_bytes']) == min(int(x['rows']), 65536) * 2560 * 2 and
        int(x['tail_bytes']) == max(0, int(x['rows']) - 65536) * 2560 * 2 and
        int(x['region_bytes']) == 1180698624 and int(x['tail_offset']) == tail_offset and
        int(x['residual_bytes']) == int(x['rows']) * 4 * 2560 * 2 for x in samples),
        'mixed owner census has unexpected logical extents')
    columns = ('layer', 'first', 'rows', 'primary_bytes', 'region_bytes',
               'tail_offset', 'tail_bytes', 'residual_bytes')
    return dict(owner_columns=columns, owner_samples=[[int(x[k]) for k in columns] for x in samples],
                retired_windows=len(retired), all_direct_owners_and_checked_drains=True)


def extract_cpu_observers(log, expected_version=None):
    """A complete accounting partition is distinct from a successful request."""
    result = []
    for prefix in ('strata prefill cpu wait:', 'strata prefill cpu host:'):
        lines = [x for x in log.splitlines() if x.startswith(prefix)]
        guards.require(len(lines) == 1, 'missing or duplicate CPU observer')
        result.append(dict(part.split('=', 1) for part in lines[0][len(prefix):].split()))
    wait, host = result
    version = int(wait['version'])
    guards.require(version in (3, 4) and int(host['version']) == version and
                   (expected_version is None or version == expected_version), 'CPU observer version mismatch')
    guards.require(wait['available'] == host['available'] == wait['complete'] == '1' and
                   wait['clock_or_accounting_error'] == wait['open_waits'] == host['open_regions'] == '0',
                   'incomplete or invalid CPU accounting')
    names = CPU_HOST_BASE + (CPU_HOST_MOE if version == 4 else ())
    for fields in result:
        throws = [int(v) for k, v in fields.items() if k.endswith('_throws')]
        guards.require(throws and not any(throws), 'CPU attribution includes thrown wait/host regions')
        guards.require(all(math.isfinite(float(v)) and float(v) >= 0 for k, v in fields.items()
                           if k.endswith('_s') or k.endswith('_count')), 'invalid CPU interval/count')
    for name in names:
        guards.require(all(name + suffix in host for suffix in
            ('_count', '_throws', '_cpu_s', '_wall_s', '_outside_wait_cpu_s', '_outside_wait_wall_s')),
            'incomplete host category: ' + name)
    cpu_sum = sum(float(host[name + '_outside_wait_cpu_s']) for name in names)
    wall_sum = sum(float(host[name + '_outside_wait_wall_s']) for name in names)
    guards.require(abs(float(wait['outside_wait_cpu_s']) - float(host['unclassified_cpu_s']) - cpu_sum) < 1e-7 and
        abs(float(wait['wall_s']) - float(wait['wait_wall_s']) - float(host['unclassified_wall_s']) - wall_sum) < 1e-7,
        'CPU partition conservation mismatch')
    return dict(version=version, wait=wait, host=host, partition_qualified=True)


def extract_resource_profile(log, phase_profile, command_profile, limits, cpu_version=None,
                             expected_window_pairs=None):
    """Validate one long-request interval; never aggregate across requests."""
    record = {}
    prefix = 'strata MoE planned job census: '
    record['planned_moe_windows'] = [json.loads(x[len(prefix):]) for x in log.splitlines() if x.startswith(prefix)]
    if expected_window_pairs is None:
        guards.require(len(record['planned_moe_windows']) == 48 and
            [x['layer'] for x in record['planned_moe_windows']] == list(range(48)), 'missing window census')
    else:
        guards.require([(w['layer'], w['first_row'], w['rows']) for w in record['planned_moe_windows']] ==
            [(layer, first, rows) for layer in range(48) for first, rows in expected_window_pairs],
            'missing, reordered or changed multi-window census')
    total_jobs = total_rows = 0
    for window in record['planned_moe_windows']:
        guards.require(sum(n for m, n in window['job_rows_histogram']) == window['jobs'] and
            sum(m*n for m, n in window['job_rows_histogram']) == window['rows']*10 and
            sum(n for m, n in window['jobs_per_expert_histogram']) == 512 and
            sum(m*n for m, n in window['jobs_per_expert_histogram']) == window['jobs'], 'census conservation')
        total_jobs += window['jobs']; total_rows += window['rows']
    totals = [x for x in log.splitlines() if x.startswith('strata MoE accumulator window: ')]
    guards.require(len(totals) == 1 and f'jobs={total_jobs} ' in totals[0] and
                   f'windows={len(record["planned_moe_windows"])} ' in totals[0], 'planned/retired job totals differ')
    record['retired_moe_totals'] = totals
    record['resource_diagnostics'] = [x for x in log.splitlines() if x.startswith(
        ('strata lm profile:', 'strata lm qsa profile:', 'strata prefill cpu wait:', 'strata prefill cpu host:'))]
    coarse_prefix = 'strata lm profile: v=' + ('3' if phase_profile else '2') + ' complete=1 errors=0'
    for prefix in (coarse_prefix, 'strata lm qsa profile: v=1 complete=1 errors=0'):
        guards.require(sum(x.startswith(prefix) for x in record['resource_diagnostics']) == 1,
                       'missing or duplicate complete resource observer: ' + prefix)
    cpu = extract_cpu_observers(log, cpu_version)
    if cpu_version == 4:
        window_count = len(record['planned_moe_windows'])
        expected = dict(zip(CPU_HOST_MOE, [window_count, window_count, 0, 0,
                                         total_jobs, total_jobs, total_jobs, window_count]))
        gu = [dict(x.split('=', 1) for x in line.split(': ', 1)[1].split())
              for line in log.splitlines() if line.startswith('strata MoE GU fusion:')]
        guards.require(len(gu) <= 1, 'duplicate request GU totals')
        if gu:
            guards.require(int(gu[0]['fused_jobs']) + int(gu[0]['separate_jobs']) == total_jobs,
                           'GU fusion/work census differs')
            expected['moe_swiglu'] = int(gu[0]['separate_jobs'])
        experts = sum(n for window in record['planned_moe_windows']
                      for m, n in window['jobs_per_expert_histogram'] if m > 0)
        expected.update(moe_dequant_gu=experts, moe_dequant_down=experts)
        if expected_window_pairs is not None:
            expected.update(source_copy=48, upload_submit=48 + experts)
        guards.require(all(int(cpu['host'][name + '_count']) == count for name, count in expected.items()),
                       'MoE host category/work census mismatch')
        record['host_counter_conservation'] = expected
    record['cpu_attribution_qualified'] = True
    record['census_scope'] = 'Planned host metadata reconciled with completed window issuer totals; profiler/validation enabled, not clean speed or physical hardware saturation.'
    if phase_profile:
        # The supported coarse observer checks every event/query and
        # expected stage count. Nested intervals are nonexclusive.
        expected_phases = ('GDN_projections', 'GDN_conv_gates', 'GDN_recurrence', 'GDN_output')
        layers, aggregate = [], None
        for line in log.splitlines():
            if not line.startswith('strata lm profile: layer='):
                continue
            fields = dict(part.split('=', 1) for part in line[len('strata lm profile: '):].split())
            queue = [float(v) for v in fields['queue_ms'].split(',')]
            samples = [int(v) for v in fields['samples'].split(',')]
            host = [float(v) for v in fields['gdn_submit_wall_ms'].split(',')]
            guards.require(len(queue) == len(samples) == 10 and len(host) == 4,
                           'wrong coarse v3 vector extent')
            guards.require(all(math.isfinite(v) and v >= 0 for v in queue + host) and
                           all(v >= 0 for v in samples), 'invalid coarse v3 interval/sample')
            row = dict(layer=int(fields['layer']), expected_chunks=int(fields['gdn_expected_chunks']),
                       gdn_parent_queue_ms=queue[2], queue_ms=queue[6:],
                       samples=samples[6:], host_submit_wall_ms=host)
            guards.require(all(n == row['expected_chunks'] for n in samples[6:]) and
                           samples[2] == row['expected_chunks'], 'GDN stage/parent count mismatch')
            if row['layer'] == -1:
                guards.require(aggregate is None, 'duplicate aggregate')
                aggregate = row
            else:
                layers.append(row)
        record['gdn_phase_queue_intervals'] = dict(order=expected_phases, layers=layers, aggregate=aggregate,
            limits=limits)
        guards.require([x['layer'] for x in layers] == list(range(48)) and
            sum(x['expected_chunks'] > 0 for x in layers) == 36 and aggregate is not None and
            aggregate['expected_chunks'] == sum(x['expected_chunks'] for x in layers),
            'incomplete per-layer GDN stage coverage')
    if command_profile:
        def rows(prefix):
            return [dict(part.split('=', 1) for part in line[len(prefix):].split())
                    for line in log.splitlines() if line.startswith(prefix)]
        summaries = rows('strata command profile: ')
        groups = rows('strata command group: ')
        samples = rows('strata command sample: ')
        invalid = rows('strata command invalid: ')
        # Keep service intervals and signed clock diagnostics, not a
        # permanent raw absolute timeline. All samples remain present.
        for sample in samples:
            submit, start, end = (int(sample.pop(key)) for key in ('submit', 'start', 'end'))
            sample.update(service_ns=end-start, start_minus_submit_ns=start-submit)
        record['command_service_profile'] = dict(version=2, summaries=summaries,
            groups=groups, samples=samples, invalid=invalid,
            scope='Deterministic samples in first accumulator expert phases of layers0,15,31,47; initial router/shared/final writes excluded; no device-busy or physical-traffic claim.')
        frame_layers = {0: 0, 1: 15, 2: 31, 3: 47}
        group_keys = [(int(g['window']), g['kind'], int(g['bin'])) for g in groups]
        guards.require(len(set(group_keys)) == len(group_keys) and
            {int(g['window']) for g in groups} == set(frame_layers) and
            all(frame_layers.get(int(g['window'])) == int(g['layer']) and int(g['first_row']) == 0
                for g in groups), 'unexpected or duplicate command window/stratum')
        observed = Counter((int(s['window']), s['kind'], int(s['bin'])) for s in samples)
        guards.require(observed == {key: int(g['service_samples']) for key, g in zip(group_keys, groups)},
                       'command samples do not match window/stratum counts')
        guards.require('strata MoE window configuration: requested_mode=10 effective_mode=10 ' in log,
            'command service must sample the admitted production accumulator route')
        guards.require(len(summaries) == 1 and summaries[0]['version'] == '2' and
            summaries[0]['service_complete'] == '1' and summaries[0]['invalid'] == '0' and
            summaries[0]['windows'] == summaries[0]['expected'] == '4' and not invalid and
            len(samples) == int(summaries[0]['records']) and groups and
            all(int(g['selected']) == int(g['service_samples']) and int(g['selected']) > 0 for g in groups) and
            sum(int(g['service_samples']) for g in groups) == len(samples) and
            sum(int(g['queue_delay_samples']) for g in groups) +
                int(summaries[0]['submit_clock_inversions']) == len(samples) and
            sum(s['start_minus_submit_ns'] < 0 for s in samples) == int(summaries[0]['submit_clock_inversions']) and
            all(s['service_ns'] >= int(summaries[0]['timer_resolution_ns']) for s in samples),
            'incomplete GPU-only command service profile')
    return record


def summarize_gu():
    """Close the committed three-pair plan without launching GPU work."""
    path = str((A / 'moe-gu-swiglu-model-current.json').relative_to(P))

    def git(*args):
        return subprocess.check_output(['git', *args], cwd=P)

    def canonical(value, root):
        if isinstance(value, str):
            return value.replace(root, '$RUN')
        if isinstance(value, list):
            return [canonical(x, root) for x in value]
        if isinstance(value, dict):
            return {k: canonical(v, root) for k, v in value.items()
                    if k != 'STRATA_SYCL_PREFILL_GU_SWIGLU'}
        return value

    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        head = git('rev-parse', 'HEAD').decode().strip()
        guards.require((P / path).read_bytes() == git('show', head + ':' + path),
                       'commit the final cohort before summary')
        current = json.loads((P / path).read_text())
        guards.require(current['mode'] == 'qsa-gu-fusion-candidate' and
            current['cohort_index'] == 2 and 'independent_process_comparison' not in current,
            'summarize the completed final GU cohort exactly once')
        records, commit, seen = [], head, set()
        while len(records) < 6:
            guards.require(commit not in seen, 'cyclic cohort provenance')
            seen.add(commit)
            guards.require(subprocess.run(['git', 'merge-base', '--is-ancestor', commit, head],
                cwd=P, check=False).returncode == 0, 'cohort source is not reachable')
            d = json.loads(git('show', commit + ':' + path))
            guards.require(d['mode'] in ('qsa-gu-fusion-base', 'qsa-gu-fusion-candidate') and
                d['passed'] and d['complete'] and d['ownership_closed'] and not d['active'] and
                d['exit_code'] == d['engine_exit_code'] == 0 and not d['fresh_faults'] and
                d['boot_unchanged'], 'incomplete, failed or faulted cohort')
            records.append((commit, d))
            commit = d['gu_qualification']['preceding_result']['commit']
        records.reverse()
        expected = [('base',0),('candidate',0),('candidate',1),('base',1),('base',2),('candidate',2)]
        guards.require([(d['mode'].rsplit('-',1)[1],d.get('cohort_index',0)) for _,d in records]
                       == expected, 'cohort order differs from the frozen plan')
        guards.require(len({(d['native_identity']['pid'],d['native_identity']['start_ticks'])
                            for _,d in records}) == 6, 'independent native processes required')
        guards.require(digest(current['binary']['path']) == current['binary']['sha256'],
                       'current measured executable changed')
        common, arms, source_sets = None, [], []
        for receipt_commit, d in records:
            root = d['environment']['STRATA_MEASUREMENT_ROOT']
            candidate = d['mode'].endswith('-candidate')
            guards.require(d['environment']['STRATA_SYCL_PREFILL_GU_SWIGLU'] ==
                           ('large' if candidate else '0'), 'wrong GU selector')
            command = canonical(d['command'], root)
            command[command.index('--port') + 1] = '$PORT'
            identity = dict(binary=d['binary'], build_source=d['build_source_commit'],
                profile=d['profile_sha256'], environment=canonical(d['environment'], root),
                config=canonical(d['config'], root), command=command)
            common = identity if common is None else common
            guards.require(identity == common, 'unmatched binary, profile, configuration or environment')
            source = d['source_commit']
            names = git('ls-tree','-r','--name-only',source,'--','serve','tools','bench.sh').decode().splitlines()
            pins = {n: hashlib.sha256(git('show', source + ':' + n)).hexdigest() for n in names
                    if n == 'bench.sh' or (n.endswith('.py') and not n.startswith('tools/hip/'))}
            guards.require(len(pins) == d['source_count'] and
                hashlib.sha256(json.dumps(pins,sort_keys=True).encode()).hexdigest() ==
                d['source_aggregate_sha256'], 'unrecoverable benchmark source identity')
            source_sets.append(pins)
            requests = d['requests']
            guards.require(len(requests) == 4 and all(r['run'] == i and r['ok'] and r['eligible'] and
                not r['eligibility_errors'] and r['stream_complete'] and r['thinking_requested'] and
                r['reasoning_observed'] and r['actual_prompt_tokens'] == r['actual_read_tokens'] == 78451 and
                r['actual_reused_tokens'] == 0 and r['generated_tokens'] == 24 and
                math.isfinite(r['prompt_tps']) and r['prompt_tps'] > 0 for i,r in enumerate(requests)),
                'ineligible or unmatched request geometry')
            guards.require([r['request_sha256'] for r in requests] ==
                           [r['request_sha256'] for r in current['requests']], 'request bytes differ')
            routes = d['gu_route']['per_request_counts']
            guards.require(len(routes) == 4 and all(
                (int(r['fused_jobs']) > 0 if candidate else int(r['fused_jobs']) == 0) and
                int(r['separate_jobs']) > 0 and int(r['min_rows']) == 2334 for r in routes),
                'actual GU dispatch differs from selected arm')
            values = [r['prompt_tps'] for r in requests]
            arms.append(dict(label=('B' if candidate else 'A') + str(d.get('cohort_index',0)),
                receipt_commit=receipt_commit, receipt_path=path, source_commit=source,
                first_use_tps=values[0], subsequent_tps=values[1:],
                subsequent_median=statistics.median(values[1:]),
                subsequent_sample_SD=statistics.stdev(values[1:])))
        # The cohort0->1 extension changed only this owner/receipt controller.
        # The actual executed bench.sh, server and workload helpers must match.
        varying = sorted(n for n in set().union(*source_sets)
                         if len({s.get(n) for s in source_sets}) != 1)
        guards.require(varying == ['tools/sycl/bench_engine.py'],
                       'unexpected cross-cohort benchmark source change')
        for i,j in ((0,1),(2,3),(4,5)):
            guards.require(source_sets[i] == source_sets[j], 'paired source snapshots differ')
        pairs, ratios = [], []
        for cohort in range(3):
            a,b = next(x for x in arms if x['label']=='A'+str(cohort)),next(x for x in arms if x['label']=='B'+str(cohort))
            matched = [y/x for x,y in zip(a['subsequent_tps'],b['subsequent_tps'])]
            ratios.extend(matched)
            pairs.append(dict(cohort=cohort,base_median=a['subsequent_median'],
                candidate_median=b['subsequent_median'],
                median_gain_percent=100*(b['subsequent_median']/a['subsequent_median']-1),
                matched_subsequent_speed_ratios=matched))
        geomean = math.exp(statistics.mean(math.log(x) for x in ratios))
        passed = all(p['median_gain_percent'] > 0 for p in pairs) and geomean > 1.002
        current['independent_process_comparison'] = dict(arms=arms,pairs=pairs,
            matched_subsequent_speed_geomean=geomean,geomean_gain_percent=100*(geomean-1),
            all_nine_matched_inputs_faster=all(x>1 for x in ratios),
            exact_committed_source_recovered=True,matched_config_workloads_and_pinned_runtime_set=True,
            runtime_verification_scope='Matches only the runtime_elfs guarded by each historical launch, not a complete loaded-library census.',
            allowed_cross_cohort_source_change=varying,performance_selection_gate_passed=passed,
            decision='Retain explicit large-job fusion for further qualification; default stays off.' if passed else
                     'Retain separate route; prospective independent performance gate failed.',
            limits='Three independent process pairs, four fresh78451-token requests each. First use excluded from '
                   'selection. Individual full receipts remain at the original reachable commits.24-token generation '
                   'does not qualify sustained decode, broad task quality or physical262144/RESTORE.')
        (P/path).write_text(json.dumps(current,indent=2)+'\n')
        print(json.dumps(current['independent_process_comparison']))
    return 0


def summarize_source_view():
    """Analyze three frozen process pairs; first use remains separate evidence."""
    from build_engine import blobs
    path = str((A / 'prefill-long-input-current.json').relative_to(P))

    def git(*args):
        return subprocess.check_output(['git', *args], cwd=P)

    def canonical(value, root):
        if isinstance(value, str):
            return value.replace(root, '$RUN')
        if isinstance(value, list):
            return [canonical(x, root) for x in value]
        if isinstance(value, dict):
            return {k: canonical(v, root) for k, v in value.items()
                    if k != 'STRATA_SYCL_PREFILL_EXPERT_SOURCE_VIEW'}
        return value

    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guards.no_model_owners([])
        head = git('rev-parse', 'HEAD').decode().strip()
        for item in (Path(__file__).resolve(), P / path):
            guards.require(item.read_bytes() == git('show', head + ':' + str(item.relative_to(P))),
                           'commit the analyzer and final cohort before summary')
        current = json.loads((P / path).read_text())
        guards.require(current['mode'] == 'qsa-long-input-source-view-candidate' and
            current['cohort_index'] == 2 and 'independent_source_view_comparison' not in current,
            'summarize the completed final source-view cohort exactly once')
        receipt_revision = git('log', '-1', '--format=%H', '--', path).decode().strip()
        guards.require(json.loads(git('show', receipt_revision + ':' + path)) == current,
                       'final cohort does not match its original result commit')
        original_build = json.loads(git('show', receipt_revision + ':' + str(BUILD.relative_to(P))))
        full_build = json.loads((B / 'main-native-gdn-engine/record.json').read_text())
        guards.require(all(full_build[k] == original_build[k] for k in (
            'source_commit', 'binary', 'compiler', 'configure_argv', 'environment', 'recipe_reference',
            'compile_database_sha256', 'configured_compile_entries', 'targets', 'fixtures')) and
            full_build['complete'] and full_build['passed'] and full_build['ownership_closed'] and
            not full_build['active'] and full_build['binary'] == current['binary'] and
            full_build['source_commit'] == current['build_source_commit'],
            'original build receipt or measured executable identity differs')
        input_sources = [full_build['project'], *full_build['dependencies']]
        guards.require(len(input_sources) == len(original_build['inputs']), 'incomplete build source sets')
        recovered_build_inputs = 0
        for source, recorded in zip(input_sources, original_build['inputs']):
            pins = source['source_sha256']
            recovered = {n: hashlib.sha256(data).hexdigest() for n, data in
                         blobs(Path(source['root']), source['revision'], sorted(pins))}
            guards.require({k: v for k, v in source.items() if k != 'source_sha256'} == recorded and
                recovered == pins and len(pins) == recorded['count'] and
                hashlib.sha256(json.dumps(pins, sort_keys=True, separators=(',', ':')).encode()).hexdigest() ==
                recorded['aggregate_sha256'], 'original engine/dependency source is not recoverable')
            recovered_build_inputs += len(pins)
        recipe = full_build['recipe_reference']
        guards.require(hashlib.sha256(git('show', full_build['project']['revision'] + ':' +
            recipe['path'])).hexdigest() == recipe['sha256'], 'original build recipe is not recoverable')
        records, revision, seen = [], receipt_revision, set()
        while len(records) < 6:
            guards.require(revision not in seen, 'cyclic source-view cohort history')
            seen.add(revision)
            guards.require(subprocess.run(['git', 'merge-base', '--is-ancestor', revision, head],
                cwd=P, check=False).returncode == 0, 'unreachable original cohort')
            record = json.loads(git('show', revision + ':' + path))
            guards.require(record['mode'] in ('qsa-long-input-source-view-base',
                'qsa-long-input-source-view-candidate') and record['passed'] and record['complete'] and
                record['ownership_closed'] and not record['active'] and
                record['exit_code'] == record['engine_exit_code'] == 0 and
                not record['fresh_faults'] and record['boot_unchanged'], 'failed or incomplete source-view cohort')
            records.append((revision, record))
            revision = record['preceding_long_input_result']['commit']
        records.reverse()
        expected = [('base', 0), ('candidate', 0), ('candidate', 1), ('base', 1), ('base', 2), ('candidate', 2)]
        guards.require([(d['mode'].rsplit('-', 1)[1], d['cohort_index']) for _, d in records] == expected,
                       'source-view process order differs from frozen plan')
        guards.require(len({(d['native_identity']['pid'], d['native_identity']['start_ticks'])
                            for _, d in records}) == 6, 'six independent native processes required')
        guards.require(digest(current['binary']['path']) == current['binary']['sha256'],
                       'measured engine binary changed before analysis')
        common, source_set, arms = None, None, []
        for receipt_commit, d in records:
            root = d['environment']['STRATA_MEASUREMENT_ROOT']
            candidate = d['mode'].endswith('-candidate')
            guards.require(d['environment']['STRATA_SYCL_PREFILL_EXPERT_SOURCE_VIEW'] ==
                           ('1' if candidate else '0'), 'wrong source-view selector')
            command = canonical(d['command'], root)
            command[command.index('--port') + 1] = '$PORT'
            comparison = {k: v for k, v in d['source_view_comparison'].items() if k != 'enabled'}
            identity = dict(binary=d['binary'], build_source=d['build_source_commit'],
                profile=d['profile_sha256'], reference=d['reference'],
                environment=canonical(d['environment'], root), config=canonical(d['config'], root),
                command=command, prospective_comparison=comparison,
                prospective_long_input_screen=d['long_input_screen'])
            common = identity if common is None else common
            guards.require(identity == common and d['source_view_comparison']['enabled'] == candidate,
                           'unmatched source-view inputs or prospective criteria')
            source = d['source_commit']
            guards.require(subprocess.run(['git', 'merge-base', '--is-ancestor', source, head],
                cwd=P, check=False).returncode == 0, 'unreachable original benchmark source')
            names = git('ls-tree', '-r', '--name-only', source, '--', 'serve', 'tools', 'bench.sh').decode().splitlines()
            names = [n for n in names if n == 'bench.sh' or
                     (n.endswith('.py') and not n.startswith('tools/hip/'))]
            pins = {n: hashlib.sha256(data).hexdigest() for n, data in blobs(P, source, names)}
            guards.require(len(pins) == d['source_count'] and
                hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest() ==
                d['source_aggregate_sha256'], 'unrecoverable original benchmark snapshot')
            source_set = pins if source_set is None else source_set
            guards.require(pins == source_set, 'frozen benchmark inputs changed between processes')
            requests = d['requests']
            guards.require(len(requests) == 4 and all(r['run'] == i and r['ok'] and r['eligible'] and
                not r['eligibility_errors'] and r['stream_complete'] and r['thinking_requested'] and
                r['reasoning_observed'] and r['actual_prompt_tokens'] == r['actual_read_tokens'] == 117489 and
                r['actual_reused_tokens'] == 0 and r['generated_tokens'] == 24 and
                math.isfinite(r['prompt_tps']) and r['prompt_tps'] > 0 for i, r in enumerate(requests)),
                'ineligible source-view request')
            guards.require([r['request_sha256'] for r in requests] ==
                           [r['request_sha256'] for r in current['requests']], 'request bytes differ')
            routes = d['expert_source_route']
            guards.require(len(routes) == 4 and all(
                int(r['requested_view']) == int(r['effective_view']) == int(candidate) and
                int(r['layers']) == (48 if candidate else 0) and
                int(r['owned_slab_bytes']) == (0 if candidate else 1363148800) and
                int(r['borrowed_layer_span_bytes']) == (50292326400 if candidate else 0) and
                r['host_copy'] == ('eliminated' if candidate else 'synchronous') and
                r['source_owner'] == 'caller_through_checked_retirement' for r in routes),
                'source-view route or original fallback differs')
            values = [r['prompt_tps'] for r in requests]
            median = statistics.median(values[1:])
            guards.require(median == d['long_input_process_summary']['median_prompt_tps'],
                           'process summary differs from original samples')
            totals = d['expert_bank_route']['per_request_totals']
            guards.require(len(totals) == 4 and all(int(t['requested_H2D_bytes']) > 0 and
                t['final_drain'] == 'checked' for t in totals), 'incomplete expert transfer accounting')
            arms.append(dict(label=('B' if candidate else 'A') + str(d['cohort_index']),
                receipt_commit=receipt_commit, receipt_path=path, source_commit=source,
                first_use_tps=values[0], subsequent_tps=values[1:], subsequent_median=median,
                subsequent_sample_SD=statistics.stdev(values[1:]),
                requested_expert_H2D_bytes=[int(t['requested_H2D_bytes']) for t in totals],
                output_hashes=[r['output_sha256'] for r in requests],
                reasoning_hashes=[r['reasoning_sha256'] for r in requests]))
        pairs = []
        for cohort in range(3):
            a = next(x for x in arms if x['label'] == 'A' + str(cohort))
            b = next(x for x in arms if x['label'] == 'B' + str(cohort))
            pairs.append(dict(cohort=cohort, base_median=a['subsequent_median'],
                candidate_median=b['subsequent_median'], median_ratio=b['subsequent_median']/a['subsequent_median'],
                first_use_ratio=b['first_use_tps']/a['first_use_tps'],
                requested_payload_matches=a['requested_expert_H2D_bytes'] == b['requested_expert_H2D_bytes'],
                output_hashes_match=a['output_hashes'] == b['output_hashes'],
                reasoning_hashes_match=a['reasoning_hashes'] == b['reasoning_hashes']))
        geomean = statistics.geometric_mean(p['median_ratio'] for p in pairs)
        selected = all(p['median_ratio'] > 1 for p in pairs) and geomean > 1.005
        candidate_medians = [a['subsequent_median'] for a in arms if a['label'].startswith('B')]
        current['independent_source_view_comparison'] = dict(analyzer_source_commit=head, arms=arms, pairs=pairs,
            independent_units='Three process pairs; the three subsequent requests within a process are repeated samples, not independent processes.',
            paired_median_speed_geomean=geomean, geomean_gain_percent=100*(geomean-1),
            performance_selection_gate_passed=selected,
            requested_expert_payload_matches_in_all_pairs=all(p['requested_payload_matches'] for p in pairs),
            candidate_process_medians=candidate_medians, prefill1000_target_passed=all(x >= 1000 for x in candidate_medians),
            exact_original_benchmark_inputs_recovered=True, frozen_input_count=len(source_set),
            exact_original_build_inputs_recovered=recovered_build_inputs,
            original_build_source_commit=full_build['source_commit'], original_recipe=recipe,
            matched_engine_configuration_workloads_and_guarded_runtime=True,
            original_outcomes_unchanged=True, GPU_executed=False, model_executed=False,
            decision='Retain immutable source views as an opt-in for further qualification; default stays off.' if selected else
                     'Retain synchronous source copying; prospective performance selection failed.',
            limits='First-use samples, including regressions, remain separate and unfiltered. Requested bytes are not wire traffic. Output hashes are descriptive and not an exact-text acceptance criterion.24-token generation does not qualify sustained70tok/s decode, broad task quality,physical262144 or session/RESTORE.')
        (P / path).write_text(json.dumps(current, indent=2) + '\n')
        print(json.dumps({k: current['independent_source_view_comparison'][k] for k in (
            'pairs', 'paired_median_speed_geomean', 'geomean_gain_percent', 'performance_selection_gate_passed',
            'candidate_process_medians', 'prefill1000_target_passed', 'decision')}))
    return 0


def reanalyze_staged_host():
    """Reconcile a closed parser failure; preserve its original failed outcome."""
    from build_engine import blobs
    path = A / 'prefill-host-attribution-current.json'
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip()
        for item in (path, Path(__file__).resolve()):
            guards.require(item.read_bytes() == subprocess.check_output(
                ['git', 'show', revision + ':' + str(item.relative_to(P))], cwd=P),
                'commit original failed result and corrected analyzer first')
        record = json.loads(path.read_text())
        guards.require(record['mode'] == 'qsa-long-input-expert-bank-host-census' and
            record['complete'] and record['ownership_closed'] and not record['active'] and
            not record['passed'] and record.get('error') == 'incomplete mixed owner/phase census' and
            not record['fresh_faults'] and record['boot_unchanged'] and
            record['exit_code'] == record['engine_exit_code'] == 0 and
            'corrected_resource_analysis' not in record, 'not the closed specified parser failure')
        for name, expected in record['read_only_reanalysis_inputs'].items():
            guards.require(Path(name).stat().st_size == expected['bytes'] and
                           digest(name) == expected['sha256'], 'captured input changed: ' + name)
        root = Path(record['environment']['STRATA_MEASUREMENT_ROOT'])
        original = json.loads((root / 'record.json').read_text())
        guards.require(original['ownership_closed'] and not original['active'] and
            all(supervision.closed(c) for c in original['owned_commands']), 'original ownership not closed')
        plan = json.loads((root / 'plan.json').read_text())
        for name, raw in blobs(P, plan['source_commit'], plan['source_sha256']):
            guards.require(hashlib.sha256(raw).hexdigest() == plan['source_sha256'][name],
                           'original benchmark input not recoverable: ' + name)
        guards.require(plan['binary'] == record['binary'] and digest(record['binary']['path']) ==
                       record['binary']['sha256'], 'original engine identity changed')
        requests = record['requests']
        guards.require(len(requests) == 2 and all(q['ok'] and q['eligible'] and q['stream_complete'] and
            q['actual_prompt_tokens'] == q['actual_read_tokens'] == 117489 and
            q['actual_reused_tokens'] == 0 and q['thinking_requested'] and q['reasoning_observed']
            for q in requests), 'original request eligibility differs')
        log = (root / 'engine.stderr').read_text(errors='replace')
        guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:', log),
                       'original runtime error is not a parser-only failure')
        parts = split_long_request_logs(log, 2)
        pairs = ((0, 81920), (81920, 35568))
        owners = [extract_mixed_owners(part, pairs, 1059131904) for part in parts]
        profiles = [extract_resource_profile(part, True, False, '', 4, pairs) for part in parts]
        negatives = []

        def reject(label, action):
            try:
                action()
            except RuntimeError:
                negatives.append(label)
                return
            raise RuntimeError('negative parser check unexpectedly accepted: ' + label)

        reject('combined_requests_are_not_one96_window_interval',
               lambda: extract_resource_profile(log, True, False, '', 4, pairs))
        first = parts[0].splitlines(keepends=True)
        census = next(i for i, line in enumerate(first) if line.startswith('strata MoE planned job census:'))
        reject('missing_planned_window', lambda: extract_resource_profile(
            ''.join(first[:census] + first[census + 1:]), True, False, '', 4, pairs))
        entry = next(i for i, line in enumerate(first) if line.startswith('strata MoE mixed owners: phase=entry'))
        reject('missing_owner_entry', lambda: extract_mixed_owners(
            ''.join(first[:entry] + first[entry + 1:]), pairs, 1059131904))
        changed = re.sub(r'(?m)^(strata MoE GU fusion: fused_jobs=)(\d+)',
            lambda m: m[1] + str(int(m[2]) + 1), parts[0], count=1)
        reject('changed_GU_work_total', lambda: extract_resource_profile(changed, True, False, '', 4, pairs))
        cold, warm = [compact_host_resource_profile(profile) for profile in profiles]
        record['corrected_resource_analysis'] = dict(source_commit=revision,
            original_failed_result=dict(commit=revision, path=str(path.relative_to(P))),
            complete=True, passed=True, model_executed=False, GPU_executed=False,
            original_outcome_unchanged=True, negative_parser_checks=negatives,
            per_request_owner_windows=[x['retired_windows'] for x in owners],
            all_owner_keys_extents_and_retirements_checked=True,
            first_use=cold, subsequent=warm,
            profiled_prompt_tps=[q['prompt_tps'] for q in requests],
            limits='Offline accounting reanalysis only: original passed=false/parser error remain. Two completed requests, not independent processes or clean speed. No physical bus traffic, arithmetic saturation, model quality/full-context/RESTORE adoption.')
        path.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(original_passed=record['passed'], analysis_passed=True,
            owner_windows=[x['retired_windows'] for x in owners], negative_checks=negatives)))
    return 0


def main():
    if sys.argv[1:] == ['reanalyze-staged-host']:
        return reanalyze_staged_host()
    if sys.argv[1:] == ['summarize-gu']:
        return summarize_gu()
    if sys.argv[1:] == ['summarize-source-view']:
        return summarize_source_view()
    modes = ('check', 'base', 'candidate', 'qsa-check', 'qsa-base', 'qsa-direct', 'qsa-census',
             'qsa-gdncols4-warm-census',
             'qsa-gdncols4-host-check', 'qsa-gdncols4-host-traced-check', 'qsa-gdncols4-host-census',
             'qsa-qcache-check', 'qsa-qcache-base', 'qsa-qcache',
             'qsa-kvec4-check', 'qsa-kvec4-base', 'qsa-kvec4', 'qsa-kvec4-census',
             'qsa-kvvec4-check', 'qsa-kvvec4-base', 'qsa-kvvec4',
             'qsa-ordered-exp-check', 'qsa-ordered-exp-base', 'qsa-ordered-exp-candidate',
             'qsa-gu-fusion-check', 'qsa-gu-fusion-base', 'qsa-gu-fusion-candidate',
             'qsa-long-input-check', 'qsa-long-input',
             'qsa-long-input-balanced-check', 'qsa-long-input-balanced',
             'qsa-long-input-scratch-check', 'qsa-long-input-scratch',
             'qsa-long-input-expert-bank-check', 'qsa-long-input-expert-bank',
             'qsa-long-input-expert-bank-64k-check', 'qsa-long-input-expert-bank-64k',
             'qsa-long-input-expert-bank-host-check', 'qsa-long-input-expert-bank-host-census',
             'qsa-long-input-source-view-check', 'qsa-long-input-source-view-base',
             'qsa-long-input-source-view-candidate',
             'qsa-kvvec4-timing-check', 'qsa-kvvec4-census') + tuple(
                 f'qsa-gdncols{n}-{phase}' for n in (2, 4, 8) for phase in ('check', 'base', 'candidate'))
    guards.require(len(sys.argv) in (3, 4) and sys.argv[1] in ('prepare', 'run') and
                   sys.argv[2] in modes, 'use prepare|run ' + '|'.join(modes))
    action, mode = sys.argv[1:3]
    cohort = int(sys.argv[3]) if len(sys.argv) == 4 else 0
    guards.require(cohort in (0, 1, 2) and (len(sys.argv) == 3 or
        mode in ('qsa-gu-fusion-base', 'qsa-gu-fusion-candidate', 'qsa-long-input', 'qsa-long-input-balanced', 'qsa-long-input-scratch', 'qsa-long-input-expert-bank', 'qsa-long-input-expert-bank-64k',
                 'qsa-long-input-source-view-base', 'qsa-long-input-source-view-candidate')),
        'optional cohort0|1|2 is for GU paired arms or independent long-input processes')
    column_mode = re.fullmatch(r'qsa-gdncols(2|4|8)-(check|base|candidate|warm-census|host-check|host-traced-check|host-census)', mode)
    ordered_trial = mode.startswith('qsa-ordered-exp-')
    long_trial = mode.startswith('qsa-long-input')
    source_view_trial = mode.startswith('qsa-long-input-source-view-')
    source_view_on = source_view_trial and not mode.endswith('-base')
    balanced_placement = mode in ('qsa-long-input-balanced-check', 'qsa-long-input-balanced')
    staged_host = mode == 'qsa-long-input-source-view-check' or mode in ('qsa-long-input-expert-bank-host-check', 'qsa-long-input-expert-bank-host-census')
    single_window = mode in ('qsa-long-input-expert-bank-64k-check', 'qsa-long-input-expert-bank-64k')
    staged_bank = source_view_trial or staged_host or single_window or mode in ('qsa-long-input-expert-bank-check', 'qsa-long-input-expert-bank')
    long_minimum = 65536 if single_window else 98304
    bank_residual_rows = 78450 if single_window else 117488
    bank_window_rows = (78450,) if single_window else (81920, 35568)
    mixed_tail_offset = 1022974976 if single_window else 1059131904
    mixed_tail_bytes = 66119680 if single_window else 83886080
    scratch_mixed = staged_bank or mode in ('qsa-long-input-scratch-check', 'qsa-long-input-scratch')
    long_clean_mode = ('qsa-long-input-expert-bank-64k' if single_window else
                      'qsa-long-input-expert-bank' if staged_bank else 'qsa-long-input-scratch' if scratch_mixed else
                       'qsa-long-input-balanced' if balanced_placement else 'qsa-long-input')
    gu_trial = mode.startswith('qsa-gu-fusion-')
    gu_candidate = long_trial or (gu_trial and mode != 'qsa-gu-fusion-base')
    ordered_candidate = long_trial or gu_trial or (ordered_trial and mode != 'qsa-ordered-exp-base')
    host_profile = staged_host or mode.startswith('qsa-gdncols4-host-')
    warm_profile = mode in ('qsa-gdncols4-warm-census', 'qsa-gdncols4-host-census',
                           'qsa-long-input-expert-bank-host-census')
    column_trial = column_mode is not None or ordered_trial or gu_trial or long_trial
    column_check = staged_host or mode in ('qsa-long-input-expert-bank-64k-check', 'qsa-long-input-expert-bank-check', 'qsa-long-input-scratch-check', 'qsa-long-input-balanced-check', 'qsa-long-input-check', 'qsa-ordered-exp-check', 'qsa-gu-fusion-check') or (column_mode is not None and column_mode[2] == 'check')
    column_partitions = 4 if ordered_trial or gu_trial or long_trial else int(column_mode[1]) if column_mode is not None and column_mode[2] != 'base' else 0
    qsa = mode.startswith('qsa-')
    short_profile = mode == 'qsa-kvvec4-timing-check' or (host_profile and not warm_profile)
    phase_profile = host_profile or warm_profile or short_profile or mode == 'qsa-kvvec4-census'
    command_profile = not staged_host and ((phase_profile and mode != 'qsa-gdncols4-host-check') or
        mode in ('qsa-kvec4-check', 'qsa-kvec4-census'))
    census = phase_profile or mode in ('qsa-census', 'qsa-kvec4-census')
    qcache = mode.startswith('qsa-qcache')
    kvvec4 = column_trial or mode.startswith('qsa-kvvec4')
    kvec4 = kvvec4 or mode.startswith('qsa-kvec4')
    check = census or column_check or mode in ('check', 'qsa-check', 'qsa-qcache-check', 'qsa-kvec4-check', 'qsa-kvvec4-check')
    request_count = 2 if warm_profile else 1 if check else 4
    chunk64 = mode in ('check', 'candidate')
    generation = 'v2' if short_profile or mode == 'qsa-kvec4-check' else 'v1'
    if cohort:
        generation += '-repeat' + str(cohort)
    root = B / (('qsa-attention-model-' + mode[4:] + '-' + generation) if qsa else
                ('native-gdn-model-' + mode + '-v2'))
    result_path = (A / 'prefill-long-input-current.json' if source_view_trial else
                   A / 'prefill-host-attribution-current.json' if host_profile else
                   A / 'prefill-long-input-current.json' if long_trial else
                   A / 'moe-gu-swiglu-model-current.json' if gu_trial else
                   A / 'qsa-ordered-exp-model-current.json' if ordered_trial else
                   A / 'moe-window-census-current.json' if census else
                   A / 'gdn-columns-model-current.json' if column_trial else
                   A / 'qsa-kvvec4-model-current.json' if kvvec4 else
                   A / 'qsa-kvec4-model-current.json' if kvec4 else
                   A / 'qsa-cached-query-model-current.json' if qcache else
                   A / 'qsa-attention-model-current.json' if qsa else RESULT)
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reference = json.loads(BASE.read_text())
        guards.no_model_owners(list(reference['model_identity']))
        guards.driver_guard()
        if action == 'prepare':
            guards.require(not root.exists(), 'use a new committed measurement revision after closing this run')
            build = json.loads(BUILD.read_text())
            guards.require(build['passed'] and build['complete'] and build['ownership_closed'] and not build['active'],
                           'qualified closed engine build required')
            profile_route_qualification = None
            if staged_host and not source_view_trial:
                revision = subprocess.check_output(['git', 'rev-parse', '692e2658d'], cwd=P, text=True).strip()
                path = 'bench/results/2026-10-15-pr6-recovery-audit/prefill-long-input-current.json'
                data = subprocess.check_output(['git', 'show', revision + ':' + path], cwd=P)
                qualified = json.loads(data)
                comparison = qualified.get('independent_process_comparison', {})
                guards.require(qualified['complete'] and qualified['passed'] and qualified['ownership_closed'] and
                    qualified['binary'] == build['binary'] and qualified['build_source_commit'] == build['source_commit'] and
                    comparison.get('source_binary_configuration_parity') and len(comparison.get('cohorts', [])) == 3,
                    'current117K source/binary no longer matches the closed three-process route')
                profile_route_qualification = dict(commit=revision, path=path,
                    sha256=hashlib.sha256(data).hexdigest(), limits='Validated opt-in117K route, not1000tok/s or full-model adoption.')
            ordered_qualification = None
            gu_qualification = None
            mixed_qualification = None
            if scratch_mixed:
                component_path = A / 'moe-mixed-scratch-current.json'
                component = json.loads(component_path.read_text())
                guards.require(component_path.read_bytes() == subprocess.check_output(
                    ['git', 'show', 'HEAD:' + str(component_path.relative_to(P))], cwd=P) and
                    component['mode'] == 'validate' and component['passed'] and component['complete'] and
                    component['ownership_closed'] and component['GPU_executed'] and
                    {x['case'] for x in component['checks']} ==
                        {'plan', 'smoke', 'contiguous', '65537', 'partial', 'full'},
                    'matching committed two-bank capacity and bit-copy gates required')
                fixture = build['fixtures']['sycl_moe_accum_test']
                guards.require(fixture['sha256'] == component['fixture']['sha256'] and
                    digest(fixture['path']) == fixture['sha256'], 'qualified mixed fixture bytes changed')
                source_pins = {}
                for name in ('include/strata/prefill/moe_accum_plan.hpp',
                             'include/strata/prefill/native16.hpp','src/prefill/sycl/native16.cpp',
                             'tests/sycl/moe_accum.cpp'):
                    data = subprocess.check_output(['git','show',component['build_source_commit']+':'+name],cwd=P)
                    source_pins[name] = hashlib.sha256(data).hexdigest()
                    guards.require(digest(P/name)==source_pins[name], 'qualified mixed implementation changed: '+name)
                mixed_qualification = dict(component_path=str(component_path.relative_to(P)),
                    component_sha256=digest(component_path), fixture=component['fixture'],
                    current_fixture=fixture, kernel_source_sha256=source_pins,
                    evidence='Prior completed numerical/bit-copy checks reused for identical executable bytes and unchanged tested kernel/plan source; this is not a new GPU run.',
                    limits='Component capacity/gather only; this model check must establish actual Prefill owners, bank phases and callback retirement.')
            bank_qualification = None
            if staged_bank:
                component_path = A/'moe-expert-bank-current.json'
                component = json.loads(component_path.read_text())
                gpu = component.get('gpu',{})
                guards.require(component_path.read_bytes()==subprocess.check_output(
                    ['git','show','HEAD:'+str(component_path.relative_to(P))],cwd=P) and
                    component['complete'] and component['passed'] and component['ownership_closed'] and
                    gpu.get('mode')=='validate' and gpu.get('complete') and gpu.get('passed') and
                    gpu.get('ownership_closed') and gpu['build_source_commit']==build['source_commit'] and
                    gpu['fixture']==build['fixtures']['sycl_expert_bank_test'],
                    'matching committed production expert-bank upload/byte/unwind gates required')
                guards.require(digest(gpu['fixture']['path'])==gpu['fixture']['sha256'],'bank fixture changed')
                bank_qualification = dict(component_path=str(component_path.relative_to(P)),
                    component_sha256=digest(component_path),build_source_commit=gpu['build_source_commit'],
                    limits=gpu['limits'])
            source_view_qualification = None
            if source_view_trial:
                guards.require(gpu.get('provider_checks_passed') is True and
                    gpu.get('provider_fixture') == build['fixtures'].get('file_expert_source_test'),
                    'matching production Arena private/shared/fallback checks required')
                index_path = A/'research-index.json'
                index = json.loads(index_path.read_text())['immutable_arena_layer_view_candidate_scope']
                guards.require(index_path.read_bytes()==subprocess.check_output(
                    ['git','show','HEAD:'+str(index_path.relative_to(P))],cwd=P) and
                    index.get('main_gpu_admission',{}).get('source_contract_cleared') is True,
                    'commit independent resident-source lifetime review and Main resolution before model work')
                source_view_qualification = index['main_gpu_admission']
                guards.require(all(digest(P/name)==expected for name,expected in
                    source_view_qualification['source_files_sha256'].items()), 'reviewed source-view implementation changed')
            if gu_trial or long_trial:
                component_path = A / 'moe-gu-swiglu-current.json'
                component = json.loads(component_path.read_text())
                guards.require(component['mode'] == 'perf' and component['passed'] and component['ownership_closed'] and
                    component['component_performance_gate_passed'], 'GU component numerical and performance gates required')
                source_pins = {}
                for name in ('src/prefill/sycl/vllm_xe2/native16.cpp', 'src/prefill/sycl/vllm_xe2/native16.hpp',
                             'src/prefill/sycl/vllm_xe2/xe_gemm_bsd.hpp', 'src/prefill/sycl/native16.cpp'):
                    original = subprocess.check_output(['git','show',component['source_commit'] + ':' + name],cwd=P)
                    source_pins[name] = hashlib.sha256(original).hexdigest()
                    guards.require(digest(P/name) == source_pins[name], 'qualified GU kernel source changed: ' + name)
                gu_qualification = dict(component_path=str(component_path.relative_to(P)), component_sha256=digest(component_path),
                    component_source_commit=component['source_commit'], kernel_source_sha256=source_pins, component_gate_passed=True,
                    selector='large: matrix-wide jobs>=2334 rows; smaller jobs retain separate GU/SwiGLU')
                if gu_trial and not mode.endswith('-check'):
                    prior = json.loads(result_path.read_text())
                    first, second = ('base', 'candidate') if cohort % 2 == 0 else ('candidate', 'base')
                    current = mode.rsplit('-', 1)[1]
                    if current == second:
                        required, preceding_cohort = 'qsa-gu-fusion-' + first, cohort
                    elif cohort:
                        required, preceding_cohort = 'qsa-gu-fusion-' + ('candidate' if (cohort - 1) % 2 == 0 else 'base'), cohort - 1
                    else:
                        required, preceding_cohort = 'qsa-gu-fusion-check', 0
                    guards.require(result_path.read_bytes() == subprocess.check_output(
                        ['git','show','HEAD:' + str(result_path.relative_to(P))],cwd=P) and
                        prior['mode'] == required and prior.get('cohort_index', 0) == preceding_cohort and
                        prior['passed'] and prior['ownership_closed'] and prior['binary'] == build['binary'],
                        'commit matching preceding GU model phase')
                    gu_qualification['preceding_result'] = dict(commit=subprocess.check_output(
                        ['git','rev-parse','HEAD'],cwd=P,text=True).strip(),path=str(result_path.relative_to(P)),mode=prior['mode'])
            previous_screen = None
            if long_trial and check and result_path.exists():
                prior = json.loads(result_path.read_text())
                guards.require(result_path.read_bytes() == subprocess.check_output(
                    ['git', 'show', 'HEAD:' + str(result_path.relative_to(P))], cwd=P) and
                    prior['complete'] and prior['ownership_closed'] and not prior['active'],
                    'preserve the closed preceding screen before a new configuration')
                previous_screen = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip(),
                    path=str(result_path.relative_to(P)), mode=prior['mode'], passed=prior['passed'], error=prior.get('error'))
            if long_trial and not check:
                prior = json.loads(result_path.read_text())
                required = long_clean_mode + '-check' if cohort == 0 else long_clean_mode
                preceding_cohort = max(0, cohort-1)
                if source_view_trial:
                    first,second = ('base','candidate') if cohort%2==0 else ('candidate','base')
                    arm = mode.rsplit('-',1)[1]
                    preceding_cohort = cohort if arm==second else max(0,cohort-1)
                    required = ('qsa-long-input-source-view-'+first if arm==second else
                        'qsa-long-input-source-view-'+('candidate' if (cohort-1)%2==0 else 'base') if cohort else
                        'qsa-long-input-source-view-check')
                guards.require(result_path.read_bytes() == subprocess.check_output(
                    ['git', 'show', 'HEAD:' + str(result_path.relative_to(P))], cwd=P) and
                    prior['mode'] == required and prior.get('cohort_index', 0) == preceding_cohort and
                    prior['passed'] and prior['ownership_closed'] and prior['binary'] == build['binary'],
                    'commit matching preceding long-input model phase')
                if cohort:
                    guards.require(not prior.get('long_input_process_summary', {}).get('futility_stop', False),
                        'preceding clean process missed the prospective futility bound; select a new configuration')
                preceding_long = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip(),
                                      path=str(result_path.relative_to(P)), mode=prior['mode'])
            if ordered_trial or long_trial:
                component_path = A / 'qsa-ordered-exp-current.json'
                component = json.loads(component_path.read_text())
                guards.require(component['passed'] and component['complete'] and component['ownership_closed'] and
                    component['component_summary']['advance_to_model'], 'ordered-exp component gate required')
                source_pins = {}
                for name in ('src/kernels/sycl/qsa_bf16_prefix.cpp', 'include/strata/kernels/qsa_bf16_prefix.hpp'):
                    original = subprocess.check_output(['git', 'show', component['source_commit'] + ':' + name], cwd=P)
                    source_pins[name] = hashlib.sha256(original).hexdigest()
                    guards.require(digest(P / name) == source_pins[name], 'qualified QSA kernel source changed: ' + name)
                ordered_qualification = dict(component_path=str(component_path.relative_to(P)),
                    component_sha256=digest(component_path), kernel_source_sha256=source_pins,
                    component_source_commit=component['source_commit'], component_gate_passed=True)
                if ordered_trial and mode != 'qsa-ordered-exp-check':
                    prior = json.loads(result_path.read_text())
                    committed_prior = subprocess.check_output(['git', 'show', 'HEAD:' + str(result_path.relative_to(P))], cwd=P)
                    required = 'qsa-ordered-exp-check' if mode.endswith('-base') else 'qsa-ordered-exp-base'
                    guards.require(committed_prior == result_path.read_bytes() and prior['mode'] == required and
                        prior['passed'] and prior['ownership_closed'] and prior['binary'] == build['binary'],
                        'commit the matching preceding ordered-exp model gate')
                    ordered_qualification['preceding_result'] = dict(
                        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip(),
                        path=str(result_path.relative_to(P)), mode=prior['mode'], passed=True)
            if host_profile:
                cpu_check = json.loads((A / 'prefill-host-cpu-current.json').read_text())
                guards.require(cpu_check['passed'] and cpu_check['ownership_closed'] and
                    cpu_check['source_sha256']['include/strata/sycl/cpu_wait_profile.hpp'] ==
                        digest(P / 'include/strata/sycl/cpu_wait_profile.hpp'), 'matching CPU accounting gates required')
                qualification = []
                if mode not in ('qsa-gdncols4-host-check', 'qsa-long-input-expert-bank-host-check',
                                'qsa-long-input-source-view-check'):
                    prior_bytes = subprocess.check_output(['git', 'show',
                        'HEAD:' + str(result_path.relative_to(P))], cwd=P)
                    guards.require(prior_bytes == result_path.read_bytes(), 'commit the previous host model gate')
                    prior = json.loads(result_path.read_text())
                    required = ('qsa-long-input-expert-bank-host-check' if staged_host else
                                'qsa-gdncols4-host-traced-check' if warm_profile else 'qsa-gdncols4-host-check')
                    guards.require(prior['mode'] == required and prior['passed'] and prior['ownership_closed'] and
                                   prior['binary'] == build['binary'], 'matching previous host model gate required')
                    qualification = prior.get('host_qualification', []) + [dict(
                        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip(),
                        path=str(result_path.relative_to(P)), mode=prior['mode'], binary=prior['binary'], passed=True)]
            for path, expected in reference['model_identity'].items():
                info = Path(path).stat()
                guards.require(info.st_size == expected['bytes'] and info.st_mtime_ns == expected['mtime_ns'], 'model changed')
            for group in ('pack_metadata_sha256', 'MTP_metadata_sha256'):
                for path, expected in reference[group].items():
                    guards.require(digest(path) == expected, 'model/pack metadata changed: ' + path)
            root.mkdir(mode=0o700)
            for name in ('tmp', 'cache'):
                (root / name).mkdir()
            profile = P / 'data/expert-profile.bin'
            shutil.copyfile(profile, root / 'expert-profile.bin')
            env = dict(reference['environment'])
            for key in list(env):
                if key.startswith('UR_LOG_') or key in ('UR_ENABLE_LAYERS', 'STRATA_TRACE',
                                                      'STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT'):
                    del env[key]
            settings = json.loads(SETTINGS.read_text())
            if check:
                env.update({k: v for k, v in settings['environment'].items() if k.startswith('UR_LOG_')})
                env.update(UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION', STRATA_TRACE='1')
            env.update(PATH='/home/yayoi/.local/share/strata-sycl/venv/bin:' + env['PATH'],
                       TMPDIR=str(root / 'tmp'), XDG_CACHE_HOME=str(root / 'cache'),
                       STRATA_MEASUREMENT_ROOT=str(root), ZES_ENABLE_SYSMAN='0',
                       STRATA_SYCL_PREFILL_GDN_COLUMNS=str(column_partitions),
                       STRATA_SYCL_PREFILL_GDN_CHUNK64='1' if chunk64 else '0')
            env['STRATA_SYCL_PREFILL_GU_SWIGLU'] = 'large' if gu_candidate else '0'
            env['STRATA_SYCL_PREFILL_EXPERT_SOURCE_VIEW'] = '1' if source_view_on else '0'
            if long_trial:
                env['STRATA_SYCL_PREFILL_DOWN_SCATTER'] = '0'
                if balanced_placement:
                    env['STRATA_PREFILL_LAYER_MAJOR_R_GPU'] = '81920'
                if scratch_mixed:
                    env.update(STRATA_SYCL_PREFILL_MIXED_SCRATCH='1',
                        STRATA_PREFILL_LAYER_MAJOR_R_GPU='98304', STRATA_PREFILL_RELEASE_HEADS='0')
                if staged_bank:
                    env.update(STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS='384',
                        STRATA_PREFILL_LAYER_MAJOR_R_GPU='auto')
            if qsa:
                env['STRATA_SYCL_PREFILL_QSA_MATH'] = (
                    'bf16-prefix-ordered-exp' if ordered_candidate else
                    'bf16-prefix-kvvec4' if column_trial or phase_profile or mode in ('qsa-kvvec4-check', 'qsa-kvvec4') else
                    'bf16-prefix-kvec4' if mode in ('qsa-kvec4-check', 'qsa-kvec4', 'qsa-kvec4-census', 'qsa-kvvec4-base') else
                    'bf16-prefix-qcache' if mode in ('qsa-qcache-check', 'qsa-qcache', 'qsa-kvec4-base') else
                    'bf16' if mode in ('qsa-check', 'qsa-direct') else 'bf16-prefix-optimized')
            if census:
                env.update(STRATA_PREFILL_LAYER_MAJOR_PROFILE='1',
                           STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE='1',
                           STRATA_PREFILL_CPU_WAIT_PROFILE='1')
            if phase_profile:
                env['STRATA_PREFILL_LAYER_MAJOR_GDN_PROFILE'] = '1'
            if command_profile:
                env['STRATA_SYCL_PREFILL_COMMAND_PROFILE'] = '1'
            if staged_host:
                # The existing command sampler has no staged-expert-copy tag.
                # Keep it off: an untagged staged copy would invalidate its frame.
                env['STRATA_SYCL_PREFILL_COMMAND_PROFILE'] = '0'
            config = dict(reference['config'])
            flags = list(config['args'])
            flags[flags.index('--expert-profile') + 1] = str(root / 'expert-profile.bin')
            if long_trial:
                guards.require(int(flags[flags.index('--max-context') + 1]) == 131072 and
                    flags[flags.index('--kv') + 1] == 'int8' and
                    int(flags[flags.index('--kv-resident') + 1]) == 131072,
                    'long-input screen requires the recorded131072 resident INT8 context placement')
            config.update(exe=str(P / 'tools/sycl/engine_once.py'), args=flags,
                          log=str(root / 'engine.stderr'), env={})
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config, indent=2) + '\n')
            command = ['/bin/sh', str(P / 'bench.sh'), '--config', str(config_path), '--port', str(port),
                       '--runs', str(request_count), '--prefill-only', '--prefill-tokens',
                       str(long_minimum) if long_trial else '32768' if short_profile or (check and not census) else '65536', '--workload-id', 'native_gdn_20261018',
                       '--think', '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root / 'bench')]
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip()
            files = subprocess.check_output(['git', 'ls-files', '-z', '--', 'serve', 'tools', 'bench.sh'], cwd=P).decode().strip('\0').split('\0')
            pins = {}
            for name in files:
                if name != 'bench.sh' and (not name.endswith('.py') or name.startswith('tools/hip/')):
                    continue
                data = subprocess.check_output(['git', 'show', revision + ':' + name], cwd=P)
                guards.require((P / name).read_bytes() == data, 'uncommitted benchmark input: ' + name)
                pins[name] = hashlib.sha256(data).hexdigest()
            plan = dict(mode=mode, source_commit=revision, build_source_commit=build['source_commit'],
                        binary=build['binary'], environment=env, config=config, command=command,
                        reference=dict(path=str(BASE.relative_to(P)), sha256=digest(BASE)),
                        source_sha256=pins, config_sha256=digest(config_path), profile_sha256=digest(profile),
                        complete=False, passed=False, active=False, model_quality_qualified=False,
                        physical262144_RESTORE_qualified=False, performance_adopted=False)
            if host_profile:
                plan.update(host_qualification=qualification, cpu_observer_version=4,
                            cpu_fixture_receipt_sha256=digest(A / 'prefill-host-cpu-current.json'))
                if staged_host:
                    plan['profile_route_qualification'] = profile_route_qualification
            if ordered_trial or long_trial:
                plan['ordered_exp_qualification'] = ordered_qualification
            if gu_trial or long_trial:
                plan['gu_qualification'] = gu_qualification
                plan['cohort_index'] = cohort
            if long_trial:
                plan['long_input_screen'] = dict(minimum_actual_tokens=long_minimum, generated_token_cap=24,
                    context_capacity=131072, required_client_context=long_minimum * 5 // 4 + 280,
                    question='Amortization at longer input on fixed placement/chunk/GU/QSA routes; not a same-workload kernel gain.',
                    qualification='One validated long request, then up to3 independent clean processes with4 fresh requests each. Separate first use and3 subsequent requests; target each independent subsequent median>=1000tok/s. Stop this configuration early if a process median<950 and all3 subsequent samples<975; retain the failed screen and do not claim3-process qualification. Not sustained decode, model-quality or physical262144 qualification.')
                if balanced_placement:
                    plan['long_input_screen'].update(residual_GPU_rows=81920,
                        question='Test the explicit81920-row residual prefix against the diagnosed automatic-placement cliff; preserve512MiB headroom and head retention. Additional RAM transfers are part of total cost; no same-placement kernel claim.')
                if scratch_mixed:
                    plan['mixed_qualification'] = mixed_qualification
                    plan['long_input_screen'].update(residual_GPU_rows=98304,
                        question='Borrow idle R/shared-arena spans for mixed input, with zero extra mixed allocation and16384 more GPU residual rows than the81920 screen. Preserve512MiB headroom and existing dense banking; heads remain resident. Placement plus storage comparison, not a kernel-only speed claim.')
                if staged_bank:
                    plan['expert_bank_qualification'] = bank_qualification
                    plan['long_input_screen'].update(residual_GPU_rows=bank_residual_rows,expert_slots=384,
                        question='Matched117489-token workload: smaller temporary expert GPU bank admits all117488 engine residual rows, eliminating the host-tail FFN path. Preserve512-expert host slab,512MiB allowance, dense banking and heads. Count extra requested H2D bytes/hits; compare total latency, not capacity arithmetic.')
                    if single_window:
                        plan['long_input_screen'].update(
                            expected_actual_read_tokens=bank_residual_rows + 1,
                            expected_windows_per_layer=list(bank_window_rows),
                            question='65536-minimum official workload, expected78451 fresh read tokens: unchanged1222 engine/384 slots/auto GPU residency with one78450-row MoE window per layer. Compare total latency and requested weight/dequant/job counts; different input size, not a same-input kernel gain or measured hardware saturation.')
                if previous_screen:
                    plan['previous_screen'] = previous_screen
                if not check:
                    plan['preceding_long_input_result'] = preceding_long
            if source_view_trial:
                plan['source_view_qualification'] = source_view_qualification
                plan['source_view_comparison'] = dict(enabled=source_view_on,
                    expected_actual_read_tokens=117489, expected_borrowed_layer_span_bytes=50292326400,
                    requested_only_change='STRATA_SYCL_PREFILL_EXPERT_SOURCE_VIEW=0|1',
                    order=['A0','B0','B1','A1','A2','B2'], requests_per_process=4,
                    prospective_selection='All3 independent paired subsequent medians favor the view; geometric mean of paired ratios>1.005, no fresh faults or eligibility/ownership failures. Target1000tok/s is checked separately, not assumed from source-copy removal.',
                    limits='Opt-in117K scheduling comparison; default adoption still needs broad quality,physical262144,session/RESTORE and sustained-generation gates.')
                plan['long_input_screen']['question'] = 'Same binary,117489 fresh tokens,384 expert slots and identical GPU placement: eliminate redundant CPU source repacking; include actual transfer/submission cost in full-request speed.'
            if gu_trial:
                plan['independent_process_plan'] = dict(order=['A0','B0','B1','A1','A2','B2'],
                    target_pairs=3, requests_per_process=4, first_use_separate=True,
                    prospective_selection='All three independent paired subsequent-request medians favor fusion; '
                        'geometric mean of per-input paired ratios>1.002, no new faults or eligibility/ownership failures.',
                    limits='Default remains disabled pending broader quality/context gates. Metadata-only controller extension after cohort0 does not change bench.sh/workload/engine.')
            if phase_profile:
                plan['phase_profile_scope'] = {
                    'question': ('Classify non-GEMM MoE host CPU with disjoint version4 regions and exact planned-work counts; command tracing off/on gates precede the two-request warm profile.' if host_profile else
                        'Compare first-use and subsequent78K CPU/queue-stage observations on the faster screened route before identifying the next limiting resource.' if warm_profile else
                        'Separate current GDN input projection, conv/gates, recurrence and output projection before selecting a topology change.'),
                    'route': f'GDN columns={column_partitions}, chunk64 disabled; qualified opt-in QSA K/Vvec4.',
                    'coverage': 'Checked coarse observer v3, four extra event pairs retired only after existing checked drains; per-layer expected counts for all36GDN layers, with zero GDN samples on12QSA layers. Legacy STRATA_PREFILL_TIMING remains prohibited.',
                    'limits': 'Event-marker intervals include host submit gaps and queue dependencies, not isolated kernel service, measured occupancy, physical traffic or a clean speed cohort. Host submit wall intervals are not thread CPU time.',
                    'retirement': 'Extract compact phase samples and faults; retire normal full logs/JIT after source/recipe recovery. No implementation adoption.'}
                if staged_host:
                    plan['phase_profile_scope'].update(
                        question='Attribute the current validated117K staged-bank route: disjoint main CPU source preparation/upload submission/explicit waits and complete96-window work counts. A checked single request precedes a separate two-request first-use/warm profile.',
                        route='Unchanged1222 engine,117488 all-GPU rows,384 slots,borrowed mixed scratch,GU-large,Down0;81920+35568 rows in each of48 layers.',
                        limits='No fine command sampler: staged expert H2D is not tagged by that compiled sampler. Coarse GPU intervals include host gaps; CPU time in waits is separate from source preparation. No physical PCIe/VRAM traffic, arithmetic saturation or clean speed claim.',
                        source_copy_hypothesis='The512-expert CPU slab copy follows a checked layer drain before activation; quantify its current wall/CPU cost before selecting CPU-only next-layer preparation overlap. Historical4.856s is not a current measurement.')
                    if source_view_trial:
                        plan['phase_profile_scope'].update(
                            question='Validate the admitted immutable Arena view on117K input and quantify remaining preparation/upload/explicit-wait CPU before clean same-binary A/B.',
                            route='Current committed candidate engine,117488 all-GPU rows,384 slots,borrowed mixed scratch,GU-large,Down0;81920+35568 rows per layer; source_view=1.',
                            source_copy_hypothesis='Legacy source_copy category now measures preparation only: actual source marker must prove view admission, zero owned upload slab and no512-expert CPU repacking. A classification change is not a speedup; clean matched request latency decides.')
            (root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
            compact = {k: v for k, v in plan.items() if k != 'source_sha256'}
            compact['source_count'] = len(pins)
            compact['source_aggregate_sha256'] = hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()
            result_path.write_text(json.dumps(compact, indent=2) + '\n')
            print('Prepared exact inputs. Commit ' + result_path.name + ' before run ' + mode)
            return 0
        record = json.loads(result_path.read_text())
        committed = subprocess.check_output(['git', 'show', 'HEAD:' + str(result_path.relative_to(P))], cwd=P)
        guards.require(committed == result_path.read_bytes() and record['mode'] == mode and
                       record.get('cohort_index', 0) == cohort and not record['complete'],
                       'commit a fresh matching plan before launch')
        plan = json.loads((root / 'plan.json').read_text())
        guards.require(all(record.get(k) == v for k, v in plan.items() if k != 'source_sha256'),
                       'external plan differs from committed inputs')
        guards.require(hashlib.sha256(json.dumps(plan['source_sha256'], sort_keys=True).encode()).hexdigest() ==
                       record['source_aggregate_sha256'], 'source identity list changed')
        guards.require(digest(BASE) == plan['reference']['sha256'], 'reference inputs changed')
        for path, expected in reference['model_identity'].items():
            info = Path(path).stat()
            guards.require(info.st_size == expected['bytes'] and info.st_mtime_ns == expected['mtime_ns'], 'model changed')
        guards.require(digest(plan['binary']['path']) == plan['binary']['sha256'], 'binary changed')
        guards.require(digest(root / 'config.json') == plan['config_sha256'] and
                       digest(root / 'expert-profile.bin') == plan['profile_sha256'], 'generated input changed')
        for name, expected in plan['source_sha256'].items():
            guards.require(digest(P / name) == expected, 'benchmark source changed: ' + name)
        built = json.loads((B / 'main-native-gdn-engine/record.json').read_text())
        for source in [built['project'], *built['dependencies']]:
            for name, expected in source['source_sha256'].items():
                guards.require(digest(Path(source['root']) / name) == expected, 'engine source changed: ' + name)
        for path, item in json.loads(SETTINGS.read_text())['runtime_elfs'].items():
            guards.require(digest(path) == item['sha256'], 'runtime asset changed: ' + path)
        if gu_trial or long_trial:
            guards.require(digest(P / plan['gu_qualification']['component_path']) ==
                           plan['gu_qualification']['component_sha256'], 'GU component evidence changed')
        if ordered_trial or long_trial:
            guards.require(digest(P / plan['ordered_exp_qualification']['component_path']) ==
                           plan['ordered_exp_qualification']['component_sha256'], 'QSA component evidence changed')
        if scratch_mixed:
            guards.require(digest(P / plan['mixed_qualification']['component_path']) ==
                           plan['mixed_qualification']['component_sha256'], 'mixed component evidence changed')
        if staged_bank:
            guards.require(digest(P/plan['expert_bank_qualification']['component_path'])==
                plan['expert_bank_qualification']['component_sha256'], 'expert-bank component evidence changed')
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        journal = subprocess.check_output(['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True)
        cursor = next(x.split(': ', 1)[1] for x in journal.splitlines() if x.startswith('-- cursor: '))
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (root / 'record.json').write_text(json.dumps(
                dict(record, owned_commands=owner.commands if owner else []), indent=2) + '\n')

        owner = supervision.Owner(root / 'owned', persist)
        try:
            entry, out, err = owner.run('bench', plan['command'], plan['environment'], P,
                wall=900 if short_profile or (check and not census) else 1800, cpu=21600, rss_cap=96 << 30,
                text_cap=64 << 20, file_cap=64 << 20, total_cap=64 << 20)
            record.update(complete=supervision.closed(entry), exit_code=entry.get('exit_code'))
            results = list((root / 'bench').glob('*/results.json'))
            guards.require(len(results) == 1, 'missing official benchmark result')
            suite = json.loads(results[0].read_text())
            requests = suite['results']['prefill_only']['requests']
            fields = ('run','ok','eligible','actual_prompt_tokens','actual_read_tokens','actual_reused_tokens',
                      'generated_tokens','thinking_requested','reasoning_observed','stream_complete',
                      'request_sha256','output_sha256','reasoning_sha256','ttft_s','total_s',
                      'prompt_tps','prompt_ms','decode_tps','decode_ms','prompt_tps_source','decode_tps_source',
                      'drafts_offered','drafts_accepted','finish_reason','err','eligibility_errors')
            record['requests'] = [{k: q[k] for k in fields if k in q} for q in requests]
            end = json.loads((root / 'engine-exit.json').read_text())
            log = (root / 'engine.stderr').read_text(errors='replace')
            record['engine_exit_code'] = end['exit_code']
            record['native_identity'] = end['native']
            record['route_selection'] = [x for x in log.splitlines() if x.startswith('[gdn-chunk64]')]
            guards.require(end['binary'] == plan['binary'] and end['exit_code'] == 0, 'native engine exit')
            guards.require(len(requests) == request_count and suite['complete'] and
                           not suite['results']['failures'], 'incomplete suite')
            guards.require(all(q['ok'] and q['eligible'] and q['stream_complete'] and q['thinking_requested'] and
                               q['actual_reused_tokens'] == 0 for q in requests), 'ineligible samples')
            guards.require(bool(record['route_selection']) == chunk64, 'wrong GDN route')
            if source_view_trial:
                source = [dict(t.split('=',1) for t in line.split(': ',1)[1].split())
                    for line in log.splitlines() if line.startswith('strata prefill expert source:')]
                guards.require(len(source)==request_count and all(
                    int(s['requested_view'])==int(s['effective_view'])==int(source_view_on) and
                    int(s['layers'])==(48 if source_view_on else 0) and
                    int(s['owned_slab_bytes'])==(0 if source_view_on else 1363148800) and
                    int(s['borrowed_layer_span_bytes'])==(50292326400 if source_view_on else 0) and
                    s['host_copy']==('eliminated' if source_view_on else 'synchronous') and
                    s['source_owner']=='caller_through_checked_retirement' for s in source),
                    'source view was not actually admitted or original copying fallback differs')
                guards.require(all(q['actual_read_tokens']==117489 for q in requests),
                    'source-view comparison workload differs from planned117489 tokens')
                record['expert_source_route'] = source
            if gu_trial or long_trial:
                selected = [x for x in log.splitlines() if x.startswith('[native16-gu]')]
                guards.require(selected == ['[native16-gu] fusion=' + ('large' if gu_candidate else '0') + ' min_rows=2334'],
                               'wrong GU startup selection')
                counts = [dict(x.split('=',1) for x in line.split(': ',1)[1].split())
                    for line in log.splitlines() if line.startswith('strata MoE GU fusion:')]
                guards.require(len(counts) == request_count and all(int(c['min_rows']) == 2334 and
                    int(c['separate_jobs']) > 0 and (int(c['fused_jobs']) > 0 and int(c['fused_rows']) >= 2334 * int(c['fused_jobs'])
                        if gu_candidate else int(c['fused_jobs']) == int(c['fused_rows']) == 0) for c in counts),
                    'actual GU route coverage mismatch')
                record['gu_route'] = dict(selection=selected, per_request_counts=counts,
                    limits='Actual submitted/completed route counts; removed logical bytes are not physical VRAM traffic.')
            if long_trial:
                guards.require(all(q['actual_prompt_tokens'] == q['actual_read_tokens'] and
                    q['actual_read_tokens'] >= long_minimum and q['reasoning_observed'] for q in requests),
                    'long-input actual-token or thinking evidence missing')
                if single_window:
                    guards.require(all(q['actual_read_tokens'] == bank_residual_rows + 1 for q in requests),
                        'single-window workload differs from the prospectively pinned78451-token shape')
                selected = [x for x in log.splitlines() if x.startswith('[native16-down]')]
                counts = [dict(x.split('=', 1) for x in line.split(': ', 1)[1].split())
                    for line in log.splitlines() if line.startswith('strata MoE Down fusion:')]
                guards.require(selected == ['[native16-down] fusion=0 min_rows=2334'] and
                    len(counts) == request_count and all(int(c['fused_jobs']) == int(c['fused_rows']) == 0 and
                        int(c['separate_jobs']) > 0 for c in counts), 'long screen changed Down route')
                record['down_route'] = dict(selection=selected, per_request_counts=counts)
                admission = [x for x in log.splitlines() if x.startswith('strata MoE window configuration:')]
                record['moe_window_admission'] = admission
                guards.require(len(admission) == request_count and all('requested_mode=10 effective_mode=10 ' in x for x in admission),
                    'long screen requires actual accumulator mode10')
                if balanced_placement or scratch_mixed:
                    placement = [x for x in log.splitlines() if x.startswith('strata prefill: layer-major candidate,')]
                    expected_rows = bank_residual_rows if staged_bank else 98304 if scratch_mixed else 81920
                    guards.require(len(placement) == request_count and all(
                        f'residual_gpu_rows={expected_rows} ' in x for x in placement), 'wrong explicit residual placement')
                    record['residual_placement'] = placement
                if scratch_mixed:
                    banks = [x for x in log.splitlines() if x.startswith('strata MoE mixed banks:')]
                    guards.require(len(banks) == request_count and all(
                        f'storage=scratch primary_bytes=335544320 tail_bytes={mixed_tail_bytes} tail_offset={mixed_tail_offset} extra_allocation_bytes=0' in x
                        for x in banks), 'wrong mixed scratch backing or extra allocation')
                    guards.require('strata prefill heads:' not in log, 'scratch trial unexpectedly retired output heads')
                    record['mixed_scratch_route'] = dict(backing=banks)
                    if staged_bank:
                        selections = [dict(t.split('=',1) for t in line.split(': ',1)[1].split())
                            for line in log.splitlines() if line.startswith('strata MoE expert bank:')]
                        totals = [dict(t.split('=',1) for t in line.split(': ',1)[1].split())
                            for line in log.splitlines() if line.startswith('strata MoE expert bank complete:')]
                        guards.require(len(selections)==len(totals)==request_count and all(
                            int(s['requested_slots'])==int(s['effective_slots'])==384 and int(s['staged'])==1 and
                            int(s['host_slab_bytes'])==1363148800 and int(s['device_bytes'])==1022361600 and
                            int(s['residual_gpu_rows'])==bank_residual_rows and
                            int(s['allocation_limit_bytes'])>=bank_residual_rows*20480
                            for s in selections) and all(0<int(s['uploads'])<=48*1024 and
                            int(s['requested_H2D_bytes'])>0 and int(s['permanent_hits'])>=0 and
                            0<int(s['permanent_slots'])<=383 and int(s['total_slots'])==384 and
                            s['final_drain']=='checked' for s in totals), 'incomplete staged-bank model route/cost')
                        record['expert_bank_route'] = dict(selections=selections,per_request_totals=totals,
                            limits='Requested raw expert H2D payload, not measured PCIe wire traffic; helper/full-request checked drains qualify this bounded model route only.')
                        if single_window:
                            work = [dict(t.split('=', 1) for t in line.split(': ', 1)[1].split())
                                for line in log.splitlines() if line.startswith('strata MoE accumulator window:')]
                            guards.require(len(work) == request_count and all(
                                int(w['windows']) == 48 and int(w['maximum_rows']) == bank_residual_rows and
                                int(w['operator_max']) == 8192 and int(w['GPU_resident_windows_only']) == 1 and
                                0 < int(w['dequants']) <= 48 * 512 and
                                int(w['dequants']) <= int(w['jobs']) == int(d['separate_jobs']) and
                                int(w['jobs']) == int(g['fused_jobs']) + int(g['separate_jobs']) and
                                int(w['source_H2D_bytes']) == bank_residual_rows * 10 * 4 * 48 and
                                int(t['uploads']) == int(w['dequants']) and int(t['permanent_hits']) == 0
                                for w, d, g, t in zip(work, record['down_route']['per_request_counts'],
                                    record['gu_route']['per_request_counts'], totals)),
                                'single-window work/upload/route counts do not conserve')
                            record['moe_window_work'] = dict(per_request_totals=work,
                                limits='Completed work counts and requested source-map H2D bytes; not physical traffic, arithmetic saturation or additive wall time.')
                    if check:
                        window_pairs = ((0, bank_window_rows[0]),) if single_window else (
                            (0, 81920), (81920, 35568 if staged_bank else 16384))
                        parts = split_long_request_logs(log, request_count) if warm_profile else [log]
                        owners = [extract_mixed_owners(part, window_pairs, mixed_tail_offset) for part in parts]
                        record['mixed_scratch_route'].update(owners[0])
                        if warm_profile:
                            record['mixed_scratch_route'].update(
                                owner_samples=[row for owner in owners for row in owner['owner_samples']],
                                retired_windows=sum(owner['retired_windows'] for owner in owners),
                                per_request_retired_windows=[owner['retired_windows'] for owner in owners])
                if not check:
                    values = [float(q['prompt_tps']) for q in requests[1:]]
                    guards.require(len(values) == 3 and all(0 < v < float('inf') for v in values),
                        'missing finite subsequent-request timings')
                    median = statistics.median(values)
                    record['long_input_process_summary'] = dict(first_use_request_index=0,
                        subsequent_request_indices=[1, 2, 3], median_prompt_tps=median,
                        sample_sd_prompt_tps=statistics.stdev(values), min_prompt_tps=min(values),
                        max_prompt_tps=max(values), process_target_passed=median >= 1000,
                        futility_stop=median < 950 and max(values) < 975,
                        limits='One process only; no3-process target/adoption or decode qualification.')
            if column_trial:
                selected = [x for x in log.splitlines() if x.startswith('strata prefill: GDN columns=')]
                expected_column_route = (f'strata prefill: GDN columns={column_partitions} '
                    f'workgroups={48 * column_partitions} lanes={512 // column_partitions} '
                    'epilogue=GPU_FP32_same_order scratch=existing_y') if column_partitions else None
                guards.require(bool(selected) == bool(column_partitions) and
                    all(x == expected_column_route for x in selected), 'wrong GDN column partition route')
                record['gdn_column_route'] = selected
            if qsa:
                record['qsa_route'] = [x for x in log.splitlines() if x.startswith(
                    'strata prefill: QSA BF16 operands, FP32 online state/accumulation;')]
                expected = 'KV=active-logical-prefix' if mode in ('qsa-base', 'qsa-census') or qcache or kvec4 else 'KV=on-the-fly'
                guards.require(record['qsa_route'] and all(expected in x for x in record['qsa_route']),
                               'wrong QSA operand route')
                if ordered_trial or gu_trial or long_trial:
                    label = 'softmax=parallel-exp-ascending-sum' if ordered_candidate else 'softmax=serial'
                    guards.require(all(label in x and 'score_pitch=33' in x for x in record['qsa_route']),
                                   'wrong ordered-exp or baseline softmax selection')
                if qcache or kvec4:
                    selected = [x for x in log.splitlines() if x.startswith('strata prefill: QSA cached_query=')]
                    wanted = kvec4 or mode in ('qsa-qcache-check', 'qsa-qcache')
                    guards.require(bool(selected) == wanted and all('cached_query=1 grf_policy=default' in x for x in selected),
                                   'wrong cached-query kernel selection')
                    record['cached_query_route'] = selected
                if kvec4:
                    selected = [x for x in log.splitlines() if x.startswith('strata prefill: QSA k_staging=')]
                    wanted = kvvec4 or mode in ('qsa-kvec4-check', 'qsa-kvec4', 'qsa-kvec4-census')
                    guards.require(bool(selected) == wanted and all('k_staging=typed4 parallel_rows=1' in x for x in selected),
                                   'wrong four-value K kernel selection')
                    record['k_staging_route'] = selected
                if kvvec4:
                    selected = [x for x in log.splitlines() if x.startswith('strata prefill: QSA v_staging=')]
                    wanted = column_trial or phase_profile or mode in ('qsa-kvvec4-check', 'qsa-kvvec4')
                    guards.require(bool(selected) == wanted and all('v_staging=typed4 layout=row-major' in x for x in selected),
                        'wrong four-value V kernel selection')
                    record['v_staging_route'] = selected
            if census:
                request_logs = [log]
                if warm_profile:
                    request_logs = split_long_request_logs(log, request_count)
                profiles = [extract_resource_profile(part, phase_profile,
                    command_profile, plan.get('phase_profile_scope', {}).get('limits', ''),
                    4 if host_profile else None,
                    ((0, 81920), (81920, 35568)) if staged_host else None) for part in request_logs]
                record.update(profiles[-1])
                if warm_profile:
                    cold = profiles[0]
                    record['first_use_resource_summary'] = {
                        'request_index': 0, 'all_profile_gates_passed': True,
                        'resource_diagnostics': [x for x in cold['resource_diagnostics'] if
                            'layer=' not in x or 'layer=-1 ' in x],
                        'gdn_phase_aggregate': cold['gdn_phase_queue_intervals']['aggregate'],
                        'command_service_summary': cold.get('command_service_profile', {}).get('summaries'),
                        'window_census': {k: v for k, v in compact_window_census(cold['planned_moe_windows']).items()
                                          if k != 'aggregate_job_rows_histogram'},
                        'detail_not_retained': 'Cold per-command/per-layer timing and detailed job histogram are not used by this aggregate first-use comparison.'}
                    record['window_census'] = compact_window_census(record.pop('planned_moe_windows'))
                    if command_profile:
                        record['command_service_profile'] = compact_command_service(record['command_service_profile'])
                    if host_profile and command_profile:
                        record['selected_service_screen'] = selected_service_screen(record['command_service_profile'])
                    record['resource_profile_request_index'] = 1
                    record['request_separation'] = {
                        'method': 'Exact long layer-major GDN route markers delimit two requests; each interval independently passes complete census/coarse/QSA/CPU/command gates.',
                        'first_request': 'First-use profiled warmup, not an independent process repetition.',
                        'second_request': ('Subsequent fresh117K request with the same immutable observer environment; no fine command sampler or clean speed claim.' if staged_host else
                            'Subsequent fresh78K request with the same immutable observer environment; profiled speed is not a clean speed sample.')}
                if host_profile and not warm_profile:
                    # The short off/on gates need checked work totals, not a
                    # second copy of every planned per-layer row distribution.
                    record['window_census'] = {k: v for k, v in
                        compact_window_census(record.pop('planned_moe_windows')).items()
                        if k != 'aggregate_job_rows_histogram'}
                    if 'command_service_profile' in record:
                        record['command_service_profile'] = compact_command_service(record['command_service_profile'])
                if staged_host:
                    # This route asks for aggregate host attribution, not a
                    # permanent second copy of all parsed layer/histogram data.
                    if warm_profile:
                        record['first_use_resource_summary'] = compact_host_resource_profile(profiles[0])
                    record.update(compact_host_resource_profile(profiles[-1]))
                    record.pop('gdn_phase_queue_intervals', None)
            guards.require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:', log), 'runtime failure')
            record['passed'] = bool(record['complete'] and entry.get('exit_code') == 0)
        except BaseException as error:
            record['error'] = str(error)
            record['error_excerpt'] = (root / 'engine.stderr').read_text(errors='replace')[-8192:] if (root / 'engine.stderr').exists() else ''
        finally:
            record['ownership_closed'] = owner.active is None and all(supervision.closed(c) for c in owner.commands)
            after = subprocess.check_output(['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager'], text=True)
            record['fresh_faults'] = [x for x in after.splitlines() if re.search(r'(?i)(gpu.*(?:hang|fault|reset|wedg)|xe.*(?:error|fault|reset|timeout))', x)]
            record['boot_unchanged'] = boot == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            try:
                record['driver_after'] = guards.driver_guard()
            except Exception as error:
                record['driver_guard_error'] = str(error)
                record['passed'] = False
            record['passed'] = bool(record['passed'] and record['ownership_closed'] and record['boot_unchanged'] and not record['fresh_faults'])
            persist()
            result_path.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({k: record.get(k) for k in ('mode', 'complete', 'passed', 'ownership_closed', 'error', 'route_selection')}))
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
