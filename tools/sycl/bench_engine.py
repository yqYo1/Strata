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


CPU_HOST_BASE = ('source_copy', 'host_group', 'gemm_submit', 'route_exchange',
    'upload_submit', 'residual_input_host', 'residual_input_device',
    'residual_store_host', 'residual_store_device')
CPU_HOST_MOE = ('moe_map_upload', 'moe_job_weights', 'moe_dequant_gu',
    'moe_dequant_down', 'moe_gather', 'moe_swiglu', 'moe_accumulate', 'moe_status')


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


def extract_resource_profile(log, phase_profile, command_profile, limits, cpu_version=None):
    """Validate one long-request interval; never aggregate across requests."""
    record = {}
    prefix = 'strata MoE planned job census: '
    record['planned_moe_windows'] = [json.loads(x[len(prefix):]) for x in log.splitlines() if x.startswith(prefix)]
    guards.require(len(record['planned_moe_windows']) == 48 and
        [x['layer'] for x in record['planned_moe_windows']] == list(range(48)), 'missing window census')
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
        expected = dict(zip(CPU_HOST_MOE, [48, 48, 0, 0, total_jobs, total_jobs, total_jobs, 48]))
        experts = sum(n for window in record['planned_moe_windows']
                      for m, n in window['jobs_per_expert_histogram'] if m > 0)
        expected.update(moe_dequant_gu=experts, moe_dequant_down=experts)
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


def main():
    modes = ('check', 'base', 'candidate', 'qsa-check', 'qsa-base', 'qsa-direct', 'qsa-census',
             'qsa-gdncols4-warm-census',
             'qsa-gdncols4-host-check', 'qsa-gdncols4-host-traced-check', 'qsa-gdncols4-host-census',
             'qsa-qcache-check', 'qsa-qcache-base', 'qsa-qcache',
             'qsa-kvec4-check', 'qsa-kvec4-base', 'qsa-kvec4', 'qsa-kvec4-census',
             'qsa-kvvec4-check', 'qsa-kvvec4-base', 'qsa-kvvec4',
             'qsa-kvvec4-timing-check', 'qsa-kvvec4-census') + tuple(
                 f'qsa-gdncols{n}-{phase}' for n in (2, 4, 8) for phase in ('check', 'base', 'candidate'))
    guards.require(len(sys.argv) == 3 and sys.argv[1] in ('prepare', 'run') and
                   sys.argv[2] in modes, 'use prepare|run ' + '|'.join(modes))
    action, mode = sys.argv[1:]
    column_mode = re.fullmatch(r'qsa-gdncols(2|4|8)-(check|base|candidate|warm-census|host-check|host-traced-check|host-census)', mode)
    host_profile = mode.startswith('qsa-gdncols4-host-')
    warm_profile = mode in ('qsa-gdncols4-warm-census', 'qsa-gdncols4-host-census')
    column_trial = column_mode is not None
    column_check = column_trial and column_mode[2] == 'check'
    column_partitions = int(column_mode[1]) if column_trial and column_mode[2] != 'base' else 0
    qsa = mode.startswith('qsa-')
    short_profile = mode == 'qsa-kvvec4-timing-check' or (host_profile and not warm_profile)
    phase_profile = host_profile or warm_profile or short_profile or mode == 'qsa-kvvec4-census'
    command_profile = (phase_profile and mode != 'qsa-gdncols4-host-check') or mode in ('qsa-kvec4-check', 'qsa-kvec4-census')
    census = phase_profile or mode in ('qsa-census', 'qsa-kvec4-census')
    qcache = mode.startswith('qsa-qcache')
    kvvec4 = column_trial or mode.startswith('qsa-kvvec4')
    kvec4 = kvvec4 or mode.startswith('qsa-kvec4')
    check = census or column_check or mode in ('check', 'qsa-check', 'qsa-qcache-check', 'qsa-kvec4-check', 'qsa-kvvec4-check')
    request_count = 2 if warm_profile else 1 if check else 4
    chunk64 = mode in ('check', 'candidate')
    generation = 'v2' if short_profile or mode == 'qsa-kvec4-check' else 'v1'
    root = B / (('qsa-attention-model-' + mode[4:] + '-' + generation) if qsa else
                ('native-gdn-model-' + mode + '-v2'))
    result_path = (A / 'prefill-host-attribution-current.json' if host_profile else
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
            if host_profile:
                cpu_check = json.loads((A / 'prefill-host-cpu-current.json').read_text())
                guards.require(cpu_check['passed'] and cpu_check['ownership_closed'] and
                    cpu_check['source_sha256']['include/strata/sycl/cpu_wait_profile.hpp'] ==
                        digest(P / 'include/strata/sycl/cpu_wait_profile.hpp'), 'matching CPU accounting gates required')
                qualification = []
                if mode != 'qsa-gdncols4-host-check':
                    prior_bytes = subprocess.check_output(['git', 'show',
                        'HEAD:' + str(result_path.relative_to(P))], cwd=P)
                    guards.require(prior_bytes == result_path.read_bytes(), 'commit the previous host model gate')
                    prior = json.loads(result_path.read_text())
                    required = ('qsa-gdncols4-host-traced-check' if warm_profile else 'qsa-gdncols4-host-check')
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
            if qsa:
                env['STRATA_SYCL_PREFILL_QSA_MATH'] = (
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
            config = dict(reference['config'])
            flags = list(config['args'])
            flags[flags.index('--expert-profile') + 1] = str(root / 'expert-profile.bin')
            config.update(exe=str(P / 'tools/sycl/engine_once.py'), args=flags,
                          log=str(root / 'engine.stderr'), env={})
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config, indent=2) + '\n')
            command = ['/bin/sh', str(P / 'bench.sh'), '--config', str(config_path), '--port', str(port),
                       '--runs', str(request_count), '--prefill-only', '--prefill-tokens',
                       '32768' if short_profile or (check and not census) else '65536', '--workload-id', 'native_gdn_20261018',
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
            if phase_profile:
                plan['phase_profile_scope'] = {
                    'question': ('Classify non-GEMM MoE host CPU with disjoint version4 regions and exact planned-work counts; command tracing off/on gates precede the two-request warm profile.' if host_profile else
                        'Compare first-use and subsequent78K CPU/queue-stage observations on the faster screened route before identifying the next limiting resource.' if warm_profile else
                        'Separate current GDN input projection, conv/gates, recurrence and output projection before selecting a topology change.'),
                    'route': f'GDN columns={column_partitions}, chunk64 disabled; qualified opt-in QSA K/Vvec4.',
                    'coverage': 'Checked coarse observer v3, four extra event pairs retired only after existing checked drains; per-layer expected counts for all36GDN layers, with zero GDN samples on12QSA layers. Legacy STRATA_PREFILL_TIMING remains prohibited.',
                    'limits': 'Event-marker intervals include host submit gaps and queue dependencies, not isolated kernel service, measured occupancy, physical traffic or a clean speed cohort. Host submit wall intervals are not thread CPU time.',
                    'retirement': 'Extract compact phase samples and faults; retire normal full logs/JIT after source/recipe recovery. No implementation adoption.'}
            (root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
            compact = {k: v for k, v in plan.items() if k != 'source_sha256'}
            compact['source_count'] = len(pins)
            compact['source_aggregate_sha256'] = hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()
            result_path.write_text(json.dumps(compact, indent=2) + '\n')
            print('Prepared exact inputs. Commit ' + result_path.name + ' before run ' + mode)
            return 0
        record = json.loads(result_path.read_text())
        committed = subprocess.check_output(['git', 'show', 'HEAD:' + str(result_path.relative_to(P))], cwd=P)
        guards.require(committed == result_path.read_bytes() and record['mode'] == mode and not record['complete'],
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
                    starts = [m.start() for m in re.finditer(r'(?m)^strata prefill: GDN columns=', log)]
                    guards.require(len(starts) == request_count, 'wrong number of long-request start markers')
                    guards.require(not any(x in log[:starts[0]] for x in
                        ('strata lm profile: v=', 'strata command profile:', 'strata prefill cpu wait:')),
                        'resource observer before first long-request marker')
                    request_logs = [log[a:b] for a, b in zip(starts, starts[1:] + [len(log)])]
                profiles = [extract_resource_profile(part, phase_profile,
                    command_profile, plan.get('phase_profile_scope', {}).get('limits', ''),
                    4 if host_profile else None) for part in request_logs]
                record.update(profiles[-1])
                if warm_profile:
                    cold = profiles[0]
                    record['first_use_resource_summary'] = {
                        'request_index': 0, 'all_profile_gates_passed': True,
                        'resource_diagnostics': [x for x in cold['resource_diagnostics'] if
                            'layer=' not in x or 'layer=-1 ' in x],
                        'gdn_phase_aggregate': cold['gdn_phase_queue_intervals']['aggregate'],
                        'command_service_summary': cold['command_service_profile']['summaries'],
                        'window_census': {k: v for k, v in compact_window_census(cold['planned_moe_windows']).items()
                                          if k != 'aggregate_job_rows_histogram'},
                        'detail_not_retained': 'Cold per-command/per-layer timing and detailed job histogram are not used by this aggregate first-use comparison.'}
                    record['window_census'] = compact_window_census(record.pop('planned_moe_windows'))
                    record['command_service_profile'] = compact_command_service(record['command_service_profile'])
                    record['resource_profile_request_index'] = 1
                    record['request_separation'] = {
                        'method': 'Exact long layer-major GDN route markers delimit two requests; each interval independently passes complete census/coarse/QSA/CPU/command gates.',
                        'first_request': 'First-use profiled warmup, not an independent process repetition.',
                        'second_request': 'Subsequent fresh78K request with the same immutable observer environment; profiled speed is not a clean speed sample.'}
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
