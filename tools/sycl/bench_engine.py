#!/usr/bin/env python3
"""Prepare/commit/run a matched official bench.sh GDN or QSA comparison; Main only."""
import fcntl
import hashlib
import json
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


def main():
    modes = ('check', 'base', 'candidate', 'qsa-check', 'qsa-base', 'qsa-direct', 'qsa-census',
             'qsa-qcache-check', 'qsa-qcache-base', 'qsa-qcache',
             'qsa-kvec4-check', 'qsa-kvec4-base', 'qsa-kvec4', 'qsa-kvec4-census',
             'qsa-kvvec4-check', 'qsa-kvvec4-base', 'qsa-kvvec4',
             'qsa-kvvec4-timing-check', 'qsa-kvvec4-census')
    guards.require(len(sys.argv) == 3 and sys.argv[1] in ('prepare', 'run') and
                   sys.argv[2] in modes, 'use prepare|run ' + '|'.join(modes))
    action, mode = sys.argv[1:]
    qsa = mode.startswith('qsa-')
    short_profile = mode == 'qsa-kvvec4-timing-check'
    phase_profile = short_profile or mode == 'qsa-kvvec4-census'
    census = phase_profile or mode in ('qsa-census', 'qsa-kvec4-census')
    qcache = mode.startswith('qsa-qcache')
    kvvec4 = mode.startswith('qsa-kvvec4')
    kvec4 = kvvec4 or mode.startswith('qsa-kvec4')
    check = census or mode in ('check', 'qsa-check', 'qsa-qcache-check', 'qsa-kvec4-check', 'qsa-kvvec4-check')
    chunk64 = mode in ('check', 'candidate')
    generation = 'v2' if mode == 'qsa-kvec4-check' else 'v1'
    root = B / (('qsa-attention-model-' + mode[4:] + '-' + generation) if qsa else
                ('native-gdn-model-' + mode + '-v2'))
    result_path = (A / 'moe-window-census-current.json' if census else
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
                       STRATA_SYCL_PREFILL_GDN_CHUNK64='1' if chunk64 else '0')
            if qsa:
                env['STRATA_SYCL_PREFILL_QSA_MATH'] = (
                    'bf16-prefix-kvvec4' if phase_profile or mode in ('qsa-kvvec4-check', 'qsa-kvvec4') else
                    'bf16-prefix-kvec4' if mode in ('qsa-kvec4-check', 'qsa-kvec4', 'qsa-kvec4-census', 'qsa-kvvec4-base') else
                    'bf16-prefix-qcache' if mode in ('qsa-qcache-check', 'qsa-qcache', 'qsa-kvec4-base') else
                    'bf16' if mode in ('qsa-check', 'qsa-direct') else 'bf16-prefix-optimized')
            if census:
                env.update(STRATA_PREFILL_LAYER_MAJOR_PROFILE='1',
                           STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE='1',
                           STRATA_PREFILL_CPU_WAIT_PROFILE='1')
            if phase_profile:
                env['STRATA_PREFILL_TIMING'] = '1'
            if phase_profile or mode in ('qsa-kvec4-check', 'qsa-kvec4-census'):
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
                       '--runs', '1' if check else '4', '--prefill-only', '--prefill-tokens',
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
            if phase_profile:
                plan['phase_profile_scope'] = {
                    'question': 'Separate current GDN input projection, conv/gates, recurrence and output projection before selecting a topology change.',
                    'route': 'Existing GDN recurrence, chunk64 disabled; qualified opt-in QSA K/Vvec4.',
                    'coverage': 'One rounded marker-interval summary per layer; GDN36/QSA12. Original checked coarse/CPU and selected MoE service observers remain separate.',
                    'limits': 'Marker intervals include host submit gaps and queue dependencies, not isolated kernel service, measured occupancy, physical traffic or a clean speed cohort. Legacy PfTimer skips failed event queries; require expected emitted phase coverage and retain this limitation.',
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
            guards.require(len(requests) == (1 if check else 4) and suite['complete'] and
                           not suite['results']['failures'], 'incomplete suite')
            guards.require(all(q['ok'] and q['eligible'] and q['stream_complete'] and q['thinking_requested'] and
                               q['actual_reused_tokens'] == 0 for q in requests), 'ineligible samples')
            guards.require(bool(record['route_selection']) == chunk64, 'wrong GDN route')
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
                    wanted = phase_profile or mode in ('qsa-kvvec4-check', 'qsa-kvvec4')
                    guards.require(bool(selected) == wanted and all('v_staging=typed4 layout=row-major' in x for x in selected),
                        'wrong four-value V kernel selection')
                    record['v_staging_route'] = selected
            if census:
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
                for prefix in ('strata lm profile: v=2 complete=1 errors=0',
                               'strata lm qsa profile: v=1 complete=1 errors=0',
                               'strata prefill cpu wait: version=3 available=1 complete=1',
                               'strata prefill cpu host: version=3 available=1 open_regions=0'):
                    guards.require(any(x.startswith(prefix) for x in record['resource_diagnostics']),
                                   'missing complete resource observer: ' + prefix)
                record['census_scope'] = 'Planned host metadata reconciled with completed window issuer totals; profiler/validation enabled, not clean speed or physical hardware saturation.'
            if phase_profile:
                # PfTimer reports rounded stream-marker intervals, not GPU-only
                # service. Source run_layer_major visits layers in order and
                # calls run_impl with the full request's prefill row count.
                expected_rows = record['planned_moe_windows'][0]['rows']
                expected_phases = ('gdn', 'gdn conv+gates', 'gdn recurrence', 'gdn out proj')
                layers = []
                for line in log.splitlines():
                    match = re.fullmatch(r'strata prefill timing: (\d+) tokens, GPU timeline (\d+) ms, '
                                         r'wall (\d+) ms, host staging (\d+) ms:(.*)', line)
                    if not match or int(match[1]) != expected_rows:
                        continue
                    fields = dict((name.strip(), int(ms)) for name, ms in re.findall(
                        r' ([a-z+ ]+) (\d+) \([0-9.]+%\)', match[5]))
                    row = dict(layer=len(layers), rows=expected_rows, timeline_ms=int(match[2]),
                               wall_ms=int(match[3]), gdn_marker_ms={k:fields[k] for k in expected_phases if k in fields})
                    layers.append(row)
                gdn = [x for x in layers if x['gdn_marker_ms']]
                record['gdn_phase_marker_intervals'] = dict(layers=layers,
                    totals_ms={k:sum(x['gdn_marker_ms'].get(k, 0) for x in gdn) for k in expected_phases},
                    limits=plan['phase_profile_scope']['limits'])
                guards.require(len(layers) == 48 and len(gdn) == 36 and
                    all(set(x['gdn_marker_ms']) == set(expected_phases) for x in gdn),
                    'incomplete per-layer GDN marker coverage')
            if phase_profile or mode in ('qsa-kvec4-check', 'qsa-kvec4-census'):
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
