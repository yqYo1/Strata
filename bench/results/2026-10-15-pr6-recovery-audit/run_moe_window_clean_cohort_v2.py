#!/usr/bin/env python3
"""Clean cohort v2: omit presence-controlled tracing variables before exec."""
import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import statistics
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEV = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    out = A / 'moe-window-clean-cohort-v2.json'
    assert not out.exists()
    owner_path = A / 'direct_owner_transient_artifacts_v1.py'
    assert sha(owner_path) == 'bddd0c5e5020e882ca3404679420faec11e820dabce2ed4c72ef82b4c9c0874a'
    owned = module('window_clean_owner', owner_path)
    reference_paths = [A / f'moe-window-model-profile-window{arm}-v1.json' for arm in (0, 2)]
    references = [json.loads(path.read_text()) for path in reference_paths]
    for r in references:
        assert r['passed'] and r['complete'] and r['ownership_closed'] and not r['active']
        assert r['fresh_faults'] == [] and r['boot_unchanged'] and not r['driver_dump_exists']
        for c in r['commands']:
            assert owned.closed(c) and c['exit_code'] == 0
            for old in c['owners'] + [c['direct_identity']]:
                now = owned.identity(old['pid'])
                assert now is None or now['start_ticks'] != old['start_ticks']
    reference = references[0]
    assert references[1]['binary'] == reference['binary']
    comparison = A / 'moe-window-model-profile-comparison-v1.json'
    compared = json.loads(comparison.read_text())
    assert compared['passed'] and compared['complete']
    assert {x['receipt']['sha256'] for x in compared['cases']} == {sha(path) for path in reference_paths}
    binary = reference['binary']
    package = A / 'own-vllm-xe2-qualification-v1'
    assert sha(package / 'manifest.json') == '7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58'
    manifest = json.loads((package / 'manifest.json').read_text())
    helper = package / 'main_controller.py'
    assert sha(helper) == manifest['package_files']['main_controller.py']['sha256']
    guard = module('window_clean_guard', helper)
    prior = json.loads((A / 'own-qsa-bf16-prefix-tilepack-profile-census-bf16-prefix-optimized-r0-v2.json').read_text())
    old_request, = reference['benchmark_result']['results']['prefill_only']['requests']
    order = [(0, 0), (2, 0), (2, 1), (0, 1), (0, 2), (2, 2)]
    record = dict(schema='strata-moe-window-clean-cohort-v2', passed=False, complete=False,
                  source_sha256=sha(__file__), adapted_from_v1_sha256='df56da7ac9f57c55cb0583651e4f323e6b1e102e8d33a3cd0c283c49b0e3b092', owner_sha256=sha(owner_path), order=order,
                  gates={str(path): sha(path) for path in [*reference_paths, comparison]},
                  binary=binary, build_source_pins=reference['build_source_pins'],
                  benchmark_source=prior['benchmark_source'], completed=[], active_sample=None,
                  performance_adopted=False, general_quality_qualified=False,
                  physical262144_RESTORE_qualified=False,
                  scope='Official bench.sh, six separate native processes, actual39K input, thinking ON, no prefix reuse; no trace/validation/profiling. 24 generation tokens are only a completion probe. No failed sample substitution.',
                  retention=dict(raw_bytes_per_sample=1 << 28, single_text_file_bytes=64 << 20,
                                 consumer='Main clean window0/2 matched performance comparison',
                                 review_point='At closed cohort boundary, extract and verify compact evidence, then retire successful full API/protocol logs.'))

    def save():
        out.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')

    def verify_pins():
        assert sha(__file__) == record['source_sha256'] and sha(binary['path']) == binary['sha256']
        assert all(sha(path) == value for path, value in record['gates'].items())
        assert all(sha(path) == value for path, value in reference['build_source_pins'].items())
        assert all(sha(v['path']) == v['sha256'] for v in reference['runtime_files'])
        assert all(sha(P / path) == value for path, value in record['benchmark_source'].items())
        for group in ('pack_metadata_sha256', 'MTP_metadata_sha256'):
            assert all(sha(path) == value for path, value in reference[group].items())
        for path, expected in reference['model_identity'].items():
            stat = Path(path).stat()
            assert stat.st_size == expected['bytes'] and stat.st_mtime_ns == expected['mtime_ns']

    def sample(arm, repetition):
        root = B / f'main-moe-window-clean-window{arm}-r{repetition}-v2'
        receipt = A / f'moe-window-clean-window{arm}-r{repetition}-v2.json'
        assert not root.exists() and not receipt.exists()
        verify_pins()
        guard.no_model_owners(manifest['known_model_files'])
        guard.driver_guard()
        assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == reference['boot_id']
        root.mkdir(mode=0o700)
        env = {k: v for k, v in reference['environment'].items()
               if k not in ('STRATA_TRACE', 'UR_ENABLE_LAYERS') and not k.startswith('UR_LOG_')}
        env.update(XDG_CACHE_HOME=str(root / 'cache'), STRATA_SYCL_PREFILL_MOE_WINDOW=str(arm),
                   STRATA_MEASUREMENT_ENGINE_START=str(root / 'engine-start.json'),
                   STRATA_MEASUREMENT_ENGINE_EXIT=str(root / 'engine-exit.json'),
                   STRATA_PREFILL_LAYER_MAJOR_PROFILE='0', STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE='0',
                   STRATA_PREFILL_CPU_WAIT_PROFILE='0', STRATA_PREFILL_LAYER_MAJOR_SHAPE_CENSUS='0')
        profile_wrapper = Path(reference['config']['exe'])
        assert sha(profile_wrapper) == reference['wrapper_sha256']
        wrapper = root / 'engine-wrapper.py'
        wrapper_text = profile_wrapper.read_text()
        assert wrapper_text.count(", 'UR_ENABLE_LAYERS']") == 1
        wrapper_text = wrapper_text.replace(", 'UR_ENABLE_LAYERS']", "]")
        anchor = "    entry['experiment_environment'] ="
        assert wrapper_text.count(anchor) == 1
        wrapper_text = wrapper_text.replace(anchor, "    assert 'STRATA_TRACE' not in env and 'UR_ENABLE_LAYERS' not in env\n" + anchor)
        wrapper.write_text(wrapper_text)
        wrapper.chmod(0o700)
        config = dict(reference['config'])
        config['args'] = list(config['args'])
        assert config['args'][config['args'].index('--prompt-cache') + 1] == '0'
        external = Path(config['args'][config['args'].index('--expert-profile') + 1])
        local = root / 'expert-profile.bin'
        assert sha(external) == reference['external_profile_sha256']
        shutil.copyfile(external, local)
        config['args'][config['args'].index('--expert-profile') + 1] = str(local)
        config.update(exe=str(wrapper), log=str(root / 'owned/engine.stderr'))
        (root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        port = 18740 + arm + repetition * 4
        (root / 'plan.json').write_text(json.dumps(dict(port=port, url=f'http://127.0.0.1:{port}',
                runs=1, benchmark_options=reference['benchmark_options'], bench_timeout_seconds=1500), indent=2) + '\n')
        rec = dict(schema='strata-moe-window-clean-sample-v2', passed=False, complete=False,
                   window=arm, repetition=repetition, active=False, GPU_executed=False,
                   root=str(root), boot_id=reference['boot_id'], binary=binary, environment=env,
                   recipe_sha256=record['source_sha256'], wrapper_sha256=sha(wrapper), config=config,
                   benchmark_options=reference['benchmark_options'], commands=[])
        owner = None

        def snapshot():
            rec['active'] = bool(owner and owner.active is not None)
            receipt.write_text(json.dumps(rec, indent=2, allow_nan=False) + '\n')

        owner = owned.Owner(root / 'owned', snapshot)
        rec['commands'] = owner.commands

        def run(label, argv, wall=60, cpu=120, rss=2 << 30):
            command, stdout, stderr = owner.run(label, argv, env, P, wall=wall, cpu=cpu,
                    rss_cap=rss, text_cap=64 << 20, total_cap=1 << 28, file_cap=64 << 20)
            assert owned.closed(command) and command['exit_code'] == 0, (label, command.get('errors'), str(stderr))
            return stdout.read_text()

        cursor = None
        snapshot()
        try:
            raw = run('journal-before', ['/usr/bin/journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])
            cursor = re.search(r'-- cursor: (\S+)', raw)[1]
            rec['journal_before_cursor'] = cursor
            run('prefill-only', ['/home/yayoi/.local/share/strata-sycl/venv/bin/python',
                                str(P / 'bench/results/2026-10-14-long-generation-benchmark/run_server_job.py'),
                                str(root), 'extended-bench'], wall=1800, cpu=21600, rss=88 << 30)
            result = json.loads((root / 'result.json').read_text())
            rec['result'] = result
            assert result['passed'] and result['server_exit_code'] == result['real_engine_exit']['exit_code'] == 0
            native_exit = result['real_engine_exit']
            assert native_exit['binary_sha256'] == binary['sha256']
            assert native_exit['experiment_environment'] == {k: env[k] for k in native_exit['experiment_environment']}
            suites = list((root / 'stock-results').glob('*/results.json'))
            suite, = [json.loads(path.read_text()) for path in suites]
            rec['benchmark_result'] = suite
            cohort = suite['results']['prefill_only']
            request, = cohort['requests']
            assert suite['complete'] and not suite['results']['failures'] and cohort['complete']
            assert suite['params']['think'] and request['thinking_requested'] and request['reasoning_observed']
            assert request['ok'] and request['eligible'] and request['stream_complete']
            assert request['actual_prompt_tokens'] == request['actual_read_tokens'] == 39262
            assert request['actual_reused_tokens'] == 0 and request['request_sha256'] == old_request['request_sha256']
            native = (root / 'owned/engine.stderr').read_text()
            rec['native_log_sha256'] = sha(root / 'owned/engine.stderr')
            assert not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:|retirement failed', native)
            assert not any(prefix in native for prefix in ('strata trace:', 'strata lm profile:',
                                                          'strata lm qsa profile:', 'strata prefill cpu wait:', 'strata prefill cpu host:'))
            rec['placement_lines'] = [line for line in native.splitlines() if line.startswith('strata prefill: layer-major candidate,')]
            assert len(rec['placement_lines']) == 1 and 'residual_gpu_rows=32768 ' in rec['placement_lines'][0]
            rec['window_lines'] = [line for line in native.splitlines() if line.startswith('strata MoE window: ')]
            assert (len(rec['window_lines']) == 1 and 'windows=96 ' in rec['window_lines'][0]) if arm else not rec['window_lines']
            assert sha(external) == reference['external_profile_sha256']
            verify_pins()
            rec.update(passed=True, complete=True, GPU_executed=True)
        except BaseException as error:
            rec['error'] = repr(error)
        finally:
            try:
                assert owner.active is None
                if cursor:
                    raw = run('journal-final', ['/usr/bin/journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
                    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
                    rec['fresh_faults'] = [row for row in rows if FAULT.search(str(row.get('MESSAGE', '')))]
                rec['ownership_closed'] = all(owned.closed(c) for c in owner.commands)
                rec['boot_unchanged'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip() == rec['boot_id']
                rec['driver_dump_exists'] = (DEV / 'devcoredump').exists() or bool(list(Path('/sys/class/devcoredump').glob('devcd*')))
                assert rec['ownership_closed'] and rec['boot_unchanged'] and not rec['driver_dump_exists'] and rec['fresh_faults'] == []
                guard.driver_guard()
            except BaseException as error:
                rec.update(closure_error=repr(error), passed=False, complete=False)
            rec['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            snapshot()
        return receipt, rec

    prior_cohort_path = A / 'moe-window-clean-cohort-v1.json'
    prior_cohort = json.loads(prior_cohort_path.read_text())
    assert not prior_cohort['passed'] and not prior_cohort['complete'] and prior_cohort['active_sample'] is None
    assert len(prior_cohort['completed']) == 1
    prior_sample_path = Path(prior_cohort['completed'][0]['path'])
    assert sha(prior_sample_path) == prior_cohort['completed'][0]['sha256']
    prior_sample = json.loads(prior_sample_path.read_text())
    assert not prior_sample['passed'] and not prior_sample['complete'] and prior_sample['ownership_closed'] and not prior_sample['active']
    assert prior_sample['fresh_faults'] == [] and prior_sample['boot_unchanged'] and not prior_sample['driver_dump_exists']
    assert prior_sample['result']['passed'] and prior_sample['result']['real_engine_exit']['exit_code'] == 0
    for command in prior_sample['commands']:
        assert owned.closed(command) and command['exit_code'] == 0
        for old in command['owners'] + [command['direct_identity']]:
            now = owned.identity(old['pid'])
            assert now is None or now['start_ticks'] != old['start_ticks']
    record['invalid_v1_predecessor'] = dict(path=str(prior_cohort_path), sha256=sha(prior_cohort_path),
        sample_sha256=sha(prior_sample_path), decision='Preserve failed original; STRATA_TRACE=0 enabled presence-controlled tracing. No v1 speed sample participates in this six-sample cohort.')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save()
        try:
            for arm, repetition in order:
                record['active_sample'] = dict(window=arm, repetition=repetition)
                save()
                print(json.dumps(dict(start=arm, repetition=repetition)), flush=True)
                path, result = sample(arm, repetition)
                item = dict(window=arm, repetition=repetition, path=str(path), sha256=sha(path),
                            passed=result['passed'], complete=result['complete'])
                record['completed'].append(item)
                record['active_sample'] = None
                save()
                assert result['passed'] and result['complete'], result.get('error', result.get('closure_error'))
                request, = result['benchmark_result']['results']['prefill_only']['requests']
                item['request'] = request
                print(json.dumps(dict(finished=arm, repetition=repetition, prompt_tps=request['prompt_tps'])), flush=True)
                save()
            record['summaries'] = {}
            for arm in (0, 2):
                values = [x['request']['prompt_tps'] for x in record['completed'] if x['window'] == arm]
                record['summaries'][str(arm)] = dict(n=len(values), samples=values, median=statistics.median(values), min=min(values), max=max(values))
            record['passed'] = record['complete'] = True
        except BaseException as error:
            record['error'] = repr(error)
        finally:
            record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            save()
    print(json.dumps(dict(passed=record['passed'], complete=record['complete'], summaries=record.get('summaries'), error=record.get('error'))), flush=True)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
