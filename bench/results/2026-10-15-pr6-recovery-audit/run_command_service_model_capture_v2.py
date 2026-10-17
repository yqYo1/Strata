#!/usr/bin/env python3
"""Owned single39K raw-timestamp diagnostic; inference completion and timing qualification are separate."""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import socket
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
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=['diagnostic'])
    opts = parser.parse_args()
    out = A / f'command-service-model-{opts.phase}-cohort-v2.json'
    buildpath = A / 'command-service-collector-engine-build-v2.json'
    build = json.loads(buildpath.read_text())
    assert build['passed'] and build['complete'] and build['ownership_closed'] and not build['active']
    fixturepath = A / 'command-profile-fixture-on-v2.json'
    fixture = json.loads(fixturepath.read_text())
    assert fixture['passed'] and fixture['complete'] and fixture['ownership_closed'] and not fixture['active']
    assert fixture['fresh_faults'] == [] and not fixture['driver_dump_entries']
    errorpath=A/'command-profile-fixture-error-v2.json'
    errorcase=json.loads(errorpath.read_text())
    assert errorcase['passed'] and errorcase['complete'] and errorcase['ownership_closed'] and not errorcase['active']
    assert errorcase['native_exit_code']==errorcase['expected_native_exit']==1
    assert not errorcase['fresh_faults'] and not errorcase['driver_dump_entries']
    ready_path = A / 'command-service-model-readiness-v2.json'
    ready = json.loads(ready_path.read_text())
    assert ready['schedulable_now']
    for path,digest in ready['pins'].items():
        assert sha(path) == digest
    if opts.phase == 'clean':
        diagpath = A / 'command-service-model-diagnostic-cohort-v1.json'
        diag = json.loads(diagpath.read_text())
        assert diag['passed'] and diag['complete'] and diag['active_sample'] is None
        for item in diag['completed']:
            assert sha(item['path']) == item['sha256']
            r = json.loads(Path(item['path']).read_text())
            assert r['passed'] and r['ownership_closed'] and not r['fresh_faults'] and not r['driver_dump_exists']

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
    reference = references[1]
    assert references[0]['binary'] == reference['binary']
    comparison = A / 'moe-window-model-profile-comparison-v1.json'
    compared = json.loads(comparison.read_text())
    assert compared['passed'] and compared['complete']
    assert {x['receipt']['sha256'] for x in compared['cases']} == {sha(path) for path in reference_paths}
    binary = build['products']['strata']
    package = A / 'own-vllm-xe2-qualification-v1'
    assert sha(package / 'manifest.json') == '7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58'
    manifest = json.loads((package / 'manifest.json').read_text())
    helper = package / 'main_controller.py'
    assert sha(helper) == manifest['package_files']['main_controller.py']['sha256']
    guard = module('window_clean_guard', helper)
    prior = json.loads((A / 'own-qsa-bf16-prefix-tilepack-profile-census-bf16-prefix-optimized-r0-v2.json').read_text())
    old_request, = reference['benchmark_result']['results']['prefill_only']['requests']
    order = [('on',0)]
    record = dict(schema='strata-command-service-model-v2',phase=opts.phase, passed=False, complete=False,
                  source_sha256=sha(__file__), adapted_from_clean_moe_v3_sha256='34f25fcba3aa38daa2e8d6a285784a7902823be7bd28d7b254d5eb08ef91b3cb', owner_sha256=sha(owner_path), order=order,
                  gates={str(path): sha(path) for path in [*reference_paths, comparison, buildpath, fixturepath, errorpath]},
                  binary=binary, build_source_pins=build['source_pins'],
                  benchmark_source=prior['benchmark_source'], completed=[], active_sample=None,
                  performance_adopted=False, general_quality_qualified=False,
                  physical262144_RESTORE_qualified=False,
                  scope='One fresh39K officialbench raw-timestamp diagnostic of v2 collector; all accepted and up to32 rejected events serialized. PASS denotes complete raw capture/native inference/closure, not timestamp qualification; profile_qualified is separate. Existing v1 failed receipt remains immutable. Trace/parameter validation,24 thinking tokens completion probe; no clean speed or inference adoption.',
                  retention=dict(raw_bytes_per_sample=64 << 20, raw_bytes_pair=64 << 20,
                                 artifact_bytes_per_sample=1 << 28, single_text_file_bytes=64 << 20,
                                 pair_budget_reason='One fresh owned raw-timestamp diagnostic job with the helper64MiB aggregate stdout/stderr hard cap;256MiB is a separate artifact cap.',
                                 consumer='Main rejected timestamp predicate diagnosis; four selected first MoE windows; no timing qualification from PASS alone',
                                 review_point='At closed cohort boundary, extract and verify compact evidence, then retire successful full API/protocol logs.'))

    def save():
        out.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')

    def verify_pins():
        assert sha(__file__) == record['source_sha256'] and sha(binary['path']) == binary['sha256']
        assert all(sha(path) == value for path, value in record['gates'].items())
        assert all(sha(path) == value for path, value in build['source_pins'].items())
        assert all(sha(v['path']) == v['sha256'] for v in reference['runtime_files'])
        assert all(sha(P / path) == value for path, value in record['benchmark_source'].items())
        for group in ('pack_metadata_sha256', 'MTP_metadata_sha256'):
            assert all(sha(path) == value for path, value in reference[group].items())
        for path, expected in reference['model_identity'].items():
            stat = Path(path).stat()
            assert stat.st_size == expected['bytes'] and stat.st_mtime_ns == expected['mtime_ns']

    def sample(arm, repetition):
        root = B / f'main-command-service-model-{opts.phase}-{arm}-r{repetition}-v2'
        receipt = A / f'command-service-model-{opts.phase}-{arm}-r{repetition}-v2.json'
        assert not root.exists() and not receipt.exists()
        verify_pins()
        guard.no_model_owners(manifest['known_model_files'])
        guard.driver_guard()
        assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == reference['boot_id']
        root.mkdir(mode=0o700)
        env = {k: v for k, v in reference['environment'].items()
               if k not in ('STRATA_TRACE', 'UR_ENABLE_LAYERS') and not k.startswith('UR_LOG_')}
        env.update(XDG_CACHE_HOME=str(root / 'cache'), STRATA_SYCL_PREFILL_MOE_WINDOW='2', STRATA_SYCL_PREFILL_QSA_MATH='bf16-prefix-optimized', STRATA_SYCL_PREFILL_COMMAND_PROFILE='0' if arm=='off' else '1',
                   STRATA_MEASUREMENT_ENGINE_START=str(root / 'engine-start.json'),
                   STRATA_MEASUREMENT_ENGINE_EXIT=str(root / 'engine-exit.json'),
                   STRATA_PREFILL_LAYER_MAJOR_PROFILE='0', STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE='0',
                   STRATA_PREFILL_CPU_WAIT_PROFILE='0', STRATA_PREFILL_LAYER_MAJOR_SHAPE_CENSUS='0')
        if opts.phase == 'diagnostic':
            env.update(STRATA_TRACE='1',UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
                       UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                       UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                       UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
        profile_wrapper = Path(reference['config']['exe'])
        assert sha(profile_wrapper) == reference['wrapper_sha256']
        wrapper = root / 'engine-wrapper.py'
        wrapper_text = profile_wrapper.read_text()
        assert wrapper_text.count("'STRATA_SYCL_PREFILL_MOE_WINDOW',") == 1
        wrapper_text = wrapper_text.replace("'STRATA_SYCL_PREFILL_MOE_WINDOW',", "'STRATA_SYCL_PREFILL_MOE_WINDOW','STRATA_SYCL_PREFILL_COMMAND_PROFILE',")
        wrapper_text = re.sub(r'^ENGINE = .*?$', 'ENGINE = '+repr(binary['path']), wrapper_text, flags=re.M)
        wrapper_text = re.sub(r'^ENGINE_SHA256 = .*?$', 'ENGINE_SHA256 = '+repr(binary['sha256']), wrapper_text, flags=re.M)
        assert wrapper_text.count(", 'UR_ENABLE_LAYERS']") == 1
        wrapper_text = wrapper_text.replace(", 'UR_ENABLE_LAYERS']", "]")
        anchor = "    entry['experiment_environment'] ="
        assert wrapper_text.count(anchor) == 1
        launch = "    child = subprocess.Popen([ENGINE, *sys.argv[1:]], env=env)"
        assert wrapper_text.count(launch) == 1
        if opts.phase == 'clean':
            wrapper_text = wrapper_text.replace(launch, "    assert 'STRATA_TRACE' not in env and 'UR_ENABLE_LAYERS' not in env\n" + launch)
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
        with socket.socket() as selected_port:
            selected_port.bind(('127.0.0.1', 0))
            port = selected_port.getsockname()[1]
        (root / 'plan.json').write_text(json.dumps(dict(port=port, url=f'http://127.0.0.1:{port}',
                runs=1, benchmark_options=reference['benchmark_options'], bench_timeout_seconds=1500), indent=2) + '\n')
        rec = dict(schema='strata-command-service-model-sample-v2',phase=opts.phase, passed=False, complete=False,
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
            if opts.phase == 'clean':
                assert not any(prefix in native for prefix in ('strata trace:', 'strata lm profile:',
                                                          'strata lm qsa profile:', 'strata prefill cpu wait:', 'strata prefill cpu host:'))
            rec['placement_lines'] = [line for line in native.splitlines() if line.startswith('strata prefill: layer-major candidate,')]
            assert len(rec['placement_lines']) == 1 and 'residual_gpu_rows=32768 ' in rec['placement_lines'][0]
            rec['window_lines'] = [line for line in native.splitlines() if line.startswith('strata MoE window: ')]
            assert len(rec['window_lines']) == 1 and 'windows=96 ' in rec['window_lines'][0]
            rec['QSA_math_lines'] = [line for line in native.splitlines() if line.startswith('strata prefill: QSA BF16 operands,')]
            math_line, = rec['QSA_math_lines']
            assert 'score_pitch=33 operand_layout=row-major softmax=serial' in math_line
            command_summary=[line for line in native.splitlines() if line.startswith('strata command profile: ')]
            command_records=[line for line in native.splitlines() if line.startswith('strata command sample: ')]
            command_groups=[line for line in native.splitlines() if line.startswith('strata command group: ')]
            invalid_lines=[line for line in native.splitlines() if line.startswith('strata command invalid: ')]
            rec.update(command_profile_summary=command_summary[0] if len(command_summary)==1 else command_summary,
                       command_profile_records=command_records,command_profile_groups=command_groups,
                       command_profile_invalid=invalid_lines)
            if arm=='on':
                summary, = command_summary
                fields=dict(x.split('=',1) for x in summary.removeprefix('strata command profile: ').split())
                assert fields['windows']==fields['expected']=='4'
                invalid_count=int(fields['invalid'])
                assert invalid_count==int(fields['invalid_saved'])==len(invalid_lines)<=32
                assert (fields['complete']=='1' and fields['error']=='none') if invalid_count==0 else (fields['complete']=='0' and fields['error']=='timestamp_order_granularity_or_query_delay')
                rec['profile_qualified']=fields['complete']=='1'
                timer=int(fields['timer_resolution_ns'])
                for line in invalid_lines:
                    bad=dict(item.split('=',1) for item in line.removeprefix('strata command invalid: ').split())
                    submitted,start,end,wall=(int(bad[k]) for k in ('submit','start','end','query_wall_ns'))
                    mask=int(start<submitted) | (int(end<=start)<<1) | (int(end>start and end-start<timer)<<2) | (int(wall>=1000000000)<<3)
                    assert mask==int(bad['violations']) and mask!=0
                assert 0<int(fields['records'])==len(command_records)<=640 and int(fields['peak_events'])<=160
                assert command_groups and all(' row=' in line for line in command_records+invalid_lines)
                groups = [dict(item.split('=',1) for item in line.removeprefix('strata command group: ').split())
                          for line in command_groups]
                selected_rows=[]
                for prefix,entries in [('strata command sample: ',command_records),('strata command invalid: ',invalid_lines)]:
                    selected_rows.extend(dict(item.split('=',1) for item in line.removeprefix(prefix).split()) for line in entries)
                assert len(selected_rows)==sum(int(g['selected']) for g in groups)<=640
                for g in groups:
                    assert sum(all(row[k]==g[k] for k in ('window','kind','bin')) for row in selected_rows)==int(g['selected'])
                phase_kinds = {'upload','dequant_gu','dequant_down','gather','gu','swiglu','down','combine','write'}
                assert {int(g['window']) for g in groups} == {0,1,2,3}
                assert {int(g['layer']) for g in groups} == {0,15,31,47}
                assert all(int(g['first_row']) == 0 and 0 < int(g['selected']) <= int(g['eligible']) for g in groups)
                for window in range(4):
                    own = [g for g in groups if int(g['window']) == window]
                    assert len({g['layer'] for g in own}) == 1
                    assert {g['kind'] for g in own} == phase_kinds
                    for kind, expected in [('upload',2),('combine',1),('write',2)]:
                        fixed, = [g for g in own if g['kind'] == kind]
                        assert int(fixed['eligible']) == int(fixed['selected']) == expected
                    jobs = {kind:{int(g['bin']):int(g['eligible']) for g in own if g['kind']==kind}
                            for kind in ('gather','gu','swiglu','down')}
                    assert all(population == jobs['gather'] for population in jobs.values())
                    gu_dq, = [g for g in own if g['kind']=='dequant_gu']
                    down_dq, = [g for g in own if g['kind']=='dequant_down']
                    assert gu_dq['eligible'] == down_dq['eligible']
                rec.update(command_profile_summary=summary,command_profile_records=command_records,command_profile_groups=command_groups)
            else:assert not command_summary and not command_records and not command_groups
            assert sha(external) == reference['external_profile_sha256']
            verify_pins()
            rec.update(passed=True, complete=True, GPU_executed=True, raw_capture_complete=True)
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
            for arm in dict(order):
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
