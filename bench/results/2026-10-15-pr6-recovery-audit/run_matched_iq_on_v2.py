#!/usr/bin/env python3
"""Main's matched ON full suite; preserves the exact closed historical OFF.

No run is allowed until host checks, nonce provenance, and the reviewed client
are real. Fixed actual inputs are paired after normal native/server closure;
natural agent feedback remains a distinct, nonpaired cohort.
"""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
PY = '/home/yayoi/.local/share/strata-sycl/venv/bin/python'
RUNNER = P / 'bench/results/2026-10-14-long-generation-benchmark/run_server_job.py'
WRAPPER = A / 'selective-iq-clean-bench-v1/engine_wrapper.py'
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha, boot_id


def good(record):
    return record.get('passed') and record.get('complete') and not record.get('active')


def main():
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        off_path = A / 'selective-iq-clean-bench-full-0-v1.json'
        host_path = A / 'bench-input-tools-host-v1.json'
        proof_path = A / 'selective-iq-full0-workload-provenance-v1.json'
        build_path = A / 'selective-iq-native-build-v1.json'
        off, host, proof, build = [json.loads(p.read_text()) for p in (off_path, host_path, proof_path, build_path)]
        require(all(good(x) for x in (off, host, proof, build)), 'real closed OFF/build/input-tool/provenance gates')
        require(off['mode'] == 'full' and off['arm'] == '0' and off['runs'] == 3, 'original full OFF identity')
        require(not any(v == '--clients' or v.startswith('--clients=') for v in off['benchmark_options']),
                'concurrent clients are outside this input-journal contract')
        require(not off['fresh_faults'] and off['boot_unchanged'] and not off['driver_dump_exists'], 'OFF fault gate')
        require(off['boot_id'] == boot_id(), 'boot changed between arms')
        for record in (off, host, build):
            require(all(module.closed(c) for c in record['commands']), 'earlier ownership incomplete')
            for c in record['commands']:
                for identity in c.get('owners', []):
                    current = module.identity(identity['pid'])
                    require(current is None or current['start_ticks'] != identity['start_ticks'], 'earlier owned identity present')
        require(host['original_benchmark_sha256'] == proof['original_receipt_sha256'] == sha(off_path), 'OFF changed since host/provenance assessment')
        require(host['original_client_sha256'] == off['benchmark_source']['tools/strata_bench.py'], 'unreviewed historical client')
        require(proof['benchmark_source'] == off['benchmark_source'], 'nonce client provenance mismatch')
        require(proof['journal_verification']['valid'] and proof['journal_verification']['complete'], 'nonce journal gate')
        require(re.fullmatch(r'[0-9]+', proof['recovered_nonce']) is not None, 'original timestamp tag required')
        require(all(sha(P / n) == h for n, h in host['proposal_inputs'].items()), 'tested proposal changed')
        require(sha(P / 'tools/strata_bench.py') == host['proposed_client_sha256'], 'reviewed client not actually applied')
        require({x['label']: x['tests'] for x in host['tests']} ==
                {'fixed-workload': 6, 'request-identity': 11}, 'host tests missing')
        require(all(sha(P / n) == h for n, h in build['source_pins'].items()), 'native source changed')
        require(all(sha(Path(n)) == h for n, h in build['linked_inputs'].items()), 'linked input changed')
        exe = Path(build['binary'])
        require(sha(exe) == build['binary_sha256'] == off['binary_sha256'], 'native must be identical to OFF')
        require(all(sha(P / n) == h for n, h in off['recipes'].items()), 'runner/wrapper changed')
        clients = {n: sha(P / n) for n in off['benchmark_source']}
        require({n for n in clients if clients[n] != off['benchmark_source'][n]} == {'tools/strata_bench.py'},
                'client delta exceeds reviewed fixed-ID/provenance change')
        require((DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists(), 'driver/dump preflight')
        for field in ('pack_metadata_sha256', 'MTP_metadata_sha256'):
            require(all(sha(Path(n)) == h for n, h in off[field].items()), 'model metadata changed')
        for n, identity in off['model_identity'].items():
            current = Path(n).stat()
            require(current.st_size == identity['bytes'] and current.st_mtime_ns == identity['mtime_ns'], 'model file identity changed')
        old_root = B / 'main-selective-iq-clean-bench-full-0-root-v1'
        require(len(off['request_journals']) == 1, 'one OFF request journal required')
        old_journal = old_root / off['request_journals'][0]['path']
        require(sha(old_journal) == proof['journal_sha256'] == off['request_journals'][0]['sha256'], 'OFF input journal changed/retired')
        root = B / 'main-selective-iq-clean-bench-full-1-root-v2'
        receipt = A / 'selective-iq-clean-bench-full-1-v2.json'
        require(not root.exists() and not receipt.exists(), 'fresh matched ON root required')
        root.mkdir(mode=0o700)
        env = dict(off['environment'], STRATA_SYCL_PARALLEL_IQ_DEQUANT='1', XDG_CACHE_HOME=str(root / 'cache'),
                   STRATA_MEASUREMENT_ENGINE_START=str(root / 'engine-start.json'),
                   STRATA_MEASUREMENT_ENGINE_EXIT=str(root / 'engine-exit.json'))
        allowed = {'STRATA_SYCL_PARALLEL_IQ_DEQUANT', 'XDG_CACHE_HOME', 'STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT'}
        require({k for k in set(env) | set(off['environment']) if env.get(k) != off['environment'].get(k)} == allowed,
                'unmatched engine environment')
        require(not any('TOKEN' in k or 'API_KEY' in k or k.startswith(('UNITRACE_', 'PTI_', 'UR_LOG_', 'ZE_ENABLE_', 'ZEL_')) or
                        k in ('UR_ENABLE_LAYERS', 'LD_PRELOAD', 'MKL_VERBOSE', 'STRATA_TRACE', 'STRATA_PREFILL_TIMING', 'STRATA_DECODE_TIMING') for k in env),
                'clean environment required')
        for k, v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', NEOReadDebugKeys='1', EnableDirectSubmission='0',
                         STRATA_SYCL_PREFILL_SAME_QUEUE='1', SYCL_CACHE_PERSISTENT='0', STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0').items():
            require(env[k] == v, 'safe runtime flag changed')
        config = dict(off['config'], exe=str(WRAPPER), log=str(root / 'owned/engine.stderr'),
                      env={'STRATA_SYCL_PROMPT_DIRECT': '0', 'STRATA_SYCL_PARALLEL_IQ_DEQUANT': '1'})
        require(config['args'] == off['engine_args'], 'native arguments changed')
        options = [*off['benchmark_options'], '--workload-id', proof['recovered_nonce']]
        wall = off['limits']['wall_seconds']
        (root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        (root / 'plan.json').write_text(json.dumps(dict(port=18097, url='http://127.0.0.1:18097', runs=3,
            benchmark_options=options, bench_timeout_seconds=wall - 300), indent=2) + '\n')
        comparator = A / 'bench-request-identity-v2/compare_requests.py'
        record = dict(schema='matched-selective-iq-full-v2', active=True, passed=False, complete=False,
            GPU_executed=False, commands=[], arm='1', mode='full', runs=3, environment=env, config=config,
            engine_args=off['engine_args'], benchmark_options=options, binary_sha256=sha(exe), boot_id=boot_id(),
            source_pins=build['source_pins'], linked_inputs=build['linked_inputs'], benchmark_source=clients,
            recipes=off['recipes'], recipe_sha256=sha(Path(__file__)), reference_receipt_sha256=sha(off_path),
            host_tools_receipt_sha256=sha(host_path), workload_provenance_sha256=sha(proof_path),
            original_benchmark_source=off['benchmark_source'], fixed_workload_id=proof['recovered_nonce'],
            model_identity=off['model_identity'], pack_metadata_sha256=off['pack_metadata_sha256'], MTP_metadata_sha256=off['MTP_metadata_sha256'],
            input_comparator_sha256=sha(comparator), limits=off['limits'], input_match_qualified=False,
            fixed_input_pair_qualified=False, agent_static_tasks_qualified=False, agent_exact_input_paired_speed=False,
            adopted=False, performance_qualified=False, quality_qualified=False, physical_262144_qualified=False, restore_qualified=False,
            scope='Three official bench requests/sessions per cohort. Reviewed client delta only selects original tag and records provenance. '
                  'Fixed request bodies and all request ordering/parameters must match. Natural agent history may differ; '
                  'agent results remain nonpaired and static task metadata/builders are checked separately. '
                  'This is not independent-process variance or quality qualification.',
            retention_consumer='Full OFF/ON input journal comparison and individual speed/work samples. Retire duplicate successful text only after verified extraction.')
        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(root / 'owned', persist)
        record['commands'] = owner.commands
        host_env = dict(PATH='/usr/bin:/bin', HOME=str(Path.home()), LANG='C.UTF-8', LC_ALL='C.UTF-8',
                        PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
        def utility(label, argv):
            require(owner.active is None, 'utility cannot overlap unresolved native owner')
            c, out, err = owner.run(label, argv, host_env, P, wall=60, cpu=60, rss_cap=2 << 30,
                                    text_cap=8 << 20, total_cap=1 << 30, file_cap=8 << 20)
            require(module.closed(c) and c['exit_code'] == 0, label + ' failed: ' + err.read_text()[-3000:])
            return out.read_text()
        cursor = None
        persist()
        try:
            cursor = re.search(r'-- cursor: (\S+)', utility('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager']))[1]
            record['journal_before_cursor'] = cursor
            record['GPU_executed'] = True
            c, out, err = owner.run('bench-full-1', [PY, str(RUNNER), str(root), 'extended-bench'], env, P,
                                    wall=wall, cpu=wall * len(os.sched_getaffinity(0)), rss_cap=88 << 30,
                                    text_cap=64 << 20, total_cap=1 << 30, file_cap=64 << 20)
            result = json.loads((root / 'result.json').read_text())
            record['result'] = result
            require(module.closed(c) and c['exit_code'] == 0, 'ON native/server abnormal or incomplete')
            require(result['passed'] and result['server_exit_code'] == result['real_engine_exit']['exit_code'] == 0, 'native/server normal closure required')
            require(result['real_engine_exit']['binary_sha256'] == sha(exe) and
                    result['real_engine_exit']['argv'] == [str(exe), '--serve', *off['engine_args']], 'real native mismatch')
            paths = list((root / 'stock-results').glob('*/results.json'))
            journals = list((root / 'stock-results').glob('*/requests.jsonl'))
            require(len(paths) == len(journals) == 1, 'one ON suite/journal required')
            suite = json.loads(paths[0].read_text())
            record.update(benchmark_result=suite, benchmark_result_sha256=sha(paths[0]))
            require(suite['complete'] and not suite['results']['failures'], 'ON suite incomplete/failed')
            prompts = suite['results']['prompts']['32K']['requests']
            require(len(prompts) == 3 and all(r['ok'] and r['reused_tokens'] == 0 for r in prompts), '32K cohort')
            sessions = suite['results']['agent']['session_samples']
            require(len(sessions) == 3 and all(r['complete'] and r['completed_turns'] == 6 for r in sessions), 'agent cohort')
            gen = suite['results']['long_generation']
            require(all(gen['classification'][k] for k in ('qualified', 'strong_case_confirmed', 'weak_case_confirmed')), 'observed MTP classes')
            require(set(gen['workloads']) == {'regular-code', 'varied-prose'} and all(len(v['requests']) == 3 and
                    all(r['eligible'] for r in v['requests']) for v in gen['workloads'].values()), 'long generation cohort')
            comparison = root / 'input-comparison.json'
            utility('compare-all-inputs', ['/usr/bin/python3', '-B', str(comparator), str(old_journal), str(journals[0]), str(comparison)])
            inputs = json.loads(comparison.read_text())
            require(inputs['input_pair_qualified'] and inputs['fixed_inputs_match'] and inputs['agent_protocol_match'] and
                    inputs['status'] in ('REQUEST_INPUTS_MATCH', 'FIXED_INPUTS_MATCH_AGENT_FEEDBACK'),
                    'fixed inputs or actual request structure/parameters differ')
            spec = importlib.util.spec_from_file_location('qualified_input_comparator_v2', comparator)
            comparison_tool = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(comparison_tool)
            tasks = comparison_tool.compare_agent_tasks(off['benchmark_result'], suite)
            require(tasks['passed'], 'agent static file/task or generation metadata differ')
            record.update(input_match_qualified=inputs['inputs_match'], fixed_input_pair_qualified=True,
                          agent_static_tasks_qualified=True, agent_static_task_comparison=tasks,
                          input_comparison=inputs, input_comparison_sha256=sha(comparison),
                          nominal_32K_actual_input_tokens=[r['prompt_tokens'] for r in prompts],
                          nominal_32K_all_at_least_32768=all(r['prompt_tokens'] >= 32768 for r in prompts))
            require(all(sha(P / n) == h for n, h in clients.items()) and all(sha(P / n) == h for n, h in build['source_pins'].items()), 'source changed during suite')
            require(all(sha(P / n) == h for n, h in host['proposal_inputs'].items()), 'qualified input tools changed during suite')
            require(all(sha(Path(n)) == h for n, h in build['linked_inputs'].items()) and sha(exe) == record['binary_sha256'], 'binary/linked input changed')
            require(all(sha(P / n) == h for n, h in off['recipes'].items()) and sha(comparator) == record['input_comparator_sha256'], 'runner/comparator changed')
            require(sha(off_path) == record['reference_receipt_sha256'] and sha(host_path) == record['host_tools_receipt_sha256'] and sha(proof_path) == record['workload_provenance_sha256'], 'reference/review changed')
            native_log = (root / 'owned/engine.stderr').read_text()
            require(not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|strata trace:|iq-progress-v3 |strata lm census:|Strata SYCL cleanup:|retirement failed|stopping with owners live|strata cpu_gu_reuse_v2|strata decode joined begin', native_log), 'native error/observer in clean suite')
            record['native_log_sha256'] = sha(root / 'owned/engine.stderr')
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if cursor and owner.active is None:
                    entries = [json.loads(s) for s in utility('journal-final', ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json']).splitlines() if s.strip()]
                    record.update(fresh_faults=[r for r in entries if FAULT.search(str(r.get('MESSAGE', '')))],
                                  boot_unchanged=boot_id() == record['boot_id'], driver_dump_exists=(DEVICE / 'devcoredump').exists())
            except BaseException as error:
                record['final_fault_error'] = repr(error)
                raise
            finally:
                for key, path in [('result', root / 'result.json'), ('native_start', root / 'engine-start.json'), ('native_exit', root / 'engine-exit.json')]:
                    if path.exists():
                        try:
                            record[key] = json.loads(path.read_text())
                        except BaseException as error:
                            record.setdefault('partial_collection_errors', []).append(dict(path=str(path), error=repr(error)))
                record.update(active=owner.active is not None, finished_utc=module.utc())
                if record['active'] or not all(module.closed(c) for c in owner.commands) or 'fresh_faults' not in record or record.get('fresh_faults') or not record.get('boot_unchanged') or record.get('driver_dump_exists'):
                    record.update(passed=False, complete=False)
                persist()
                with receipt.open('x') as stream:
                    json.dump(record, stream, indent=2)
                    stream.write('\n')
        print(json.dumps(dict(passed=record['passed'], inputs_match=record['input_match_qualified'],
                              fixed_input_pair_qualified=record['fixed_input_pair_qualified'],
                              agent_exact_input_paired_speed=False, performance_adopted=False)))


if __name__ == '__main__':
    main()
