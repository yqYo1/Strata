#!/usr/bin/env python3
"""Same-revision native16/FP32 clean official bench.sh independent processes."""
import fcntl
import json
from pathlib import Path
import re
import shutil
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
PY = '/home/yayoi/.local/share/strata-sycl/venv/bin/python'
RUNNER = P / 'bench/results/2026-10-14-long-generation-benchmark/run_server_job.py'
WRAPPER = A / 'qsa-device-batch-model-wrapper-v1/engine_wrapper.py'
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha, boot_id


def main():
    require(len(sys.argv) == 4 and sys.argv[1] in ('v2', 'v3') and
            sys.argv[2] in ('fp32', 'bf16') and sys.argv[3] in ('0', '1', '2'),
            'fresh build version, native arm, repetition')
    version, arm, repetition = sys.argv[1], sys.argv[2], int(sys.argv[3])
    order = [('fp32', 0), ('bf16', 0), ('bf16', 1), ('fp32', 1), ('fp32', 2), ('bf16', 2)]
    position = order.index((arm, repetition))
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refs = {name: json.loads((A / name).read_text()) for name in (
            f'own-native16-prefill-build-{version}.json', f'own-native16-model-smoke-{version}.json',
            'own-selector-clean-1-r2-v1.json', 'bench-prefill-only-host-v2.json')}
        build, smoke, old, host = refs.values()
        def closed(v):
            require(v['passed'] and v['complete'] and not v['active'], 'prior gate')
            require(not v.get('fresh_faults') and not v.get('driver_dump_exists'), 'prior GPU fault')
            for c in v['commands']:
                require(module.closed(c) and c['exit_code'] == 0, 'prior normal closure')
                for identity in c.get('owners', []):
                    now = module.identity(identity['pid'])
                    require(now is None or now['start_ticks'] != identity['start_ticks'], 'prior owner present')
        for v in refs.values():
            closed(v)
        require(smoke['binary_sha256'] == build['binary_sha256'] and
                set(smoke['cases']) == {'fp32', 'bf16'}, 'same-binary two-arm mechanics')
        for a, n in order[:position]:
            closed(json.loads((A / f'own-native16-clean-{a}-r{n}-{version}.json').read_text()))
        require(all(sha(P / n) == h for n, h in build['source_pins'].items()), 'entire source freeze')
        exe = Path(build['binary'])
        require(sha(exe) == build['binary_sha256'], 'native binary')
        for n, v in host['source_pins'].items():
            require(sha(P / n) == v['proposed_sha256'], 'qualified benchmark client')
        require(sha(P / 'tools/test_benchmark_evidence.py') == host['evidence_fixture_sha256'], 'client fixture')
        runtime = json.loads((A / 'own-qsa-select-model-a1-library-hashes-v1.json').read_text())
        require(all(sha(Path(v['path'])) == v['sha256'] for v in runtime['files']), 'runtime files')
        require(all(sha(P / n) == h for n, h in old['benchmark_source'].items()), 'official benchmark source')
        off = json.loads((A / 'selective-iq-clean-bench-full-0-v1.json').read_text())
        require(all(sha(P / n) == h for n, h in off['recipes'].items()), 'runner source')
        for field in ('pack_metadata_sha256', 'MTP_metadata_sha256'):
            require(all(sha(Path(n)) == h for n, h in old[field].items()), 'model metadata')
        for n, v in old['model_identity'].items():
            st = Path(n).stat()
            require(st.st_size == v['bytes'] and st.st_mtime_ns == v['mtime_ns'], 'model identity')
        root = B / f'main-own-native16-clean-{arm}-r{repetition}-root-{version}'
        target = A / f'own-native16-clean-{arm}-r{repetition}-{version}.json'
        require(not root.exists() and not target.exists(), 'fresh independent process')
        root.mkdir(mode=0o700)
        env = {k: v for k, v in old['environment'].items() if k.startswith('STRATA_') or k in (
            'PATH', 'HOME', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'MKLROOT', 'OMP_NUM_THREADS',
            'CPLUS_INCLUDE_PATH', 'ONEAPI_DEVICE_SELECTOR', 'UR_ADAPTERS_FORCE_LOAD', 'SYCL_CACHE_PERSISTENT',
            'NEOReadDebugKeys', 'EnableDirectSubmission', 'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD',
            'PYTHONDONTWRITEBYTECODE', 'PYTHONUNBUFFERED')}
        env.update(XDG_CACHE_HOME=str(root / 'cache'),
            STRATA_MEASUREMENT_ENGINE_START=str(root / 'engine-start.json'),
            STRATA_MEASUREMENT_ENGINE_EXIT=str(root / 'engine-exit.json'),
            STRATA_SYCL_PREFILL_NATIVE16='bf16' if arm == 'bf16' else '0',
            STRATA_PREFILL_LAYER_MAJOR_R_DTYPE=arm)
        for k, v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', NEOReadDebugKeys='1',
                EnableDirectSubmission='0', STRATA_SYCL_PREFILL_SAME_QUEUE='1', SYCL_CACHE_PERSISTENT='0',
                STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0', STRATA_PREFILL_LAYER_MAJOR='1',
                STRATA_PREFILL_LAYER_MAJOR_R_GPU='32768', STRATA_PREFILL_RING='8').items():
            require(env[k] == v, 'qualified runtime/admission configuration: ' + k)
        require(not any(k.startswith(('UNITRACE_', 'PTI_', 'UR_LOG_', 'ZE_ENABLE_', 'ZEL_', 'IGC_')) or
                k in ('UR_ENABLE_LAYERS', 'LD_PRELOAD', 'MKL_VERBOSE', 'STRATA_TRACE', 'STRATA_PREFILL_NLL',
                      'STRATA_PREFILL_LOGITS', 'STRATA_PREFILL_TIMING', 'STRATA_DECODE_TIMING') for k in env), 'clean timing')
        for k in ('STRATA_PREFILL_BOUNDARY_AUDIT', 'STRATA_PREFILL_PHASE_AUDIT',
                'STRATA_PREFILL_LAYER_MAJOR_PROFILE', 'STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE',
                'STRATA_PREFILL_LAYER_MAJOR_SHAPE_CENSUS', 'STRATA_IQ_TIMING_PROGRESS'):
            require(env.get(k, '0') == '0', 'disabled diagnostic: ' + k)
        template = WRAPPER.read_text().replace("assert env['STRATA_SYCL_PROMPT_DIRECT'] == '0'",
                                              "assert env['STRATA_SYCL_PROMPT_DIRECT'] == '1'")
        template = re.sub(r'^ENGINE = .*?$', 'ENGINE = ' + repr(str(exe)), template, flags=re.M)
        template = re.sub(r'^ENGINE_SHA256 = .*?$', 'ENGINE_SHA256 = ' + repr(sha(exe)), template, flags=re.M)
        keys = ('STRATA_SYCL_PREFILL_NATIVE16', 'STRATA_PREFILL_LAYER_MAJOR_R_DTYPE',
                'STRATA_PREFILL_LAYER_MAJOR_R_GPU', 'STRATA_SYCL_SELECT_RADIX4', 'STRATA_SYCL_PROMPT_DIRECT')
        anchor = "    pipe = os.stat('/proc/' + str(child.pid) + '/fd/0')"
        require(template.count(anchor) == 1, 'actual environment capture seam')
        template = template.replace(anchor, "    entry['experiment_environment'] = {k:env[k] for k in " + repr(keys) + "}\n" + anchor)
        wrapper = root / 'engine-wrapper.py'; wrapper.write_text(template); wrapper.chmod(0o700)
        args = list(old['config']['args'])
        profile = Path(args[args.index('--expert-profile') + 1])
        local = root / 'expert-profile.bin'; shutil.copyfile(profile, local)
        args[args.index('--expert-profile') + 1] = str(local)
        config = dict(old['config'], exe=str(wrapper), args=args, env={}, log=str(root / 'owned/engine.stderr'))
        options = list(old['benchmark_options'])
        (root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        (root / 'plan.json').write_text(json.dumps(dict(port=18430 + position,
            url=f'http://127.0.0.1:{18430 + position}', runs=1, benchmark_options=options,
            bench_timeout_seconds=1500), indent=2) + '\n')
        record = dict(schema='strata-native16-clean-v1', passed=False, complete=False, active=True,
            commands=[], arm=arm, repetition=repetition, measurement_order=position, declared_order=order,
            source_commit=build['source_commit'], source_pins=build['source_pins'], binary_sha256=sha(exe),
            recipe_sha256=sha(Path(__file__)), wrapper_sha256=sha(wrapper),
            references={n: sha(A / n) for n in refs}, environment=env, config=config,
            runtime_files=runtime['files'], benchmark_source=old['benchmark_source'], benchmark_options=options,
            model_identity=old['model_identity'], pack_metadata_sha256=old['pack_metadata_sha256'],
            MTP_metadata_sha256=old['MTP_metadata_sha256'], external_profile_sha256=sha(profile),
            local_profile_initial_sha256=sha(local), boot_id=boot_id(),
            performance_qualified=False, quality_qualified=False, physical_262144_qualified=False,
            scope='Official bench.sh fresh 78453-token input, thinking ON, 24 output cap, chunk8192. Same binary, matched GPU32768/host remainder residual row placement. Native BF16 HC/MoE vs FP32, separate processes, no tracing/validation/numerical capture. No quality/fullcontext/RESTORE or decode-speed qualification.')
        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(root / 'owned', persist); record['commands'] = owner.commands; persist()
        hostenv = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
        def utility(label, argv):
            c, out, err = owner.run(label, argv, hostenv, P, wall=60, cpu=120, rss_cap=2 << 30,
                text_cap=8 << 20, total_cap=1 << 30, file_cap=8 << 20)
            require(module.closed(c) and c['exit_code'] == 0, label + ': ' + err.read_text()[-2000:])
            return out.read_text()
        cursor = None
        try:
            require((DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists(), 'GPU preflight')
            cursor = re.search(r'-- cursor: (\S+)', utility('journal-before',
                ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager']))[1]
            c, out, err = owner.run('prefill-only', [PY, str(RUNNER), str(root), 'extended-bench'], env, P,
                wall=1800, cpu=21600, rss_cap=88 << 30, text_cap=64 << 20, total_cap=1 << 30, file_cap=64 << 20)
            require(module.closed(c) and c['exit_code'] == 0, 'runner: ' + err.read_text()[-5000:])
            result = json.loads((root / 'result.json').read_text()); record['result'] = result
            require(result['passed'] and result['server_exit_code'] == result['real_engine_exit']['exit_code'] == 0, 'native/server normal exit')
            require(result['real_engine_exit']['binary_sha256'] == sha(exe), 'actual binary')
            require(result['real_engine_exit']['experiment_environment'] == {k: env[k] for k in keys}, 'actual arm')
            suites = list((root / 'stock-results').glob('*/results.json')); require(len(suites) == 1, 'one suite')
            suite = json.loads(suites[0].read_text()); cohort = suite['results']['prefill_only']
            require(suite['complete'] and not suite['results']['failures'] and cohort['complete'] and
                cohort['eligible_runs'] == len(cohort['requests']) == 1, 'complete measured request')
            require(suite['params']['think'] and suite['params']['mode'] == 'prefill-only', 'thinking and workload')
            require(all(r['ok'] and r['eligible'] and r['actual_prompt_tokens'] == 78453 and
                r['actual_read_tokens'] == 78453 and r['actual_reused_tokens'] == 0 and
                r['generated_tokens'] == 24 and r['thinking_requested'] for r in cohort['requests']), 'actual uncached long input')
            record['benchmark_result'] = suite
            check = json.loads(utility('verify-request-journal', ['/usr/bin/python3', '-B', '-c',
                'import sys,json;sys.path.insert(0,"tools");from strata_bench import verify_request_journal;print(json.dumps(verify_request_journal(sys.argv[1])))', str(suites[0].parent)]))
            require(check['valid'] and check['complete'] and not check['pending'], 'actual request identity')
            record['request_journal_verification'] = check
            native = (root / 'owned/engine.stderr').read_text()
            require(not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:|retirement failed', native), 'native failure')
            placement = [l for l in native.splitlines() if l.startswith('strata prefill: layer-major candidate,')]
            require(len(placement) == 1 and f'chunk=8192 residual_dtype={arm} ' in placement[0] and
                'residual_gpu_rows=32768 ' in placement[0], 'actual matched placement, no backoff')
            require(('strata prefill: native16 BF16 HC residual/projection IO' in native) == (arm == 'bf16'), 'actual native16 path')
            record['placement_lines'] = placement
            record['native_log_sha256'] = sha(root / 'owned/engine.stderr')
            require(sha(profile) == record['external_profile_sha256'], 'user profile unchanged')
            require(all(sha(P / n) == h for n, h in build['source_pins'].items()), 'source freeze')
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if cursor and owner.active is None:
                    rows = [json.loads(s) for s in utility('journal-final', ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json']).splitlines() if s.strip()]
                    record.update(fresh_faults=[v for v in rows if FAULT.search(str(v.get('MESSAGE', '')))],
                        boot_unchanged=boot_id() == record['boot_id'], driver_dump_exists=(DEVICE / 'devcoredump').exists())
            finally:
                record.update(active=owner.active is not None, GPU_executed=(root / 'engine-start.json').exists(), finished_utc=module.utc())
                if record['active'] or not all(module.closed(c) for c in owner.commands) or 'fresh_faults' not in record or record.get('fresh_faults') or not record.get('boot_unchanged') or record.get('driver_dump_exists'):
                    record.update(passed=False, complete=False)
                record['performance_qualified'] = bool(record['passed'] and record['complete'])
                persist()
                with target.open('x') as f:
                    json.dump(record, f, indent=2); f.write('\n')
        print(json.dumps(dict(passed=record['passed'], arm=arm, repetition=repetition,
            prompt_tps=record['benchmark_result']['results']['prefill_only']['prompt_tps'])), flush=True)


if __name__ == '__main__':
    main()
