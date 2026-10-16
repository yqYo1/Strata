#!/usr/bin/env python3
"""Separate current BF16-prefix queue/CPU diagnostic, official bench.sh actual32K+."""
import argparse
import fcntl
import importlib.util
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
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=['profile'])
    ap.add_argument('arm',choices=['bf16-prefix'])
    ap.add_argument('repetition',type=int,choices=[0])
    ap.add_argument('--prior',type=Path,required=True)
    ap.add_argument('--prior-sha256',required=True)
    opts=ap.parse_args()
    arm,repetition=opts.arm,opts.repetition
    order=[('profile','bf16-prefix',0)]
    require((opts.phase,arm,repetition) in order,'declared independent-process order')
    position=order.index((opts.phase,arm,repetition))
    dtype='bf16'
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refs = {name: json.loads((A / name).read_text()) for name in (
            'own-qsa-bf16-prefix-engine-build-v1.json', 'own-qsa-bf16-prefix-model-smoke-v1.json',
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
        build['source_pins']={str(Path(n).relative_to(P)):v['sha256'] for n,v in build['frozen_files'].items() if Path(n).is_relative_to(P)}
        build['pins']=build['frozen_files'];build['binary_sha256']=build['binary']['sha256'];build['binary']=build['binary']['path']
        require(smoke['binary_sha256']==build['binary_sha256'] and
                set(smoke['cases'])=={'existing','bf16','bf16-prefix'},'same-binary three-arm mechanics')
        package=A/'own-vllm-xe2-qualification-v1'
        meta=json.loads((package/'manifest.json').read_text())
        require(sha(package/'manifest.json')=='7027faaa150d65f8581aebef1e731afe858288792dcedf8fb4409991806e1b58','reviewed manifest')
        helper=package/'main_controller.py'
        require(sha(helper)==meta['package_files']['main_controller.py']['sha256'],'reviewed helper')
        spec=importlib.util.spec_from_file_location('vllm_long_helpers',helper)
        h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
        prior,prior_gate=h.receipt_gate(module,opts.prior,opts.prior_sha256,boot_id(),qualification=True)
        cohort_path=A/'qsa-bf16-prefix-clean-cohort-v1.json'
        cohort=json.loads(cohort_path.read_text())
        require(cohort['passed'] and cohort['complete'] and cohort['active_controller'] is None and len(cohort['completed'])==9,'all clean samples closed before profile')
        for item in cohort['completed']:
            require(sha(Path(item['path']))==item['sha256'],'clean sample hash')
            closed(json.loads(Path(item['path']).read_text()))
        require(prior['schema']=='strata-qsa-bf16-prefix-long-v1' and prior['phase']=='clean' and prior['arm']==arm and prior['repetition']==2 and prior['binary_sha256']==build['binary_sha256'],'latest same-binary clean gate')
        for phase,a,n in order[:position]:
            closed(json.loads((A/f'own-qsa-bf16-prefix-{phase}-{a}-r{n}-v1.json').read_text()))
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
        root=B/f'main-qsa-bf16-prefix-{opts.phase}-{arm}-r{repetition}-v1'
        target=A/f'own-qsa-bf16-prefix-{opts.phase}-{arm}-r{repetition}-v1.json'
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
            STRATA_SYCL_PREFILL_NATIVE16='bf16',STRATA_SYCL_PREFILL_QSA_MATH='existing' if arm=='current' else arm,
            STRATA_PREFILL_LAYER_MAJOR_R_DTYPE=dtype,
            STRATA_SYCL_PREFILL_NATIVE16_BACKEND='vllm',
            STRATA_SYCL_PREFILL_NATIVE16_QSA='0' if arm=='current' else '1',
            STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE='auto',STRATA_ROPE_TABLE='1')
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
        if opts.phase=='profile':
            env.update(STRATA_TRACE='1',STRATA_PREFILL_LAYER_MAJOR_PROFILE='1',
                STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE='1',STRATA_PREFILL_CPU_WAIT_PROFILE='1',
                UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
        template = WRAPPER.read_text().replace("assert env['STRATA_SYCL_PROMPT_DIRECT'] == '0'",
                                              "assert env['STRATA_SYCL_PROMPT_DIRECT'] == '1'")
        template = re.sub(r'^ENGINE = .*?$', 'ENGINE = ' + repr(str(exe)), template, flags=re.M)
        template = re.sub(r'^ENGINE_SHA256 = .*?$', 'ENGINE_SHA256 = ' + repr(sha(exe)), template, flags=re.M)
        keys = ('STRATA_SYCL_PREFILL_NATIVE16', 'STRATA_PREFILL_LAYER_MAJOR_R_DTYPE',
                'STRATA_PREFILL_LAYER_MAJOR_R_GPU', 'STRATA_SYCL_SELECT_RADIX4', 'STRATA_SYCL_PROMPT_DIRECT',
                'STRATA_SYCL_PREFILL_NATIVE16_BACKEND','STRATA_SYCL_PREFILL_NATIVE16_QSA','STRATA_SYCL_PREFILL_QSA_MATH',
                'STRATA_SYCL_PREFILL_NATIVE16_VLLM_TILE','STRATA_ROPE_TABLE',
                'STRATA_TRACE','STRATA_PREFILL_LAYER_MAJOR_PROFILE','STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE',
                'STRATA_PREFILL_CPU_WAIT_PROFILE','UR_ENABLE_LAYERS')
        anchor = "    pipe = os.stat('/proc/' + str(child.pid) + '/fd/0')"
        require(template.count(anchor) == 1, 'actual environment capture seam')
        template = template.replace(anchor, "    entry['experiment_environment'] = {k:env[k] for k in " + repr(keys) + "}\n" + anchor)
        wrapper = root / 'engine-wrapper.py'; wrapper.write_text(template); wrapper.chmod(0o700)
        args = list(old['config']['args'])
        for flag in ['--max-context','--kv-resident']:
            args[args.index(flag)+1]='65536'
        profile = Path(args[args.index('--expert-profile') + 1])
        local = root / 'expert-profile.bin'; shutil.copyfile(profile, local)
        args[args.index('--expert-profile') + 1] = str(local)
        config = dict(old['config'], exe=str(wrapper), args=args, env={}, log=str(root / 'owned/engine.stderr'))
        options = list(old['benchmark_options'])
        options[options.index('--prefill-tokens')+1]='32768'
        (root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        (root / 'plan.json').write_text(json.dumps(dict(port=18630 + position,
            url=f'http://127.0.0.1:{18630 + position}', runs=1, benchmark_options=options,
            bench_timeout_seconds=1500), indent=2) + '\n')
        record = dict(schema='strata-qsa-bf16-prefix-profile-v1', passed=False, complete=False, active=True,phase=opts.phase,
            commands=[], arm=arm, repetition=repetition, measurement_order=position, declared_order=order,
            source_commit=build['source_commit'], source_pins=build['source_pins'], binary_sha256=sha(exe),
            recipe_sha256=sha(Path(__file__)), wrapper_sha256=sha(wrapper),
            adapted_from=dict(path='run_qsa_bf16_prefix_long_v1.py',sha256='d4ac64ecb52c6eb4d9fcb95956530cfbcf27de7217f06ef3c39f3658c099a1b6'),
            clean_cohort_sha256=sha(cohort_path),
            references={n: sha(A / n) for n in refs}, environment=env, config=config,
            runtime_files=runtime['files'], benchmark_source=old['benchmark_source'], benchmark_options=options,
            model_identity=old['model_identity'], pack_metadata_sha256=old['pack_metadata_sha256'],
            MTP_metadata_sha256=old['MTP_metadata_sha256'], external_profile_sha256=sha(profile),
            local_profile_initial_sha256=sha(local), boot_id=boot_id(),
            prior_gate=prior_gate,performance_qualified=False,quality_qualified=False,physical_262144_qualified=False,
            retention=dict(owner='Main',consumer='current faster-path resource and queue interval attribution',raw_byte_budget=1<<30,review='at paired closure; retain individual samples and compact results, retire successful full protocol'),
            scope='Separate official bench.sh actual32K+ fresh input with BF16-prefix, same current binary/context65536/chunk8192/GPU32768 residuals. TRACE, parameter validation, CPU/wait clocks and coarse/QSA queue markers enabled. Nonexclusive queue intervals include host gaps; do not infer PCIe/VRAM/XMX saturation or GPU hardware utilization. No clean performance, quality/full256K/RESTORE qualification.')
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
            h.no_model_owners(meta['known_model_files']);h.driver_guard()
            for path,value in build['pins'].items(): h.pinned(path,value)
            require((DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists(), 'GPU preflight')
            cursor = re.search(r'-- cursor: (\S+)', utility('journal-before',
                ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager']))[1]
            c, out, err = owner.run('prefill-only', [PY, str(RUNNER), str(root), 'extended-bench'], env, P,
                wall=1800, cpu=21600, rss_cap=88 << 30, text_cap=64 << 20, total_cap=1 << 30, file_cap=64 << 20)
            require(module.closed(c) and c['exit_code'] == 0, 'runner: ' + err.read_text()[-5000:])
            result = json.loads((root / 'result.json').read_text()); record['result'] = result
            require(result['passed'] and result['server_exit_code'] == result['real_engine_exit']['exit_code'] == 0, 'native/server normal exit')
            require(result['real_engine_exit']['binary_sha256'] == sha(exe), 'actual binary')
            record['requested_QSA_math']=env['STRATA_SYCL_PREFILL_QSA_MATH']
            require(result['real_engine_exit']['experiment_environment'] == {k: env[k] for k in keys}, 'actual arm')
            suites = list((root / 'stock-results').glob('*/results.json')); require(len(suites) == 1, 'one suite')
            suite = json.loads(suites[0].read_text()); cohort = suite['results']['prefill_only']
            require(suite['complete'] and not suite['results']['failures'] and cohort['complete'] and
                cohort['eligible_runs'] == len(cohort['requests']) == 1, 'complete measured request')
            require(suite['params']['think'] and suite['params']['mode'] == 'prefill-only', 'thinking and workload')
            require(all(r['ok'] and r['eligible'] and r['actual_prompt_tokens']>=32768 and
                r['actual_read_tokens']==r['actual_prompt_tokens'] and r['actual_reused_tokens'] == 0 and
                r['generated_tokens'] == 24 and r['thinking_requested'] for r in cohort['requests']), 'actual uncached long input')
            record['benchmark_result'] = suite
            check = json.loads(utility('verify-request-journal', ['/usr/bin/python3', '-B', '-c',
                'import sys,json;sys.path.insert(0,"tools");from strata_bench import verify_request_journal;print(json.dumps(verify_request_journal(sys.argv[1])))', str(suites[0].parent)]))
            require(check['valid'] and check['complete'] and not check['pending'], 'actual request identity')
            record['request_journal_verification'] = check
            native = (root / 'owned/engine.stderr').read_text()
            require(not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:|retirement failed', native), 'native failure')
            placement = [l for l in native.splitlines() if l.startswith('strata prefill: layer-major candidate,')]
            require(len(placement) == 1 and f'chunk=8192 residual_dtype={dtype} ' in placement[0] and
                'residual_gpu_rows=32768 ' in placement[0], 'actual matched placement, no backoff')
            require(('strata prefill: native16 BF16 HC residual/projection IO' in native)==True,'actual native16 path')
            require('[native16-route] engine=bf16 backend=vllm' in native,'actual backend')
            arithmetic_lines=[l for l in native.splitlines() if l.startswith('strata prefill: QSA BF16 operands,')]
            require(len(arithmetic_lines)==(0 if arm=='current' else 1),'actual QSA arithmetic path')
            if arm!='current':require(('KV=active-logical-prefix' if arm=='bf16-prefix' else 'KV=on-the-fly') in arithmetic_lines[0],'actual BF16 KV path')
            record['QSA_arithmetic_lines']=arithmetic_lines
            record['coarse_profile_lines']=[l for l in native.splitlines() if l.startswith('strata lm profile:')]
            record['qsa_profile_lines']=[l for l in native.splitlines() if l.startswith('strata lm qsa profile:')]
            record['cpu_wait_lines']=[l for l in native.splitlines() if l.startswith('strata prefill cpu wait:')]
            record['cpu_host_lines']=[l for l in native.splitlines() if l.startswith('strata prefill cpu host:')]
            require(any('complete=1 errors=0' in l for l in record['coarse_profile_lines']),'complete coarse queue profile')
            require(any('complete=1 errors=0' in l for l in record['qsa_profile_lines']),'complete QSA queue profile')
            require(len(record['cpu_wait_lines'])==len(record['cpu_host_lines'])==1 and 'available=1 complete=1' in record['cpu_wait_lines'][0] and 'available=1 open_regions=0' in record['cpu_host_lines'][0],'closed CPU/wait accounting')
            completed=list(map(int,re.findall(r'layer-major layer (\d+) of 48: completed \d+ chunks',native)))
            require(completed==list(range(48)),'all48 traced layer completion')
            record['interpretation']='CLOCK_THREAD_CPUTIME_ID for issuing thread; waits subtracted from host regions. Coarse/QSA event intervals are nonexclusive with host gaps. No per-kernel physical memory, PCIe wire or execution-unit counters.'
            record['placement_lines'] = placement
            record['native_log_sha256'] = sha(root / 'owned/engine.stderr')
            require(sha(profile) == record['external_profile_sha256'], 'user profile unchanged')
            require(all(sha(P / n) == h for n, h in build['source_pins'].items()), 'source freeze')
            for path,value in build['pins'].items(): h.pinned(path,value)
            h.driver_guard()
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
        finally:
            try:
                if cursor and owner.active is None:
                    rows = [json.loads(s) for s in utility('journal-final', ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json']).splitlines() if s.strip()]
                    record.update(fresh_faults=[v for v in rows if FAULT.search(str(v.get('MESSAGE', '')))],
                        boot_unchanged=boot_id() == record['boot_id'], driver_dump_exists=(DEVICE / 'devcoredump').exists())
            finally:
                record.update(active=owner.active is not None,GPU_executed=(root/'engine-start.json').exists(),finished_utc=module.utc(),ownership_closed=owner.active is None and all(module.closed(c) for c in owner.commands))
                if record['active'] or not all(module.closed(c) for c in owner.commands) or 'fresh_faults' not in record or record.get('fresh_faults') or not record.get('boot_unchanged') or record.get('driver_dump_exists'):
                    record.update(passed=False, complete=False)
                record['performance_qualified']=bool(record['passed'] and record['complete'] and opts.phase=='clean')
                persist()
                with target.open('x') as f:
                    json.dump(record, f, indent=2); f.write('\n')
        print(json.dumps(dict(passed=record['passed'], arm=arm, repetition=repetition,
            prompt_tps=record.get('benchmark_result',{}).get('results',{}).get('prefill_only',{}).get('prompt_tps'),receipt=str(target),sha256=sha(target))),flush=True)
        return 0 if record['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
