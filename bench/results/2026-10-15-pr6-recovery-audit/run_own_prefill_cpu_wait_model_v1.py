#!/usr/bin/env python3
"""Separate owned actual78K diagnostic of caller CPU inside explicit SYCL waits."""
import fcntl
import json
from pathlib import Path
import re
import sys

A=Path(__file__).resolve().parent;P=A.parents[2]
B=Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
DEVICE=Path('/sys/bus/pci/devices/0000:05:00.0')
PY='/home/yayoi/.local/share/strata-sycl/venv/bin/python'
RUNNER=P/'bench/results/2026-10-14-long-generation-benchmark/run_server_job.py'
WRAPPER=A/'qsa-device-batch-model-wrapper-v1/engine_wrapper.py'
FAULT=re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
sys.dont_write_bytecode=True
sys.path.insert(0,str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner,require,sha,boot_id


def main(arm, repetition):
    require((arm,repetition)==(1,0), "declared independent CPU diagnostic")
    module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        paths={k:A/n for k,n in dict(build='own-prefill-cpu-wait-build-v1.json',
               off='selective-iq-clean-bench-full-0-v1.json',closed='own-qsa-select-model-a1-v2.json',
               host='bench-prefill-only-host-v2.json',parity='own-qsa-select-model-a1-comparison-v2.json',
               full='own-qsa-select-scope-full-comparison-v2.json',component='own-qsa-select-scope-bench-comparison-v2.json').items()}
        refs={k:json.loads(p.read_text()) for k,p in paths.items()}
        build,off,prior,host=[refs[k] for k in ('build','off','closed','host')]
        basebuild=json.loads((A/'own-prefill-select-scope-build-v1.json').read_text())
        require(build['replaced_native_sha256']==basebuild['binary_sha256'] and len(build['host_cases'])==8 and len(build['gpu_cases'])==2,'actual observer host/smallGPU gates')
        require(all(x['passed'] and x['complete'] and not x['active'] for x in (build,off,host)), 'actual build/host gates')
        require(host['total_host_tests']==31,'host coverage')
        parity=refs['parity']
        require(parity['passed'] and parity['complete'] and not parity['active'] and
            parity['sampled_FP32_values_equal'] and parity['binary_sha256_after']==basebuild['binary_sha256'], 'exact model parity gate')
        require(build['canonical_policy_passed'] and basebuild['device_kernel_template_byte_identical'] and build['source_diff_sha256']=='1acb2c04be892d72174545a9945f3c2210a007bcbf7b0c9661b61dfac3cd74df', 'qualified original/prefill separation')
        require(refs['full']['passed'] and refs['full']['complete'] and refs['full']['all_full_bytes_equal']
                and refs['full']['API_queries_per_process']==2022, 'all3APIs full component gate')
        require(refs['component']['passed'] and refs['component']['complete'] and refs['component']['all_full_bytes_equal']
                and refs['component']['component_positive']['1'], 'positive clean radix component gate')
        require(prior['dedicated_prefill_only'] and prior['binary_sha256']==basebuild['binary_sha256'], 'same scoped native model correctness gate')
        order=[(1,0)]
        position=order.index((arm,repetition))
        for flag,index in order[:position]:
            previous_path=A/f'own-selector-clean-{flag}-r{index}-v1.json'
            previous=json.loads(previous_path.read_text())
            require(previous['passed'] and previous['complete'] and not previous['active'] and
                not previous['fresh_faults'] and previous['boot_unchanged'] and not previous['driver_dump_exists'],
                'prior independent clean sample closed')
            require(all(module.closed(c) for c in previous['commands']),'prior sample owner closure')
            for c in previous['commands']:
                for old in c['owners']:
                    now=module.identity(old['pid'])
                    require(now is None or now['start_ticks']!=old['start_ticks'],'prior sample owner present')
        require(prior['passed'] and prior['complete'] and not prior['active'] and prior['result']['server_exit_code']==prior['result']['real_engine_exit']['exit_code']==0 and not prior['fresh_faults'] and prior['boot_unchanged'] and not prior['driver_dump_exists'] and prior['boot_id']==boot_id(),'actual NativeHead component gate')
        for record in (build,prior,host):
            require(all(module.closed(c) for c in record['commands']),'prior ownership incomplete')
            for c in record['commands']:
                for old in c.get('owners',[]):
                    now=module.identity(old['pid'])
                    require(now is None or now['start_ticks']!=old['start_ticks'],'prior identity present')
        for n,v in host['source_pins'].items():require(sha(P/n)==v['proposed_sha256'],'qualified client changed')
        require(sha(P/'tools/test_benchmark_evidence.py')==host['evidence_fixture_sha256'],'qualified fixture changed')
        require(all(sha(P/n)==h for n,h in build['source_pins'].items()),'native source changed')
        bb=Path(build['build_path']);mkl=Path('/opt/intel/oneapi/mkl/2026.1/lib')
        linkpaths=[bb/n for n in ['libstrata_prefill.a','libstrata_spec.a','libstrata_engine.a',
            'libstrata_kernels.a','libstrata_core.a','libstrata_sycl_runtime.a','libstrata_kernels_cpu.a',
            'ggml/src/libggml-cpu.a','ggml/src/libggml-base.a','CMakeFiles/strata.dir/src/program/generate.cpp.o']]
        linkpaths += [mkl/n for n in ['libmkl_sycl_blas.so','libmkl_intel_ilp64.so','libmkl_sequential.so','libmkl_core.so']]
        linked_inputs={str(x):sha(x) for x in linkpaths}
        require(all(sha(Path(n))==h for n,h in linked_inputs.items()),'native linked input changed')
        exe=Path(build['binary']);require(sha(exe)==build['binary_sha256'],'native binary changed')
        runtime_path=A/'own-qsa-select-model-a1-library-hashes-v1.json';runtime=json.loads(runtime_path.read_text())
        require(all(sha(Path(v['path']))==v['sha256'] for v in runtime['files']),'qualified loaded runtime library files unchanged')
        require(all(sha(P/n)==h for n,h in off['recipes'].items()),'existing runner changed')
        require((DEVICE/'driver').resolve().name=='xe' and not (DEVICE/'devcoredump').exists(),'GPU preflight')
        for field in ('pack_metadata_sha256','MTP_metadata_sha256'):
            require(all(sha(Path(n))==h for n,h in off[field].items()),'model metadata changed')
        for n,v in off['model_identity'].items():
            st=Path(n).stat();require(st.st_size==v['bytes'] and st.st_mtime_ns==v['mtime_ns'],'model identity changed')
        root=B/'main-own-prefill-cpu-wait-model-root-v1';receipt=A/'own-prefill-cpu-wait-model-v1.json'
        require(not root.exists() and not receipt.exists(),'fresh prefill run root')
        root.mkdir(mode=0o700)
        env=dict(off['environment'],STRATA_SYCL_PARALLEL_IQ_DEQUANT='1',XDG_CACHE_HOME=str(root/'cache'),
                 STRATA_MEASUREMENT_ENGINE_START=str(root/'engine-start.json'),STRATA_MEASUREMENT_ENGINE_EXIT=str(root/'engine-exit.json'))
        env.update(STRATA_PREFILL_LAYER_MAJOR_R_GPU='32768',STRATA_PREFILL_RELEASE_HEADS='0',STRATA_PREFILL_NATIVE_DENSE_BANK='1',STRATA_SYCL_PROMPT_DIRECT='1',
            STRATA_SYCL_PREFILL_SCRATCH_ALIAS='1',STRATA_SYCL_QSA_INDEXER_DEVICE_BATCH='1',
            STRATA_GR_UNFUSED='0',STRATA_PREFILL_BF16X2='0',STRATA_PF_FUSED='0')
        retired_env_keys=('STRATA_CPU_GU_REUSE_OBSERVER','STRATA_DECODE_ROUTE_LEDGER','STRATA_DECODE_SERVICE_CPU')
        for key in retired_env_keys:env.pop(key,None)
        env.update(STRATA_PREFILL_CANONICAL_DENSE_BANK='1',STRATA_PREFILL_LAYER_MAJOR_PROFILE='0',
                   STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE='0',STRATA_SYCL_SELECT_RADIX4=str(arm),
                   STRATA_SYCL_SELECT_CYCLIC='0',STRATA_SYCL_PROMPT_K_CELL_MAJOR='0',STRATA_SYCL_PROMPT_NATIVE_KV_MIN='0')
        require(not any('TOKEN' in k or 'API_KEY' in k or k.startswith(('UNITRACE_','PTI_','UR_LOG_','ZE_ENABLE_','ZEL_','IGC_')) or k in ('UR_ENABLE_LAYERS','LD_PRELOAD','MKL_VERBOSE','STRATA_TRACE','STRATA_PREFILL_TIMING','STRATA_DECODE_TIMING') for k in env),'clean environment required')
        require(all(env.get(n,'0')=='0' for n in ('STRATA_PREFILL_BOUNDARY_AUDIT','STRATA_PREFILL_PHASE_AUDIT','STRATA_PREFILL_ROUTED_FP16_OBSERVE','STRATA_PREFILL_LAYER_MAJOR_PROFILE','STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE','STRATA_PREFILL_LAYER_MAJOR_SHAPE_CENSUS','STRATA_IQ_TIMING_PROGRESS')),'no numerical capture/observer in clean sample')
        env.update(STRATA_PREFILL_CPU_WAIT_PROFILE='1', STRATA_TRACE='1', UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION', UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr', UR_LOG_LOADER='level:warning;flush:warning;output:stderr', UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
        for k,v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',EnableDirectSubmission='0',
                        STRATA_SYCL_PREFILL_SAME_QUEUE='1',SYCL_CACHE_PERSISTENT='0',STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0',
                        STRATA_PREFILL_LAYER_MAJOR='1',STRATA_PREFILL_LAYER_MAJOR_R_GPU='32768',
                        STRATA_PREFILL_LAYER_MAJOR_R_DTYPE='fp32',STRATA_PREFILL_RING='8').items():
            require(env[k]==v,'runtime/route flag changed: '+k)
        template=WRAPPER.read_text()
        require(template.count("assert env['STRATA_SYCL_PROMPT_DIRECT'] == '0'")==1,'template Direct guard')
        template=template.replace("assert env['STRATA_SYCL_PROMPT_DIRECT'] == '0'","assert env['STRATA_SYCL_PROMPT_DIRECT'] == '1'")
        anchor="    pipe = os.stat('/proc/' + str(child.pid) + '/fd/0')"
        require(template.count(anchor)==1,'actual environment record seam')
        template=template.replace(anchor,"    entry['experiment_environment'] = {k:env[k] for k in ('STRATA_PREFILL_CANONICAL_DENSE_BANK','STRATA_PREFILL_LAYER_MAJOR_PROFILE','STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE','STRATA_PREFILL_NATIVE_DENSE_BANK','STRATA_SYCL_PROMPT_DIRECT','STRATA_PREFILL_LAYER_MAJOR_R_GPU','STRATA_PREFILL_LAYER_MAJOR_R_DTYPE','STRATA_SYCL_SELECT_RADIX4','STRATA_SYCL_SELECT_CYCLIC','STRATA_SYCL_PROMPT_K_CELL_MAJOR','STRATA_PREFILL_BOUNDARY_AUDIT','STRATA_PREFILL_PHASE_AUDIT','STRATA_PREFILL_CPU_WAIT_PROFILE','STRATA_TRACE','UR_ENABLE_LAYERS')}\n"+anchor)
        template=re.sub(r"^ENGINE = .*?$",'ENGINE = '+repr(str(exe)),template,flags=re.M)
        template=re.sub(r"^ENGINE_SHA256 = .*?$",'ENGINE_SHA256 = '+repr(sha(exe)),template,flags=re.M)
        wrapper=root/'engine-wrapper.py';wrapper.write_text(template);wrapper.chmod(0o700)
        config=dict(off['config'],exe=str(wrapper),log=str(root/'owned/engine.stderr'),
                    env={'STRATA_SYCL_PROMPT_DIRECT':'1','STRATA_SYCL_PARALLEL_IQ_DEQUANT':'1','STRATA_PREFILL_CANONICAL_DENSE_BANK':'1','STRATA_PREFILL_LAYER_MAJOR_PROFILE':'0','STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE':'0','STRATA_SYCL_SELECT_RADIX4':str(arm),'STRATA_SYCL_SELECT_CYCLIC':'0'})
        config['args']=list(off['engine_args'])
        config['args'][config['args'].index('--prefill')+1]='8192'
        require(sum(a!=b for a,b in zip(config['args'],off['engine_args']))==1,'only chunk argument changes before explicit graph guard')
        require('--no-token-graph' not in config['args'],'initial graph flag')
        config['args'].append('--no-token-graph')
        options=['--prefill-only','--prefill-tokens','65536','--workload-id','prefill_v1_20261015']
        (root/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        (root/'plan.json').write_text(json.dumps(dict(port=18230+position,url=f'http://127.0.0.1:{18230+position}',runs=1,
            benchmark_options=options,bench_timeout_seconds=1200),indent=2)+'\n')
        clients={n:sha(P/n) for n in off['benchmark_source']}
        record=dict(schema='owned-own-prefill-cpu-wait-diagnostic-v1',dedicated_prefill_only=True,selection_arm=arm,repetition=repetition,measurement_order=position,wrapper_sha256=sha(wrapper),wrapper_template_sha256=sha(WRAPPER),active=True,passed=False,complete=False,GPU_executed=False,
                    commands=[],environment=env,config=config,benchmark_options=options,runs=1,
                    recipe_sha256=sha(Path(__file__)),references={k:sha(v) for k,v in paths.items()},
                    binary_sha256=sha(exe),source_pins=build['source_pins'],linked_inputs=linked_inputs,
                    runtime_stack=dict(previous_loaded_files_receipt=runtime_path.name,receipt_sha256=sha(runtime_path),files_verified_unchanged_before_exec=True,new_process_maps_observed=False),
                    benchmark_source=clients,boot_id=boot_id(),engine_args=config['args'],
                    model_identity=off['model_identity'],pack_metadata_sha256=off['pack_metadata_sha256'],MTP_metadata_sha256=off['MTP_metadata_sha256'],
                    limits=dict(wall_seconds=1500,RSS_bytes=88<<30,text_bytes=64<<20,job_bytes=1<<30,file_bytes=64<<20),
                    performance_adopted=False,quality_qualified=False,physical_262144_qualified=False,restore_qualified=False,
                    scope='Separate official bench.sh prefill-only actual78453 input, batched78452, C8192/FP32/GPU32768/bank1/Direct1/radix1. TRACE+parameter validation+explicit wait CPU clocks enabled; NOT clean speed evidence. One owned native; no quality/physicalcontext/decode/restore qualification.',performance_qualified=False,
                    retention_consumer='Individual actual input/read counts, engine PP/TTFT, request-body hashes, placement and normal retirement/fault evidence; then retire successful duplicate logs.')
        def persist():(root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=module.Owner(root/'owned',persist);record['commands']=owner.commands;persist()
        host_env=dict(PATH='/usr/bin:/bin',HOME=str(Path.home()),LANG='C.UTF-8',LC_ALL='C.UTF-8',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
        def utility(label,argv):
            c,out,err=owner.run(label,argv,host_env,P,wall=60,cpu=60,rss_cap=2<<30,text_cap=8<<20,total_cap=1<<30,file_cap=8<<20)
            require(module.closed(c) and c['exit_code']==0,label+': '+err.read_text()[-3000:]);return out.read_text()
        cursor=None
        try:
            cursor=re.search(r'-- cursor: (\S+)',utility('journal-before',['journalctl','-k','-b','-n','0','--show-cursor','--no-pager']))[1]
            record['GPU_launch_attempted']=True
            c,out,err=owner.run('prefill-only',[PY,str(RUNNER),str(root),'extended-bench'],env,P,
                                wall=1500,cpu=18000,rss_cap=88<<30,text_cap=64<<20,total_cap=1<<30,file_cap=64<<20)
            record['GPU_executed']=(root/'engine-start.json').exists()
            require(module.closed(c) and c['exit_code']==0 and (root/'result.json').exists(),
                'runner/native launch or completion failed: '+err.read_text()[-5000:])
            result=json.loads((root/'result.json').read_text());record['result']=result
            require(module.closed(c) and c['exit_code']==0 and result['passed'] and
                    result['server_exit_code']==result['real_engine_exit']['exit_code']==0,'native/server completion')
            require(result['real_engine_exit']['binary_sha256']==sha(exe),'actual native differs')
            effective=result['real_engine_exit']['experiment_environment']
            require(all(v==env[k] for k,v in effective.items()) and effective['STRATA_SYCL_SELECT_RADIX4']==str(arm)
                    and effective['STRATA_PREFILL_CANONICAL_DENSE_BANK']==effective['STRATA_PREFILL_CPU_WAIT_PROFILE']==effective['STRATA_TRACE']=='1'
                    and effective['UR_ENABLE_LAYERS']=='UR_LAYER_PARAMETER_VALIDATION', 'actual native diagnostic experiment environment')
            record['actual_experiment_environment']=effective
            suites=list((root/'stock-results').glob('*/results.json'));require(len(suites)==1,'one suite')
            suite=json.loads(suites[0].read_text());cohort=suite['results']['prefill_only']
            require(suite['complete'] and not suite['results']['failures'] and cohort['complete'] and
                    len(cohort['requests'])==cohort['eligible_runs']==1,'prefill cohort incomplete')
            require(suite['params']['think'] and suite['params']['mode']=='prefill-only','thinking/scope')
            require(all(r['ok'] and r['eligible'] and r['actual_prompt_tokens']==78453 and
                        r['actual_read_tokens']==78453 and r['actual_reused_tokens']==0 and r['generated_tokens']==24 and r['thinking_requested'] for r in cohort['requests']),'actual input gate')
            record.update(benchmark_result=suite,benchmark_result_sha256=sha(suites[0]))
            check=json.loads(utility('verify-request-journal',['/usr/bin/python3','-B','-c',
                'import sys,json;sys.path.insert(0,"tools");from strata_bench import verify_request_journal;print(json.dumps(verify_request_journal(sys.argv[1])))',
                str(suites[0].parent)]))
            require(check['valid'] and check['complete'] and not check['pending'],'actual request journal incomplete')
            record['request_journal_verification']=check
            record['request_journals']=[dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=sha(f)) for f in (root/'stock-results').glob('*/requests.jsonl')]
            require(len(record['request_journals'])==1,'one actual request journal')
            require(all(sha(P/n)==h for n,h in clients.items()) and all(sha(P/n)==h for n,h in build['source_pins'].items()),'source changed during run')
            native=(root/'owned/engine.stderr').read_text()
            require(not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:|retirement failed|stopping with owners live',native),'native error')
            require('strata prefill heads: target detached' not in native and 'strata prefill heads: draft detached' not in native,'excluded head combination absent')
            record['native_dense_lifecycle_scope']='Separate TRACE/validation/CPU-clock diagnostic; throughput is not clean-qualified.'
            placement=[line for line in native.splitlines() if line.startswith(('strata prefill: layer-major candidate,','strata prefill: FP16 MoE compact'))]
            require(len([line for line in placement if 'chunk=8192 residual_dtype=fp32' in line and 'residual_gpu_rows=32768 ' in line and 'residual_gpu=1342177280 ' in line])==1,'actual C8192/R_GPU32768 admission, no backoff')
            record['placement_lines']=placement
            require(not any(line.startswith(('strata lm profile:','strata lm qsa profile:','strata native dense host profile:')) for line in native.splitlines()), 'clean profile hooks disabled')
            lines=[line for line in native.splitlines() if line.startswith('strata prefill cpu wait: ')]
            require(len(lines)==1,'exactly one long-prefill scope, no warmup/decode scope')
            fields=dict(re.findall(r'(\w+)=([^ ]+)',lines[0]))
            require(fields['available']==fields['complete']=='1' and fields['clock_or_accounting_error']==fields['open_waits']==fields['wait_throws']=='0','complete valid wait accounting')
            cpu={k:float(v) for k,v in fields.items() if k.endswith('_s')}
            require(all(v>=0 for v in cpu.values()) and abs(cpu['main_cpu_s']-cpu['wait_cpu_s']-cpu['outside_wait_cpu_s'])<3e-8,'valid CPU union arithmetic')
            require(cpu['main_cpu_s']>0 and cpu['wall_s']>=cpu['wait_wall_s'] and int(fields['wait_count'])>=480,'actual long phase/wait coverage')
            record['CPU_wait_diagnostic']=dict(line=lines[0],fields=fields,seconds=cpu,wait_share_of_caller_CPU=cpu['wait_cpu_s']/cpu['main_cpu_s'],outside_share_of_caller_CPU=cpu['outside_wait_cpu_s']/cpu['main_cpu_s'],scope='Caller CPU inside explicit blocking API calls; no internal stack or pure-polling proof; outside includes CPU work/submission/implicit blocking. Observer run is not clean speed evidence.')
            record['native_log_sha256']=sha(root/'owned/engine.stderr')
            record.update(passed=True,complete=True,performance_qualified=False)
        except BaseException as error:record['error']=repr(error);raise
        finally:
            try:
                if cursor and owner.active is None:
                    entries=[json.loads(s) for s in utility('journal-final',['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if s.strip()]
                    record.update(fresh_faults=[v for v in entries if FAULT.search(str(v.get('MESSAGE','')))],boot_unchanged=boot_id()==record['boot_id'],driver_dump_exists=(DEVICE/'devcoredump').exists())
            finally:
                if (root/'result.json').exists():record['result']=json.loads((root/'result.json').read_text())
                record.update(active=owner.active is not None,GPU_executed=(root/'engine-start.json').exists(),finished_utc=module.utc())
                if record['active'] or not all(module.closed(c) for c in owner.commands) or 'fresh_faults' not in record or record.get('fresh_faults') or not record.get('boot_unchanged') or record.get('driver_dump_exists'):
                    record.update(passed=False,complete=False)
                record['performance_qualified']=False
                record['diagnostic_qualified']=bool(record['passed'] and record['complete'])
                persist()
                with receipt.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
        print(json.dumps(dict(passed=record['passed'],actual_inputs=[r['actual_prompt_tokens'] for r in cohort['requests']],CPU=record['CPU_wait_diagnostic'],performance_qualified=False)),flush=True)


if __name__=='__main__':
    main(1,0)
