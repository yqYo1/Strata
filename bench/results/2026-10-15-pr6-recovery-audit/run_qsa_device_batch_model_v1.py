#!/usr/bin/env python3
"""Owned official bench.sh diagnostic QSA device-batch model boundary gate; no long decode."""
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


def main(arm):
    require(arm in (0,1), "immutable alias arm")
    module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        paths={k:A/n for k,n in dict(build='qsa-device-batch-build-v1.json',
               off='selective-iq-clean-bench-full-0-v1.json',closed='qsa-device-batch-matrix-v1.json',
               host='bench-prefill-only-host-v2.json').items()}
        refs={k:json.loads(p.read_text()) for k,p in paths.items()}
        build,off,prior,host=[refs[k] for k in ('build','off','closed','host')]
        require(all(x['passed'] and x['complete'] and not x['active'] for x in (build,off,host)), 'actual build/host gates')
        require(host['total_host_tests']==31,'host coverage')
        require(prior['passed'] and prior['complete'] and not prior['active'] and len(prior['samples'])==7 and all(s['passed'] for s in prior['samples']) and not prior['fresh_faults'] and prior['boot_unchanged'] and not prior['driver_dump_exists'] and prior['boot_id']==boot_id(),'actual QSA component/fault gate')
        for record in (build,prior,host):
            require(all(module.closed(c) for c in record['commands']),'prior ownership incomplete')
            for c in record['commands']:
                for old in c.get('owners',[]):
                    now=module.identity(old['pid'])
                    require(now is None or now['start_ticks']!=old['start_ticks'],'prior identity present')
        for n,v in host['source_pins'].items():require(sha(P/n)==v['proposed_sha256'],'qualified client changed')
        require(sha(P/'tools/test_benchmark_evidence.py')==host['evidence_fixture_sha256'],'qualified fixture changed')
        require(all(sha(P/n)==h for n,h in build['source_pins'].items()),'native source changed')
        bb=Path(build['build_path'])
        mkl=Path('/opt/intel/oneapi/mkl/2026.1/lib')
        linkpaths=[bb/n for n in ['libstrata_prefill.a','libstrata_spec.a','libstrata_engine.a',
            'libstrata_kernels.a','libstrata_core.a','libstrata_sycl_runtime.a','libstrata_kernels_cpu.a',
            'ggml/src/libggml-cpu.a','ggml/src/libggml-base.a','CMakeFiles/strata.dir/src/program/generate.cpp.o']]
        linkpaths += [mkl/n for n in ['libmkl_sycl_blas.so','libmkl_intel_ilp64.so','libmkl_sequential.so','libmkl_core.so']]
        linked_inputs={str(x):sha(x) for x in linkpaths}
        require(all(sha(Path(n))==h for n,h in linked_inputs.items()),'native linked input changed')
        exe=Path(build['binary']);require(sha(exe)==build['binary_sha256'],'native binary changed')
        require(all(sha(P/n)==h for n,h in off['recipes'].items()),'existing runner changed')
        require((DEVICE/'driver').resolve().name=='xe' and not (DEVICE/'devcoredump').exists(),'GPU preflight')
        for field in ('pack_metadata_sha256','MTP_metadata_sha256'):
            require(all(sha(Path(n))==h for n,h in off[field].items()),'model metadata changed')
        for n,v in off['model_identity'].items():
            st=Path(n).stat();require(st.st_size==v['bytes'] and st.st_mtime_ns==v['mtime_ns'],'model identity changed')
        root=B/f'main-qsa-device-batch-model-{arm}-root-v1';receipt=A/f'qsa-device-batch-model-{arm}-v1.json'
        require(not root.exists() and not receipt.exists(),'fresh prefill run root')
        root.mkdir(mode=0o700)
        env=dict(off['environment'],STRATA_SYCL_PARALLEL_IQ_DEQUANT='1',XDG_CACHE_HOME=str(root/'cache'),
                 STRATA_MEASUREMENT_ENGINE_START=str(root/'engine-start.json'),STRATA_MEASUREMENT_ENGINE_EXIT=str(root/'engine-exit.json'))
        env.update(STRATA_PREFILL_LAYER_MAJOR_R_GPU='0',STRATA_PREFILL_RELEASE_HEADS='1',
            STRATA_SYCL_PREFILL_SCRATCH_ALIAS='1', STRATA_SYCL_QSA_INDEXER_DEVICE_BATCH=str(arm), STRATA_PREFILL_BOUNDARY_AUDIT='1',
            STRATA_PREFILL_BOUNDARY_ROWS='0,1,255,256,4095,4096,8191,8192,16383,16384,32767,32768,65535,65536,78450,78451',
            STRATA_PREFILL_BOUNDARY_PATH=str(root/'boundary.txt'), STRATA_PREFILL_BOUNDARY_ROWS_ABSOLUTE='0',
            STRATA_GR_UNFUSED='0', STRATA_PREFILL_BF16X2='0', STRATA_PF_FUSED='0',
            STRATA_TRACE='1', UR_ENABLE_LAYERS='UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
            UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
            UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
        require(not any('TOKEN' in k or 'API_KEY' in k or k.startswith(('UNITRACE_','PTI_')) or k in ('LD_PRELOAD','MKL_VERBOSE','STRATA_PREFILL_TIMING','STRATA_DECODE_TIMING') for k in env),'bounded diagnostic environment')
        for k,v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',EnableDirectSubmission='0',
                        STRATA_SYCL_PREFILL_SAME_QUEUE='1',SYCL_CACHE_PERSISTENT='0',STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0',
                        STRATA_PREFILL_LAYER_MAJOR='1',STRATA_PREFILL_LAYER_MAJOR_R_GPU='0',
                        STRATA_PREFILL_LAYER_MAJOR_R_DTYPE='fp32',STRATA_PREFILL_RING='8').items():
            require(env[k]==v,'runtime/route flag changed: '+k)
        config=dict(off['config'],exe=str(WRAPPER),log=str(root/'owned/engine.stderr'),
                    env={'STRATA_SYCL_PROMPT_DIRECT':'0','STRATA_SYCL_PARALLEL_IQ_DEQUANT':'1'})
        config['args']=list(off['engine_args'])
        config['args'][config['args'].index('--prefill')+1]='8192'
        require(sum(a!=b for a,b in zip(config['args'],off['engine_args']))==1,'only chunk argument changes')
        options=['--prefill-only','--prefill-tokens','65536','--workload-id','prefill_v1_20261015']
        (root/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        (root/'plan.json').write_text(json.dumps(dict(port=18102+arm,url=f'http://127.0.0.1:{18102+arm}',runs=1,
            benchmark_options=options,bench_timeout_seconds=1200),indent=2)+'\n')
        clients={n:sha(P/n) for n in off['benchmark_source']}
        record=dict(schema='owned-QSA-device-batch-model-boundary-v1',qsa_batch_arm=arm,wrapper_sha256=sha(WRAPPER),active=True,passed=False,complete=False,GPU_executed=False,
                    commands=[],environment=env,config=config,benchmark_options=options,runs=1,
                    recipe_sha256=sha(Path(__file__)),references={k:sha(v) for k,v in paths.items()},
                    binary_sha256=sha(exe),source_pins=build['source_pins'],linked_inputs=linked_inputs,
                    benchmark_source=clients,boot_id=boot_id(),engine_args=config['args'],
                    model_identity=off['model_identity'],pack_metadata_sha256=off['pack_metadata_sha256'],MTP_metadata_sha256=off['MTP_metadata_sha256'],
                    limits=dict(wall_seconds=1500,RSS_bytes=88<<30,text_bytes=64<<20,job_bytes=512<<20,file_bytes=64<<20),
                    performance_adopted=False,quality_qualified=False,physical_262144_qualified=False,restore_qualified=False,
                    scope='Official bench.sh prefill-only C8192/R_GPU0 one fresh>=65536 request, thinking ON,24 output cap. Same-native immutable QSA-batch arm boundary diagnostic; not clean speed, task quality, RESTORE or physical max context',performance_qualified=False,
                    retention_consumer='Main exact QSA0-vs1 selected residual comparator consumes one31457280B raw capture per arm. Paired budget64MiB. Retire raw after complete byte/row comparison; keep full diagnostic only until extraction')
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
            suites=list((root/'stock-results').glob('*/results.json'));require(len(suites)==1,'one suite')
            suite=json.loads(suites[0].read_text());cohort=suite['results']['prefill_only']
            require(suite['complete'] and not suite['results']['failures'] and cohort['complete'] and
                    len(cohort['requests'])==cohort['eligible_runs']==1,'prefill cohort incomplete')
            require(suite['params']['think'] and suite['params']['mode']=='prefill-only','thinking/scope')
            require(all(r['ok'] and r['eligible'] and r['actual_prompt_tokens']>=65536 and
                        r['actual_read_tokens']>=65536 and r['actual_reused_tokens']==0 for r in cohort['requests']),'actual input gate')
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
            target_releases=re.findall(r'strata prefill heads: target detached bytes=(\d+)',native)
            draft_releases=re.findall(r'strata prefill heads: draft detached bytes=(\d+)',native)
            require(len(target_releases)>=1 and len(target_releases)==len(draft_releases) and all(int(x)==521472000 for x in target_releases) and all(int(x)==223227900 for x in draft_releases),'both real long scopes and draft subset retirement')
            record['head_lifecycle']=dict(target_release_bytes=list(map(int,target_releases)),draft_release_bytes=list(map(int,draft_releases)),scope_completed_requests=len(cohort['requests']),diagnostic_lines=[line for line in native.splitlines() if line.startswith(('strata prefill heads:','strata mtp: prompt expert phase=')) or 'recapture' in line])
            placement=[line for line in native.splitlines() if line.startswith(('strata prefill: layer-major candidate,','strata prefill: FP16 MoE compact'))]
            require(len([line for line in placement if 'chunk=8192 residual_dtype=fp32' in line and 'residual_gpu=0 ' in line])==1,'actual C8192/R_GPU0 admission, no backoff')
            record['placement_lines']=placement
            record['native_log_sha256']=sha(root/'owned/engine.stderr')
            audit=root/'boundary.txt';raw=Path(str(audit)+'.rows.f32')
            audit_text=audit.read_text();lines=audit_text.splitlines()
            require(lines[0]=='tokens=78452 pos0=0 width=10240 layers=48 chunk=8192 selected=16', 'actual full boundary shape')
            require(lines[-1]=='status=complete boundary_memcmp=equal' and
                'memcmp=different' not in audit_text and not re.search(r'nonfinite=[1-9]',audit_text), 'same-run boundary copies')
            rows=list(map(int,env['STRATA_PREFILL_BOUNDARY_ROWS'].split(',')))
            expected_order=[(l,row) for l in range(48) for row in rows]
            order=[tuple(map(int,m.groups())) for line in lines if (m:=re.search(r'phase=output_before_store layer=(\d+).*?relative=(\d+) ',line))]
            require(order==expected_order and
                len([line for line in lines if line.startswith('phase=stored ')])==47*16 and
                len([line for line in lines if line.startswith('phase=next_input ')])==47*16, 'complete row and phase coverage')
            require(raw.stat().st_size==48*16*10240*4, 'bounded complete raw capture')
            record['boundary']=dict(text_path=str(audit), text_sha256=sha(audit), text_bytes=audit.stat().st_size,
                raw_path=str(raw),raw_sha256=sha(raw),raw_bytes=raw.stat().st_size,
                shape=lines[0], rows=rows, output_order=order, same_run_copy_equal=True)
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
                persist()
                with receipt.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
        print(json.dumps(dict(passed=record['passed'],actual_inputs=[r['actual_prompt_tokens'] for r in cohort['requests']],head_cycles=len(record.get('head_lifecycle',{}).get('target_release_bytes',[])),prefill=cohort['prompt_tps'],performance_qualified=record['performance_qualified'])),flush=True)


if __name__=='__main__':
    require(len(sys.argv)==2, 'expected QSA arm0 or1')
    main(int(sys.argv[1]))
