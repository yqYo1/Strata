#!/usr/bin/env python3
"""Owned official bench.sh long-prefill baseline; no long decode cohort."""
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
WRAPPER=A/'selective-iq-clean-bench-v1/engine_wrapper.py'
FAULT=re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
sys.dont_write_bytecode=True
sys.path.insert(0,str(A/'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner,require,sha,boot_id


def main():
    module=load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        paths={k:A/n for k,n in dict(build='selective-iq-native-build-v1.json',
               off='selective-iq-clean-bench-full-0-v1.json',closed='selective-iq-clean-bench-full-1-v2.json',
               host='bench-prefill-only-host-v2.json').items()}
        refs={k:json.loads(p.read_text()) for k,p in paths.items()}
        build,off,prior,host=[refs[k] for k in ('build','off','closed','host')]
        require(all(x['passed'] and x['complete'] and not x['active'] for x in (build,off,host)), 'actual build/host gates')
        require(host['total_host_tests']==31,'host coverage')
        require(not prior['active'] and all(module.closed(c) for c in prior['commands']) and
                prior['native_exit']['exit_code']==prior['result']['server_exit_code']==0 and
                prior['result']['bench_exit_code']==-2,'old deliberate interruption not retired')
        require(not prior['fresh_faults'] and prior['boot_unchanged'] and not prior['driver_dump_exists'] and
                prior['boot_id']==boot_id(),'old GPU fault/boot gate')
        for record in (build,prior,host):
            require(all(module.closed(c) for c in record['commands']),'prior ownership incomplete')
            for c in record['commands']:
                for old in c.get('owners',[]):
                    now=module.identity(old['pid'])
                    require(now is None or now['start_ticks']!=old['start_ticks'],'prior identity present')
        for n,v in host['source_pins'].items():require(sha(P/n)==v['proposed_sha256'],'qualified client changed')
        require(sha(P/'tools/test_benchmark_evidence.py')==host['evidence_fixture_sha256'],'qualified fixture changed')
        require(all(sha(P/n)==h for n,h in build['source_pins'].items()),'native source changed')
        require(all(sha(Path(n))==h for n,h in build['linked_inputs'].items()),'native linked input changed')
        exe=Path(build['binary']);require(sha(exe)==build['binary_sha256'],'native binary changed')
        require(all(sha(P/n)==h for n,h in off['recipes'].items()),'existing runner changed')
        require((DEVICE/'driver').resolve().name=='xe' and not (DEVICE/'devcoredump').exists(),'GPU preflight')
        for field in ('pack_metadata_sha256','MTP_metadata_sha256'):
            require(all(sha(Path(n))==h for n,h in off[field].items()),'model metadata changed')
        for n,v in off['model_identity'].items():
            st=Path(n).stat();require(st.st_size==v['bytes'] and st.st_mtime_ns==v['mtime_ns'],'model identity changed')
        root=B/'main-prefill-only-clean-iq1-c4096-root-v1';receipt=A/'prefill-only-clean-iq1-c4096-v1.json'
        require(not root.exists() and not receipt.exists(),'fresh prefill run root')
        root.mkdir(mode=0o700)
        env=dict(off['environment'],STRATA_SYCL_PARALLEL_IQ_DEQUANT='1',XDG_CACHE_HOME=str(root/'cache'),
                 STRATA_MEASUREMENT_ENGINE_START=str(root/'engine-start.json'),STRATA_MEASUREMENT_ENGINE_EXIT=str(root/'engine-exit.json'))
        require(not any('TOKEN' in k or 'API_KEY' in k or k.startswith(('UNITRACE_','PTI_','UR_LOG_','ZE_ENABLE_','ZEL_')) or
                        k in ('UR_ENABLE_LAYERS','LD_PRELOAD','MKL_VERBOSE','STRATA_TRACE','STRATA_PREFILL_TIMING','STRATA_DECODE_TIMING') for k in env),
                'clean environment required')
        for k,v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',EnableDirectSubmission='0',
                        STRATA_SYCL_PREFILL_SAME_QUEUE='1',SYCL_CACHE_PERSISTENT='0',STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0',
                        STRATA_PREFILL_LAYER_MAJOR='1',STRATA_PREFILL_LAYER_MAJOR_R_GPU='4096',
                        STRATA_PREFILL_LAYER_MAJOR_R_DTYPE='fp32',STRATA_PREFILL_RING='8').items():
            require(env[k]==v,'runtime/route flag changed: '+k)
        config=dict(off['config'],exe=str(WRAPPER),log=str(root/'owned/engine.stderr'),
                    env={'STRATA_SYCL_PROMPT_DIRECT':'0','STRATA_SYCL_PARALLEL_IQ_DEQUANT':'1'})
        require(config['args']==off['engine_args'],'engine args changed')
        options=['--prefill-only','--prefill-tokens','65536','--workload-id','prefill_v1_20261015']
        (root/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        (root/'plan.json').write_text(json.dumps(dict(port=18098,url='http://127.0.0.1:18098',runs=3,
            benchmark_options=options,bench_timeout_seconds=5100),indent=2)+'\n')
        clients={n:sha(P/n) for n in off['benchmark_source']}
        record=dict(schema='owned-prefill-only-clean-v1',active=True,passed=False,complete=False,GPU_executed=False,
                    commands=[],environment=env,config=config,benchmark_options=options,runs=3,
                    recipe_sha256=sha(Path(__file__)),references={k:sha(v) for k,v in paths.items()},
                    binary_sha256=sha(exe),source_pins=build['source_pins'],linked_inputs=build['linked_inputs'],
                    benchmark_source=clients,boot_id=boot_id(),engine_args=off['engine_args'],
                    model_identity=off['model_identity'],pack_metadata_sha256=off['pack_metadata_sha256'],MTP_metadata_sha256=off['MTP_metadata_sha256'],
                    limits=dict(wall_seconds=5400,RSS_bytes=88<<30,text_bytes=64<<20,job_bytes=1<<30,file_bytes=64<<20),
                    performance_adopted=False,quality_qualified=False,physical_262144_qualified=False,restore_qualified=False,
                    scope='Official bench.sh prefill-only: one warmup plus three fresh measured requests; actual prompt/read >=65536; thinking on; 24 output cap. Clean timing, not independent-process variance or task quality.',
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
            record['GPU_executed']=True
            c,out,err=owner.run('prefill-only',[PY,str(RUNNER),str(root),'extended-bench'],env,P,
                                wall=5400,cpu=64800,rss_cap=88<<30,text_cap=64<<20,total_cap=1<<30,file_cap=64<<20)
            result=json.loads((root/'result.json').read_text());record['result']=result
            require(module.closed(c) and c['exit_code']==0 and result['passed'] and
                    result['server_exit_code']==result['real_engine_exit']['exit_code']==0,'native/server completion')
            require(result['real_engine_exit']['binary_sha256']==sha(exe),'actual native differs')
            suites=list((root/'stock-results').glob('*/results.json'));require(len(suites)==1,'one suite')
            suite=json.loads(suites[0].read_text());cohort=suite['results']['prefill_only']
            require(suite['complete'] and not suite['results']['failures'] and cohort['complete'] and
                    len(cohort['requests'])==cohort['eligible_runs']==3,'prefill cohort incomplete')
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
            record['native_log_sha256']=sha(root/'owned/engine.stderr')
            record.update(passed=True,complete=True)
        except BaseException as error:record['error']=repr(error);raise
        finally:
            try:
                if cursor and owner.active is None:
                    entries=[json.loads(s) for s in utility('journal-final',['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if s.strip()]
                    record.update(fresh_faults=[v for v in entries if FAULT.search(str(v.get('MESSAGE','')))],boot_unchanged=boot_id()==record['boot_id'],driver_dump_exists=(DEVICE/'devcoredump').exists())
            finally:
                if (root/'result.json').exists():record['result']=json.loads((root/'result.json').read_text())
                record.update(active=owner.active is not None,finished_utc=module.utc())
                if record['active'] or not all(module.closed(c) for c in owner.commands) or 'fresh_faults' not in record or record.get('fresh_faults') or not record.get('boot_unchanged') or record.get('driver_dump_exists'):
                    record.update(passed=False,complete=False)
                persist()
                with receipt.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
        print(json.dumps(dict(passed=record['passed'],prefill=cohort['prompt_tps'],actual_inputs=[r['actual_prompt_tokens'] for r in cohort['requests']])),flush=True)


if __name__=='__main__':main()
