#!/usr/bin/env python3
"""Main-owned clean bench.sh screen/full suite, one fresh server per explicit arm."""
import argparse,fcntl,gzip,hashlib,importlib.util,json,os,re,subprocess
from pathlib import Path
from check_gemm_orientation import P,B,D,OWNER,DEVICE,FAULT,sha
PY='/home/yayoi/.local/share/strata-sycl/venv/bin/python'
RUNNER=P/'bench/results/2026-10-14-long-generation-benchmark/run_server_job.py'
WRAPPER=P/'bench/results/2026-10-14-primary-model-baseline/engine_wrapper_v3.py'


def good(d):return d['passed'] and d['complete'] and not d['active']
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['standard','full']);ap.add_argument('direct',choices=['0','1']);a=ap.parse_args()
    bp=D/'cpu-gu-reuse-build-v3.json';gp=D/'cpu-gu-reuse-small-v1.json'
    build,gate=[json.loads(p.read_text()) for p in [bp,gp]];assert good(build) and good(gate)
    assert gate['binary_sha256']==build['binaries']['strata'] and not gate['fresh_faults'] and gate['boot_unchanged'] and not gate['driver_dump_exists']
    # These historical records supply only quiet runtime and model identities, never speed samples.
    quiet_path=P/'bench/results/2026-10-14-primary-routed-fp16-census/receipts/main-primary-long-routed-reference-root-v3/record.json.gz'
    quiet=json.loads(gzip.decompress(quiet_path.read_bytes()))
    stock_path=P/'bench/results/2026-10-14-primary-compact-moe-recovery/stock-evidence/compact1/record.json.gz'
    stock=json.loads(gzip.decompress(stock_path.read_bytes()))
    old_profile=json.loads((D/'model-device-profile-v2.json').read_text());assert good(old_profile)
    assert sha(OWNER)=='a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec=importlib.util.spec_from_file_location('owner',OWNER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert all(sha(P/n)==h for n,h in build['source_pins'].items())
        exe=Path(build['build_path'])/'strata';assert sha(exe)==build['binaries']['strata']
        assert (DEVICE/'driver').resolve().name=='xe' and not (DEVICE/'devcoredump').exists()
        for group in ['pack_metadata_sha256','MTP_metadata_sha256']:
            for n,h in stock[group].items():assert sha(Path(n))==h
        for n,pin in stock['model_identity'].items():
            info=Path(n).stat();assert info.st_size==pin['bytes'] and info.st_mtime_ns==pin['mtime_ns']
        # Check the matching OFF arm closed before starting ON, never restart a live arm.
        comparison_gate=None
        if a.direct=='1':
            comparison_path=D/('prompt-direct-bench-'+a.mode+'-0-v1.json')
            comparison=json.loads(comparison_path.read_text());assert good(comparison)
            assert comparison['binary_sha256']==sha(exe) and not comparison['fresh_faults'] and comparison['boot_unchanged'] and not comparison['driver_dump_exists']
            for c in comparison['commands']:
                assert m.closed(c)
                for i in c['owners']:
                    now=m.identity(i['pid']);assert now is None or now['start_ticks']!=i['start_ticks'] or now['state']=='Z'
            comparison_gate=sha(comparison_path)
        output=B/('main-primary-prompt-direct-bench-'+a.mode+'-'+a.direct+'-root-v1')
        assert not output.exists();output.mkdir(mode=0o700)
        env=dict(quiet['environment'])
        for k in ['STRATA_DUMP_FIRST_LOGITS','STRATA_MEASUREMENT_ENGINE_START','STRATA_MEASUREMENT_ENGINE_EXIT']:env.pop(k,None)
        assert not any('TOKEN' in k or 'API_KEY' in k or k.startswith(('UNITRACE_','PTI_','UR_LOG_','ZE_ENABLE_','ZEL_')) or k in ['UR_ENABLE_LAYERS','LD_PRELOAD','MKL_VERBOSE','STRATA_TRACE','STRATA_PREFILL_TIMING','STRATA_DECODE_TIMING','STRATA_DBG_NAN'] for k in env)
        env.update(STRATA_SYCL_PROMPT_DIRECT=a.direct,STRATA_PREFILL_LAYER_MAJOR='1',STRATA_PREFILL_LAYER_MAJOR_R_DTYPE='fp32',
            STRATA_PREFILL_LAYER_MAJOR_R_GPU='4096',STRATA_PREFILL_RELEASE_MTP_EXPERTS='1',STRATA_PREFILL_PHASE_AUDIT='0',
            STRATA_PREFILL_BOUNDARY_AUDIT='0',STRATA_SYCL_KV_COPY_GROUPS='0',STRATA_CPU_GU_REUSE_OBSERVER='0',
            STRATA_DECODE_ROUTE_LEDGER='0',STRATA_DECODE_SERVICE_CPU='0',STRATA_IQ_SINGLE_EXACT='0',
            XDG_CACHE_HOME=str(output/'cache'),STRATA_MEASUREMENT_ENGINE_START=str(output/'engine-start.json'),
            STRATA_MEASUREMENT_ENGINE_EXIT=str(output/'engine-exit.json'))
        for k,v in {'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD':'1','NEOReadDebugKeys':'1','EnableDirectSubmission':'0',
                'STRATA_SYCL_PREFILL_SAME_QUEUE':'1','SYCL_CACHE_PERSISTENT':'0','STRATA_SYCL_VERIFY_NATIVE_CAPTURE':'0'}.items():assert env[k]==v
        args=list(old_profile['argv'][old_profile['argv'].index(str(exe))+1:]);assert args.count('--serve')==1;args.remove('--serve')
        assert '--no-coupled-draft' in args and args[args.index('--max-context')+1]=='262144'
        options=['--long','--think']+(['--long-generation'] if a.mode=='full' else [])
        wall=25800 if a.mode=='full' else 14400
        config=dict(exe=str(WRAPPER),args=args,cwd=str(P),tokenizer=str(Path(args[args.index('--pack')+1])/'tokenizer'),
            model_name='qwen3.8-flash-next-iq3_s-primary',aliases=['strata'],host='127.0.0.1',log=str(output/'owned/engine.stderr'),
            engine_silence_s=480,parallel=1,open_browser=False,env={'STRATA_SYCL_PROMPT_DIRECT':a.direct})
        (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        (output/'plan.json').write_text(json.dumps(dict(port=18097,url='http://127.0.0.1:18097',runs=3,
            benchmark_options=options,bench_timeout_seconds=wall-300),indent=2)+'\n')
        clients=['bench.sh','tools/strata_bench.py','tools/long_generation_bench.py','tools/ab_engine.py','tools/agent_turns_bench.py','tools/serve_load.py','serve/server.py']
        record=dict(active=True,passed=False,complete=False,GPU_executed=True,commands=[],mode=a.mode,direct=a.direct,runs=3,
            environment=env,config=config,engine_args=args,benchmark_options=options,source_pins=build['source_pins'],
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),binary_sha256=sha(exe),
            build_receipt_sha256=sha(bp),safety_gate_sha256=sha(gp),comparison_gate_sha256=comparison_gate,
            recipe_sha256=sha(Path(__file__)),recipes={str(p.relative_to(P)):sha(p) for p in [RUNNER,WRAPPER]},
            benchmark_source={n:sha(P/n) for n in clients},model_identity=stock['model_identity'],pack_metadata_sha256=stock['pack_metadata_sha256'],MTP_metadata_sha256=stock['MTP_metadata_sha256'],
            quiet_reference_sha256=sha(quiet_path),identity_reference_sha256=sha(stock_path),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            limits=dict(wall_seconds=wall,cpu_seconds_per_process=wall*len(os.sched_getaffinity(0)),RSS_bytes=88<<30,text_bytes=64<<20,job_bytes=1<<30),
            adopted=False,performance_qualified=False,quality_qualified=False,physical_262144_qualified=False,restore_qualified=False,
            scope='Clean official bench.sh repeated-request screen. Three requests/sessions in one fresh owned server per arm; no independent-process variance claim. No short-input performance conclusion. Full mode includes thinking strong/weak long generation; standard screen does not qualify long-generation stability.',
            retention_consumer='Preserve individual timings and compact content/work fingerprints, verify request journal then retire successful full HTTP traces.',
            next='Compare matched OFF/ON individual32K and session samples; full thinking strong/weak and reverse-order process replication before speed adoption if effects overlap.')
        if a.direct=='1':
            assert args==comparison['engine_args'] and record['benchmark_source']==comparison['benchmark_source']
            allowed={'STRATA_SYCL_PROMPT_DIRECT','XDG_CACHE_HOME','STRATA_MEASUREMENT_ENGINE_START','STRATA_MEASUREMENT_ENGINE_EXIT'}
            assert {k for k in set(env)|set(comparison['environment']) if env.get(k)!=comparison['environment'].get(k)}<=allowed
        def persist():(output/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner=m.Owner(output/'owned',persist);record['commands']=owner.commands;persist()
        def utility(label,argv):
            e,o,s=owner.run(label,argv,build['environment'],P,wall=30,total_cap=1<<30)
            assert m.closed(e) and e['exit_code']==0,(label,e.get('error'));return o.read_text()
        cursor=None
        try:
            cursor=re.search(r'-- cursor: (\S+)',utility('journal-before',['journalctl','-k','-b','-n','0','--show-cursor','--no-pager']))[1]
            record['journal_before_cursor']=cursor
            print('Clean bench.sh '+a.mode+' Direct='+a.direct+' runs=3 thinking=on',flush=True)
            e,out,err=owner.run('bench-'+a.mode+'-'+a.direct,[PY,str(RUNNER),str(output),'extended-bench'],env,P,
                wall=wall,cpu=wall*len(os.sched_getaffinity(0)),rss_cap=88<<30,text_cap=64<<20,total_cap=1<<30,file_cap=64<<20)
            result=json.loads((output/'result.json').read_text()) if (output/'result.json').exists() else {}
            record['result']=result
            paths=list((output/'stock-results').glob('*/results.json'))
            journals=list((output/'stock-results').glob('*/requests.jsonl'))
            import sys;sys.path.insert(0,str(P/'tools'))
            from strata_bench import verify_request_journal
            record['request_journals']=[dict(path=str(p.relative_to(output)),sha256=sha(p),bytes=p.stat().st_size,
                verification=verify_request_journal(p.parent),state=json.loads((p.parent/'journal-state.json').read_text())) for p in journals]
            if len(paths)==1:record['benchmark_result']=json.loads(paths[0].read_text());record['benchmark_result_sha256']=sha(paths[0])
            persist()
            assert m.closed(e) and e['exit_code']==0,(e.get('error'),err.read_text()[-3000:])
            assert result['passed'] and result['real_engine_exit']['exit_code']==result['server_exit_code']==0
            assert result['real_engine_exit']['binary_sha256']==sha(exe)
            assert len(paths)==len(journals)==1 and record['request_journals'][0]['verification']['valid'] and record['request_journals'][0]['verification']['complete']
            suite=record['benchmark_result'];assert suite['complete'] and not suite['results']['failures']
            prompts=suite['results']['prompts']['32K']['requests'];assert len(prompts)==3 and all(x['ok'] and not x['reused_tokens'] for x in prompts)
            sessions=suite['results']['agent']['session_samples'];assert len(sessions)==3 and all(x['complete'] and x['completed_turns']==6 for x in sessions)
            record['nominal_32K_actual_input_tokens']=[x['prompt_tokens'] for x in prompts]
            record['nominal_32K_all_at_least_32768']=all(x['prompt_tokens']>=32768 for x in prompts)
            if a.mode=='full':
                gen=suite['results']['long_generation'];assert gen['classification']['qualified']
                assert all(len(c['requests'])==3 and all(x['eligible'] for x in c['requests']) for c in gen['workloads'].values())
            native_log=(output/'owned/engine.stderr').read_text()
            assert not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|what\(\):|Strata SYCL cleanup:|retirement failed|stopping with owners live|strata cpu_gu_reuse_v2|strata decode joined begin',native_log)
            record['native_log_sha256']=sha(output/'owned/engine.stderr')
            assert all(sha(P/n)==h for n,h in build['source_pins'].items()) and sha(exe)==record['binary_sha256']
            assert all(sha(P/n)==h for n,h in record['benchmark_source'].items())
            record.update(passed=True,complete=True)
        except BaseException as error:record['error']=repr(error);raise
        finally:
            try:
                if cursor:
                    entries=[json.loads(s) for s in utility('journal-final',['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if s.strip()]
                    record['fresh_faults']=[s for s in entries if FAULT.search(str(s.get('MESSAGE','')))];record['driver_dump_exists']=(DEVICE/'devcoredump').exists()
                    record['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==record['boot_id']
                    if record['fresh_faults'] or record['driver_dump_exists'] or not record['boot_unchanged']:record.update(passed=False,complete=False)
            finally:
                record['active']=False;persist();(D/('prompt-direct-bench-'+a.mode+'-'+a.direct+'-v1.json')).write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(dict(passed=record['passed'],fresh_faults=len(record['fresh_faults']))),flush=True)


if __name__=='__main__':main()
