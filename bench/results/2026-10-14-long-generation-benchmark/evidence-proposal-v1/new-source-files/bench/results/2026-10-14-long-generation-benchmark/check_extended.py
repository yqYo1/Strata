#!/usr/bin/env python3
"""Serialize one fresh-server screening/suite with existing native correctness gates."""
import fcntl
import gzip
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent
BUILD = B/'main-primary-routed-fp16-census-build-root-v1'
OWNER = P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
WRAPPER = P/'bench/results/2026-10-14-primary-model-baseline/engine_wrapper_v3.py'
PY = '/home/yayoi/.local/share/strata-sycl/venv/bin/python'
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    mode = sys.argv[1]
    assert mode in ['screen', 'suite']
    runs = 1 if mode == 'screen' else 3
    options = ['--long-generation-only'] if mode == 'screen' else ['--long', '--long-generation', '--think']
    sys.path.insert(0,str(P/'tools'))
    from long_generation_bench import WORKLOAD_VERSION
    O = B/('main-primary-long-generation-' + mode + '-root-v' + str(WORKLOAD_VERSION))
    build = json.loads((BUILD/'record.json').read_text())
    gate = json.loads((B/'main-primary-routed-census129-pair-root-v1/record.json').read_text())
    reference_archive=P/'bench/results/2026-10-14-primary-routed-fp16-census/receipts/main-primary-long-routed-reference-root-v3/record.json.gz'
    reference_raw=gzip.decompress(reference_archive.read_bytes())
    assert hashlib.sha256(reference_raw).hexdigest() == 'b3087c4d27d04f6a41de685c00fac7c5a271c05d3a525f8f479dde858b7d2d38'
    reference = json.loads(reference_raw)
    stock = json.loads(gzip.decompress((P/'bench/results/2026-10-14-primary-compact-moe-recovery/stock-evidence/compact1/record.json.gz').read_bytes()))
    for record in [build, gate, reference]:
        assert record['passed'] and record['complete'] and not record['active']
    for record in [gate, reference]:
        assert record['binary_sha256'] == build['binaries']['strata']
        assert not record['fresh_faults'] and not record['driver_dump_exists'] and record['boot_unchanged']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not O.exists()
        assert (DEVICE/'driver').resolve().name == 'xe' and not (DEVICE/'devcoredump').exists()
        for name, pin in build['source_pins'].items():
            assert sha(P/name) == pin
        engine = Path(build['build_path'])/'strata'
        assert sha(engine) == build['binaries']['strata']
        for group in ['pack_metadata_sha256', 'MTP_metadata_sha256']:
            for name,pin in stock[group].items():
                assert sha(Path(name)) == pin
        for name,pin in stock['model_identity'].items():
            info=Path(name).stat()
            assert info.st_size == pin['bytes'] and info.st_mtime_ns == pin['mtime_ns']
        O.mkdir()
        env = dict(reference['environment'])
        for name in ['STRATA_DUMP_FIRST_LOGITS', 'STRATA_MEASUREMENT_ENGINE_START', 'STRATA_MEASUREMENT_ENGINE_EXIT']:
            env.pop(name, None)
        assert not any(k in env for k in ['UR_ENABLE_LAYERS', 'ZE_ENABLE_VALIDATION_LAYER',
            'STRATA_TRACE', 'STRATA_PREFILL_TIMING', 'STRATA_DECODE_TIMING', 'LD_PRELOAD', 'STRATA_DBG_NAN'])
        env.update(XDG_CACHE_HOME=str(O/'cache'), STRATA_MEASUREMENT_ENGINE_START=str(O/'engine-start.json'),
                   STRATA_MEASUREMENT_ENGINE_EXIT=str(O/'engine-exit.json'))
        args = reference['engine_args'][:-4]  # remove --tokens-file PATH --max-new 1; server adds --serve
        assert reference['engine_args'][-4] == '--tokens-file' and reference['engine_args'][-2:] == ['--max-new','1']
        config = dict(exe=str(WRAPPER), args=args, cwd=str(P),
            tokenizer=str(Path(args[args.index('--pack')+1])/'tokenizer'), model_name='qwen3.8-flash-next-iq3_s-primary',
            aliases=['strata'], host='127.0.0.1', log=str(O/'owned/engine.stderr'),
            engine_silence_s=480, parallel=1, open_browser=False)
        (O/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        (O/'plan.json').write_text(json.dumps(dict(port=18096,url='http://127.0.0.1:18096',runs=runs,
            benchmark_options=options,bench_timeout_seconds=9000 if mode=='screen' else 25200),indent=2)+'\n')
        clients = ['bench.sh','tools/strata_bench.py','tools/long_generation_bench.py','tools/ab_engine.py',
                   'tools/agent_turns_bench.py','tools/serve_load.py']
        recipes = [Path(__file__),D/'run_server_job.py',WRAPPER]
        record = dict(active=True,passed=False,complete=False,commands=[],mode=mode,runs=runs,
            environment=env,engine_args=args,source_pins=build['source_pins'],binary_sha256=sha(engine),
            benchmark_source={n:sha(P/n) for n in clients},recipes={str(n.relative_to(P)):sha(n) for n in recipes},
            source_revision=subprocess.check_output(['git','-C',str(P),'rev-parse','HEAD'],text=True).strip(),
            build_receipt_sha256=sha(BUILD/'record.json'),numerical_gate_sha256=sha(B/'main-primary-routed-census129-pair-root-v1/record.json'),
            model_identity=stock['model_identity'],pack_metadata_sha256=stock['pack_metadata_sha256'],
            MTP_metadata_sha256=stock['MTP_metadata_sha256'],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            limits=dict(wall_seconds=9300 if mode=='screen' else 25800,raw_text_bytes=64<<20,RSS_bytes=88<<30,job_bytes=1<<30),
            performance_qualified=False,quality_qualified=False,physical_262144_qualified=False,restore_qualified=False,adopted=False,
            scope='Extended Niko community client; actual long output/MTP screening or full suite, warm server requests in one owned process. No inference change, profiler, reset or service change.')
        def persist():
            (O/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner = module.Owner(O/'owned',persist)
        record['commands'] = owner.commands
        persist()
        def utility(label,argv):
            entry,out,err=owner.run(label,argv,build['environment'],P,wall=30,text_cap=64<<20,total_cap=1<<30)
            assert module.closed(entry) and entry['exit_code']==0
            return out.read_text()
        cursor = None
        try:
            cursor=re.search(r'-- cursor: (\S+)',utility('journal-before',
                ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager'])).group(1)
            record['journal_before_cursor']=cursor
            entry,out,err=owner.run('extended-'+mode,[PY,str(D/'run_server_job.py'),str(O),'extended-bench'],env,P,
                wall=record['limits']['wall_seconds'],cpu=record['limits']['wall_seconds'],
                text_cap=64<<20,rss_cap=88<<30,total_cap=1<<30,file_cap=64<<20)
            journals = list((O/'stock-results').glob('*/requests.jsonl'))
            record['request_journals'] = [{"path": str(p.relative_to(O)), "sha256": sha(p),
                "bytes": p.stat().st_size,
                "state": json.loads((p.parent/'journal-state.json').read_text())
                    if (p.parent/'journal-state.json').exists() else {"complete": False}}
                for p in journals]
            paths=list((O/'stock-results').glob('*/results.json'))
            if len(paths)==1:
                record.update(benchmark_result=json.loads(paths[0].read_text()),benchmark_result_sha256=sha(paths[0]))
            result=json.loads((O/'result.json').read_text()) if (O/'result.json').exists() else {}
            record['result']=result
            persist()
            assert module.closed(entry) and entry['exit_code']==0,(entry.get('error'),err.read_text()[-1200:])
            assert result['passed'] and result['real_engine_exit']['exit_code']==0
            assert result['real_engine_exit']['binary_sha256']==record['binary_sha256']
            assert len(paths)==1
            suite=json.loads(paths[0].read_text())
            record.update(benchmark_result=suite,benchmark_result_sha256=sha(paths[0]),result=result)
            gen=suite['results']['long_generation']
            assert all(len(c['requests'])==runs and all(x['eligible'] for x in c['requests']) for c in gen['workloads'].values())
            if mode=='suite':
                assert gen['classification']['qualified']
                assert len(suite['results']['prompts']['32K']['requests'])==3
                assert len(suite['results']['agent']['sessions'])==3
                assert all(s['complete'] and s['completed_turns']==6 for s in suite['results']['agent']['session_samples'])
            for name,pin in build['source_pins'].items():
                assert sha(P/name)==pin
            assert sha(engine)==record['binary_sha256']
            assert all(sha(P/name)==pin for name,pin in record['benchmark_source'].items())
            record.update(passed=True,complete=True)
        except BaseException as error:
            record['error']=repr(error)
            raise
        finally:
            try:
                if cursor:
                    text=utility('journal-after',['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json'])
                    entries=[json.loads(x) for x in text.splitlines() if x.strip()]
                    record['fresh_faults']=[x for x in entries if FAULT.search(str(x.get('MESSAGE','')))]
                    record['driver_dump_exists']=(DEVICE/'devcoredump').exists()
                    record['boot_unchanged']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==record['boot_id']
                    if record['fresh_faults'] or record['driver_dump_exists'] or not record['boot_unchanged']:
                        record.update(passed=False,complete=False)
            finally:
                record['active']=False
                persist()
        print(json.dumps(dict(passed=record['passed'],complete=record['complete'],mode=mode,
            pair_qualified=gen['classification']['qualified'],fresh_faults=len(record['fresh_faults']))),flush=True)


if __name__=='__main__':
    main()
