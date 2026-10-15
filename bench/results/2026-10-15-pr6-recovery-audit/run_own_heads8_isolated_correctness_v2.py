#!/usr/bin/env python3
"""Owned isolated attention smoke/full gates; leaves the native NLL cohort intact."""
import fcntl
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha, boot_id
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')


def main():
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build = json.loads((A / 'own-heads8-isolated-build-v1.json').read_text())
        require(build['passed'] and build['complete'] and not build['active'] and not build['GPU_executed'], 'isolated build gate')
        for c in build['commands']:
            require(module.closed(c) and c['exit_code'] == 0, 'build command closed')
            for old in c['owners']:
                now = module.identity(old['pid'])
                require(now is None or now['start_ticks'] != old['start_ticks'], 'build owner alive')
        failed = json.loads((A/'own-heads8-isolated-correctness-v1.json').read_text())
        require(not failed['active'] and not failed['fresh_faults'] and failed['boot_unchanged'] and not failed['driver_dump_exists'], 'first configuration failure closed healthy')
        for c in failed['commands']:
            require(module.closed(c), 'failed command fully retired')
            for old in c['owners']:
                now = module.identity(old['pid'])
                require(now is None or now['start_ticks'] != old['start_ticks'], 'old failed owner alive')
        exe = Path(build['binary'])
        require(sha(exe) == build['binary_sha256'], 'isolated fixture pin')
        require(all(sha(P/n) == h for n,h in build['canonical_source_pins'].items()) and sha(Path(build['canonical_native'])) == build['canonical_native_sha256'], 'NLL cohort source and native binary intact')
        libraries = json.loads((A / 'own-qsa-select-model-a1-library-hashes-v1.json').read_text())
        require(all(sha(Path(v['path'])) == v['sha256'] for v in libraries['files']), 'qualified runtime files')
        root = B / 'main-own-heads8-isolated-correctness-root-v2'
        target = A / 'own-heads8-isolated-correctness-v2.json'
        require(not root.exists() and not target.exists(), 'fresh correctness job')
        root.mkdir(mode=0o700)
        oldenv = json.loads((A / 'own-host-transfer-component-v1.json').read_text())['environment']
        names = ['PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH','CPLUS_INCLUDE_PATH','ONEAPI_DEVICE_SELECTOR',
                 'UR_ADAPTERS_FORCE_LOAD','SYCL_CACHE_PERSISTENT','NEOReadDebugKeys','EnableDirectSubmission',
                 'UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD','STRATA_SYCL_VERIFY_NATIVE_CAPTURE','STRATA_SYCL_PREFILL_SAME_QUEUE']
        env = {k:oldenv[k] for k in names}
        env.update(XDG_CACHE_HOME=str(root/'cache'),STRATA_SYCL_PROMPT_DIRECT='1',STRATA_SYCL_PROMPT_K_CELL_MAJOR='0',
                   STRATA_SYCL_PROMPT_NATIVE_KV_MIN='0',UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                   UR_LOG_LOADER='level:warning;flush:warning;output:stderr',UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
        record = dict(passed=False,complete=False,active=True,GPU_executed=False,commands=[],cases={},boot_id=boot_id(),
                      prior_configuration_failure_sha256=sha(A/'own-heads8-isolated-correctness-v1.json'),
                      layer_configuration_source='https://oneapi-src.github.io/unified-runtime/core/INTRO.html#ur-enable-layers; known passing own-host-transfer-component-v1 smoke environment',
                      recipe_sha256=sha(Path(__file__)),build_receipt_sha256=sha(A/'own-heads8-isolated-build-v1.json'),
                      binary_sha256=sha(exe),canonical_source_pins=build['canonical_source_pins'],canonical_native_sha256=build['canonical_native_sha256'],
                      runtime_files=libraries['files'],environment=env,raw_tensor_budget_bytes=16 << 20,
                      scope='Own opt-in Mode1 direct attention: two smoke and96full cases per arm; independent full FP64 oracle, guards/immutability/missing tables/invalid widths and exact inter-arm dump comparison. Not PP/performance/model-quality/full physicalcontext qualification.',
                      performance_qualified=False,native_adopted=False)
        def persist():
            (root/'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner = module.Owner(root/'owned',persist)
        record['commands'] = owner.commands
        persist()
        hostenv = dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
        def utility(label,argv):
            c,out,err = owner.run(label,argv,hostenv,P,wall=60,cpu=120,rss_cap=2 << 30,text_cap=8 << 20,total_cap=1 << 30,file_cap=8 << 20)
            require(module.closed(c) and c['exit_code'] == 0,label+': '+err.read_text()[-2000:])
            return out.read_text()
        cursor = None
        try:
            require((DEVICE/'driver').resolve().name == 'xe' and not (DEVICE/'devcoredump').exists(), 'GPU preflight')
            cursor = re.search(r'-- cursor: (\S+)',utility('journal-before',['journalctl','-k','-b','-n','0','--show-cursor','--no-pager']))[1]
            for phase,count,rawbytes in [('smoke',2,49408),('full',96,4730880)]:
                dumps = []
                for arm in (0,1):
                    label = phase+'-'+str(arm)
                    dump = root/(label+'.bin')
                    current = dict(env,STRATA_SYCL_PROMPT_HEADS8=str(arm),
                                   UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION' if phase == 'smoke' else 'UR_LAYER_PARAMETER_VALIDATION',
                                   UR_LOG_TRACING='level:info;flush:info;output:stderr')
                    args = ['taskset','-c','0',str(exe),*(['--smoke'] if phase == 'smoke' else []),'--dump',str(dump)]
                    record['GPU_executed'] = True
                    c,out,err = owner.run(label,args,current,P,wall=180 if phase == 'smoke' else 900,cpu=9000,rss_cap=8 << 30,
                                          text_cap=64 << 20,total_cap=1 << 30,file_cap=64 << 20)
                    require(module.closed(c) and c['exit_code'] == 0,label+': '+err.read_text()[-6000:])
                    stdout,stderr = out.read_text(),err.read_text()
                    lines = [line for line in stdout.splitlines() if line.startswith('case nq=')]
                    require(len(lines) == count and all('oracle=FP64_full' in line and 'full_finite=PASS prefix_suffix_guards=PASS inputs=PASS allmissing=PASS restored_table_width0=PASS width-1=PASS width2052=PASS' in line for line in lines), 'exact full oracle/guard case coverage')
                    require(not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:',stderr),'native/runtime errors')
                    require(dump.stat().st_size == rawbytes,'exact full output count')
                    record['cases'][label] = dict(environment=current,argv=args,stdout=stdout,stderr_sha256=sha(err),dump_sha256=sha(dump),dump_bytes=rawbytes,
                                                   full_FP64_oracle_cases=count,normal_closed=True,trace=phase == 'smoke')
                    dumps.append(dump)
                    persist()
                require(dumps[0].read_bytes() == dumps[1].read_bytes(),phase+': full outputs must match across arms')
                record[phase+'_full_dump_byteexact'] = True
            require(all(sha(P/n) == h for n,h in build['canonical_source_pins'].items()) and sha(Path(build['canonical_native'])) == build['canonical_native_sha256'], 'NLL cohort still unchanged')
            require(all(sha(Path(n)) == h for n,h in build['library_pins'].items()),'canonical archive pins still unchanged')
            record.update(passed=True,complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if cursor and owner.active is None:
                    rows = [json.loads(s) for s in utility('journal-final',['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if s.strip()]
                    record.update(fresh_faults=[v for v in rows if FAULT.search(str(v.get('MESSAGE','')))],boot_unchanged=boot_id() == record['boot_id'],driver_dump_exists=(DEVICE/'devcoredump').exists())
            finally:
                record.update(active=owner.active is not None,finished_utc=module.utc())
                if record['active'] or not all(module.closed(c) for c in owner.commands) or 'fresh_faults' not in record or record.get('fresh_faults') or not record.get('boot_unchanged') or record.get('driver_dump_exists'):
                    record.update(passed=False,complete=False)
                persist()
                with target.open('x') as f:
                    json.dump(record,f,indent=2)
                    f.write('\n')
        print(json.dumps(dict(passed=record['passed'],cases=list(record['cases']),smoke_byteexact=record['smoke_full_dump_byteexact'],full_byteexact=record['full_full_dump_byteexact'],native_unchanged=True)),flush=True)


if __name__ == '__main__':
    main()
