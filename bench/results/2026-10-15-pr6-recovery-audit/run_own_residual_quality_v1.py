#!/usr/bin/env python3
"""One owned long-input loss arm; Main chooses the next arm after closure."""
import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha, boot_id
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
SOURCE_COMMIT = 'dfca0e14'


def main():
    require(len(sys.argv) == 2 and sys.argv[1] in ('fp32-a','fp32-b','fp16-matched','fp16-all-gpu'), 'one named arm')
    arm = sys.argv[1]
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build = json.loads((A / 'own-dense-prefill-nll-build-v3.json').read_text())
        smoke = json.loads((A / 'own-dense-prefill-nll-smoke-v4.json').read_text())
        inp = json.loads((A / 'own-residual-quality-input-v1.json').read_text())
        require(build['passed'] and smoke['passed'] and not smoke['active'] and inp['complete'], 'build/paired smoke/input gates')
        for c in smoke['commands']:
            require(module.closed(c) and c['exit_code'] == 0, 'prior command closed')
            for old in c['owners']:
                now = module.identity(old['pid'])
                require(now is None or now['start_ticks'] != old['start_ticks'], 'prior owner alive')
        commit = subprocess.check_output(['git','rev-parse',SOURCE_COMMIT + '^{commit}'], cwd=P, text=True).strip()
        for name, h in build['source_pins'].items():
            require(sha(P / name) == h, 'frozen compiled source')
            actual = subprocess.check_output(['git','show',commit + ':' + name], cwd=P)
            require(hashlib.sha256(actual).hexdigest() == h, 'compiled source belongs to immutable committed snapshot')
        exe = Path(build['binary'])
        require(sha(exe) == build['binary_sha256'], 'native binary')
        require(all(sha(Path(v['path'])) == v['sha256'] for v in smoke['runtime_files']), 'runtime library pins')
        token_file = Path(inp['token_file'])
        require(sha(token_file) == inp['file_sha256'], 'frozen official workload token IDs')
        for name, v in smoke['model_identity'].items():
            st = Path(name).stat()
            require(st.st_size == v['bytes'] and st.st_mtime_ns == v['mtime_ns'], 'model stat identity')
        prior = json.loads((A / 'own-prefill-host-profile-model-v3.json').read_text())
        for key in ('pack_metadata_sha256','MTP_metadata_sha256'):
            require(all(sha(Path(n)) == h for n,h in prior[key].items()), 'pack/MTP metadata pins')
        common = B / 'main-own-residual-quality-root-v1'
        root = common / arm
        target = A / f'own-residual-quality-{arm}-v1.json'
        require(not root.exists() and not target.exists(), 'fresh arm; never duplicate a live/failed run')
        prerequisites = {'fp32-a':[], 'fp32-b':['fp32-a'], 'fp16-matched':['fp32-a','fp32-b'], 'fp16-all-gpu':['fp32-a','fp32-b','fp16-matched']}[arm]
        for name in prerequisites:
            r = json.loads((A / f'own-residual-quality-{name}-v1.json').read_text())
            require(r['passed'] and r['complete'] and not r['active'], 'previous arm must close normally; no quality acceptance implied')
            require(r['recipe_sha256'] == sha(Path(__file__)), 'same controller throughout cohort')
            for c in r['commands']:
                require(module.closed(c) and c['exit_code'] == 0, 'preceding owned commands closed')
                for old in c['owners']:
                    now = module.identity(old['pid'])
                    require(now is None or now['start_ticks'] != old['start_ticks'], 'preceding owner present')
        root.mkdir(parents=True, mode=0o700)
        rows = inp['count'] - 1
        dtype = 'fp32' if arm.startswith('fp32') else 'fp16'
        gpu_rows = rows if arm == 'fp16-all-gpu' else 32768
        args = list(prior['engine_args'])
        args += ['--tokens-file',str(token_file),'--max-new','1','--stream-token']
        require(args[args.index('--max-context')+1] == '262144' and args[args.index('--prefill')+1] == '8192', 'actual previous long-route context/chunk')
        env = dict(smoke['environment'], XDG_CACHE_HOME=str(common / 'cache'),
                   STRATA_PREFILL_LAYER_MAJOR_R_DTYPE=dtype,
                   STRATA_PREFILL_LAYER_MAJOR_R_GPU=str(gpu_rows),
                   STRATA_PREFILL_NLL=str(root / 'nll.bin'))
        require(env['STRATA_PREFILL_CPU_WAIT_PROFILE'] == '0' and env['STRATA_PREFILL_RELEASE_HEADS'] == '0', 'no profiler/head release')
        for k, v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1',NEOReadDebugKeys='1',EnableDirectSubmission='0',
                         STRATA_SYCL_PREFILL_SAME_QUEUE='1',SYCL_CACHE_PERSISTENT='0',STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0',
                         ONEAPI_DEVICE_SELECTOR='level_zero:gpu').items():
            require(env[k] == v, 'immutable safety envelope')
        digest = lambda v: hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        model = dict(stat_identity=smoke['model_identity'], pack_metadata_sha256=prior['pack_metadata_sha256'],
                     MTP_metadata_sha256=prior['MTP_metadata_sha256'])
        settings = dict(dtype=dtype,residual_gpu_rows=gpu_rows,chunk=8192,context=262144,kv='int8; resident32768',
                        mtp_manifest_sha256=digest(prior['MTP_metadata_sha256']))
        recipe = dict(schema='strata-dense-prefill-nll-recipe-v2',arm=arm,
                      source=dict(commit=commit,snapshot_sha256=digest(build['source_pins']),binary_sha256=sha(exe)),
                      model=dict(manifest_sha256=digest(model)),
                      input={k:inp[k] for k in ('count','file_sha256','ids_le_u32_sha256','template_sha256')},
                      runtime=dict(hardware='Intel Arc B570 10GiB; Ryzen5600X;128GiB RAM',compiler=build['compiler'] + ';sha256=' + build['compiler_sha256'],
                                   runtime='Pinned own-qsa-select-model-a1-library-hashes-v1.json; verified identical files',
                                   driver='Qualified loaded IGC2.41.5+1788943183; driver1.17.39395+14; files pinned, not a new process-map proof'),
                      command=[str(exe),*args],environment=env,settings=settings,
                      model_identity_basis='Full pack/MTP metadata hashes and model byte size/mtime; not a new hash of all weight bytes.',
                      raw_consumer='Same-input two-FP32 controls versus FP16-matched, then separate allGPU placement comparison.',
                      quality_acceptance='UNSET; word-list workload does not establish task/general language quality',
                      raw_total_budget_bytes=16 << 20,performance_cohort='excluded')
        recipe_path = root / 'recipe.json'
        recipe_path.write_text(json.dumps(recipe,indent=2)+'\n')
        current = dict(env,STRATA_PREFILL_NLL_RECIPE_SHA256=sha(recipe_path))
        record = dict(passed=False,complete=False,active=True,GPU_executed=False,arm=arm,commands=[],
                      boot_id=boot_id(),recipe_sha256=sha(Path(__file__)),external_recipe_sha256=sha(recipe_path),
                      binary_sha256=sha(exe),source_pins=build['source_pins'],input_receipt_sha256=sha(A / 'own-residual-quality-input-v1.json'),
                      recipe=recipe,model_identity=model,runtime_files=smoke['runtime_files'],
                      scope='Long official word-list teacher-forced NLL diagnostic; no PP score/quality acceptance/fullcontext/RESTORE qualification.',
                      performance_qualified=False,quality_qualified=False,physical_262144_qualified=False,restore_qualified=False,
                      limits=dict(wall_seconds=2400,text_bytes=64 << 20,rss_bytes=88 << 30,raw_total_bytes=16 << 20))
        def persist():
            (root / 'record.json').write_text(json.dumps(record,indent=2)+'\n')
        owner = module.Owner(root / 'owned',persist)
        record['commands'] = owner.commands
        persist()
        hostenv = dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8',LC_ALL='C.UTF-8')
        def utility(label, argv):
            c,out,err = owner.run(label,argv,hostenv,P,wall=60,cpu=60,rss_cap=2 << 30,text_cap=8 << 20,total_cap=1 << 30,file_cap=8 << 20)
            require(module.closed(c) and c['exit_code'] == 0,label + ': ' + err.read_text()[-2000:])
            return out.read_text()
        cursor = None
        try:
            require((DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists(), 'GPU preflight')
            cursor = re.search(r'-- cursor: (\S+)',utility('journal-before',['journalctl','-k','-b','-n','0','--show-cursor','--no-pager']))[1]
            record['GPU_executed'] = True
            c,out,err = owner.run('native-'+arm,[str(exe),*args],current,P,wall=2400,cpu=28800,rss_cap=88 << 30,
                                  text_cap=64 << 20,total_cap=1 << 30,file_cap=64 << 20)
            require(module.closed(c) and c['exit_code'] == 0,'native: '+err.read_text()[-6000:])
            text = err.read_text()
            require(not re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|retirement failed|Strata SYCL cleanup:',text),'native errors')
            chunks = (rows + 8191) // 8192
            require(list(map(int,re.findall(r'layer-major layer (\d+) of 48: completed '+str(chunks)+r' chunks',text))) == list(range(48)), 'all48 layers/allchunks')
            placements = re.findall(r'layer-major candidate, tokens=(\d+) chunk=(\d+) residual_dtype=(fp32|fp16) residual_element_bytes=(\d+) residual_total=(\d+) residual_gpu_rows=(\d+) residual_gpu=(\d+) residual_host=(\d+) residual_staging=(\d+)',text)
            require(len(placements) == 1,'one actual native placement')
            count,chunk,dt,element,total,rg,gbytes,hbytes,stage = placements[0]
            require((int(count),int(chunk),dt,int(rg)) == (rows,8192,dtype,gpu_rows),'actual placement equals request')
            require('strata dense prefill NLL: rows='+str(rows)+' ' in text, 'full dense diagnostic completed')
            record.update(native_stdout_sha256=sha(out),native_stderr_sha256=sha(err),all48_layers_completed=True,
                          placement=dict(rows=int(count),chunk=int(chunk),dtype=dt,element_bytes=int(element),total_bytes=int(total),
                                         GPU_rows=int(rg),GPU_bytes=int(gbytes),host_bytes=int(hbytes),staging_bytes=int(stage)))
            require(all(sha(P/n) == h for n,h in build['source_pins'].items()), 'source frozen throughout')
            events = [json.loads(s) for s in utility('journal-after',['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if s.strip()]
            faults = [v for v in events if FAULT.search(str(v.get('MESSAGE','')))]
            fault_receipt = dict(cursor=cursor,filter=FAULT.pattern,fresh_faults=faults,boot_unchanged=boot_id() == record['boot_id'],
                                 driver_dump_exists=(DEVICE / 'devcoredump').exists(),journal_events=len(events),normal_exit=True)
            fault_path = root / 'faults.json'
            fault_path.write_text(json.dumps(fault_receipt,indent=2)+'\n')
            require(not faults and fault_receipt['boot_unchanged'] and not fault_receipt['driver_dump_exists'],'normal GPU health')
            for old in c['owners']:
                now = module.identity(old['pid'])
                require(now is None or now['start_ticks'] != old['start_ticks'],'native owner absent')
            identity = next(v for v in c['owners'] if v['pid'] == c['pid'])
            deadline = datetime.datetime.fromisoformat(c['started_utc']) + datetime.timedelta(seconds=c['wall_seconds'])
            observed = dict(dtype=dtype,chunk=int(chunk),context=262144,residual_gpu_rows=int(rg),n_input=inp['count'],rows=rows,
                            head_release=0,residual_host_bytes=int(hbytes),residual_staging_bytes=int(stage),
                            context_basis='Exact executed native --max-context flag and immutable parser source; no independent physical262144 consume test.')
            closed = dict(schema='strata-dense-prefill-nll-closed-v2',recipe_sha256=sha(recipe_path),
                          process=dict(boot_id=record['boot_id'],deadline_utc=deadline.isoformat(),pid=c['pid'],start_ticks=identity['start_ticks'],exit_code=0,owned_absent=True),
                          faults=dict(fresh_boundaries=True,count=0,receipt_sha256=sha(fault_path)),observed=observed,
                          raw_sha256=sha(root / 'nll.bin'),metadata_sha256=sha(root / 'nll.bin.meta'))
            (root / 'closed.json').write_text(json.dumps(closed,indent=2)+'\n')
            spec = importlib.util.spec_from_file_location('dense_comparator', P / 'tools/compare_dense_prefill_nll.py')
            comparator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(comparator)
            ids,file_sha,ids_sha = comparator.token_input(token_file)
            checked = comparator.load_run(root,ids,file_sha,ids_sha)
            record.update(passed=True,complete=True,loss_summary=comparator.summarize(checked['losses']),callback_ms=checked['callback_ms'],
                          closure_sha256=sha(root / 'closed.json'),raw_sha256=closed['raw_sha256'],metadata_sha256=closed['metadata_sha256'])
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if cursor and owner.active is None:
                    events = [json.loads(s) for s in utility('journal-final',['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if s.strip()]
                    record.update(fresh_faults=[v for v in events if FAULT.search(str(v.get('MESSAGE','')))],
                                  boot_unchanged=boot_id() == record['boot_id'],driver_dump_exists=(DEVICE / 'devcoredump').exists())
            finally:
                record.update(active=owner.active is not None,finished_utc=module.utc())
                if record['active'] or not all(module.closed(c) for c in owner.commands) or 'fresh_faults' not in record or record.get('fresh_faults') or not record.get('boot_unchanged') or record.get('driver_dump_exists'):
                    record.update(passed=False,complete=False)
                persist()
                with target.open('x') as f:
                    json.dump(record,f,indent=2)
                    f.write('\n')
        print(json.dumps(dict(arm=arm,passed=record['passed'],loss_summary=record['loss_summary'],callback_ms=record['callback_ms'])),flush=True)


if __name__ == '__main__':
    main()
