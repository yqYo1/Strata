#!/usr/bin/env python3
"""Main-owned ordinary/host-USM copy comparison; all GPU work serialized."""
import fcntl
import json
import math
from pathlib import Path
import re
import statistics
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha, boot_id
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')
DEVICE = Path('/sys/bus/pci/devices/0000:05:00.0')
ORDER = ['ordinary', 'usm', 'usm', 'ordinary', 'ordinary', 'usm']


def main():
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = json.loads((A / 'own-prefill-host-profile-build-v3.json').read_text())
        model = json.loads((A / 'own-prefill-host-profile-model-v3.json').read_text())
        assessment = json.loads((A / 'own-prefill-host-profile-model-assessment-v3.json').read_text())
        require(old['passed'] and old['complete'] and not old['active'] and assessment['passed'] and
                assessment['original_receipt_sha256'] == sha(A / 'own-prefill-host-profile-model-v3.json'), 'closed qualified preceding work')
        for r in (old, model):
            for c in r['commands']:
                require(module.closed(c) and c['exit_code'] == 0, 'prior owned command closure')
                for identity in c['owners']:
                    now = module.identity(identity['pid'])
                    require(now is None or now['start_ticks'] != identity['start_ticks'], 'prior process present')
        require(all(sha(P / n) == h for n, h in old['source_pins'].items() if n != 'cmake/sycl_backend.cmake'), 'only explicit new target/tool changed')
        require(sha(old['compiler']) == old['compiler_sha256'], 'same compiler')
        root = B / 'main-own-host-transfer-component-root-v1'
        target = A / 'own-host-transfer-component-v1.json'
        require(not root.exists() and not target.exists(), 'fresh measurement')
        root.mkdir(mode=0o700)
        record = dict(passed=False, complete=False, active=True, GPU_executed=False, commands=[], cases={},
            boot_id=boot_id(), recipe_sha256=sha(Path(__file__)), compiler=old['compiler'], compiler_sha256=old['compiler_sha256'],
            source_base_commit='57477f26', source_pins=dict(old['source_pins']), build_environment=old['environment'],
            matrix=dict(bytes=[2 << 20, 8192*10240*4], order=ORDER, independent_processes_per_arm=3,
                measured_repetitions_per_process=3, warmup_repetitions_per_process=1),
            allocation_scope='Both arms align protected host regions to4096 with body offset128; ordinarynew[] versus HostUSM. Setup/pretouch outside copy timing. Verification uses a separate HostUSM buffer.',
            performance_adopted=False, model_performance_qualified=False, model_quality_qualified=False)
        record['source_pins'].update({n: sha(P / n) for n in ('cmake/sycl_backend.cmake', 'tools/sycl/host_transfer_probe.cpp')})
        def persist(): (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(root / 'owned', persist); record['commands'] = owner.commands; persist()
        hostenv = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
        def run(label, argv, env=hostenv, wall=60, cpu=120, rss=2 << 30, file_cap=8 << 20):
            c, out, err = owner.run(label, argv, env, P, wall=wall, cpu=cpu, rss_cap=rss,
                text_cap=64 << 20, total_cap=1 << 30, file_cap=file_cap)
            require(module.closed(c) and c['exit_code'] == 0, label + ': ' + err.read_text()[-4000:])
            return out.read_text(), err.read_text()
        cursor = None
        try:
            build = Path(old['build_path'])
            record['native_binary_sha256_before'] = sha(build / 'strata')
            run('build-transfer-probe', ['cmake', '--build', str(build), '--target', 'sycl_host_transfer_probe', '--parallel', '2'],
                old['environment'], wall=600, cpu=2400, rss=16 << 30, file_cap=64 << 20)
            require(sha(build / 'strata') == record['native_binary_sha256_before'], 'production native not rebuilt or changed')
            exe = build / 'sycl_host_transfer_probe'; record.update(binary=str(exe), binary_sha256=sha(exe))
            libraries = json.loads((A / 'own-qsa-select-model-a1-library-hashes-v1.json').read_text())
            require(all(sha(Path(v['path'])) == v['sha256'] for v in libraries['files']), 'same qualified runtime library files')
            record['runtime_library_files'] = libraries['files']
            baseline = json.loads((A / 'own-selector-clean-0-r0-v1.json').read_text())
            env = dict(baseline['environment'], XDG_CACHE_HOME=str(root / 'cache'))
            require(not any('TOKEN' in k or 'API_KEY' in k or k.startswith(('UNITRACE_', 'PTI_', 'UR_LOG_', 'ZE_ENABLE_', 'ZEL_', 'IGC_')) or k in ('UR_ENABLE_LAYERS','LD_PRELOAD','MKL_VERBOSE','STRATA_TRACE') for k in env), 'clean immutable baseline environment')
            for k, v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', NEOReadDebugKeys='1', EnableDirectSubmission='0',
                    STRATA_SYCL_PREFILL_SAME_QUEUE='1', SYCL_CACHE_PERSISTENT='0', STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0', ONEAPI_DEVICE_SELECTOR='level_zero:gpu').items():
                require(env[k] == v, 'safety setting changed: ' + k)
            record['environment'] = env
            require((DEVICE / 'driver').resolve().name == 'xe' and not (DEVICE / 'devcoredump').exists(), 'GPU preflight')
            out, _ = run('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])
            cursor = re.search(r'-- cursor: (\S+)', out)[1]
            def gpu_case(label, kind, size, repeats, tracing):
                current = dict(env)
                if tracing:
                    current.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                        UR_LOG_TRACING='level:info;flush:info;output:stderr',
                        UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                        UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                        UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
                record['GPU_executed'] = True
                argv = ['/usr/bin/taskset', '-c', '0', str(exe), '--host', kind, '--bytes', str(size), '--repeats', str(repeats)]
                out, err = run(label, argv, current, wall=180, cpu=1200, rss=4 << 30)
                require(not re.search(r'(?:UR|ZE)_RESULT_ERROR_|FAIL |retirement failed|Strata SYCL cleanup:', out + err), 'component failure')
                if tracing: require('UR_RESULT_SUCCESS' in err, 'firstcheck trace absent')
                require('DONE host='+kind+' repeats='+str(repeats)+' warmups=1 scope=transfer_component_not_model' in out, 'complete bounded component')
                lines = out.splitlines()
                setup = [dict(re.findall(r'(\w+)=([^ ]+)', l)) for l in lines if l.startswith('SETUP ')]
                require(len(setup)==1 and setup[0]['body_alignment_mod4096']=='128', 'same aligned host subrange')
                samples = [dict(re.findall(r'(\w+)=([^ ]+)', l)) for l in lines if l.startswith('COPY ')]
                checks = [dict(re.findall(r'(\w+)=([^ ]+)', l)) for l in lines if l.startswith('CHECK ')]
                require(len(samples) == len(checks) == 2*(repeats+1), 'all copy/check pairs')
                for sample, check in zip(samples, checks):
                    require(sample['host']==kind and int(sample['bytes'])==size and sample['direction']==check['direction'] and
                            sample['repetition']==check['repetition'] and check['full']==check['guards']==check['source']=='PASS', 'full data/guards/input oracle')
                    numbers={k:float(v) for k,v in sample.items() if k.endswith('_s') or k=='total_GBps'}
                    require(all(math.isfinite(v) and v>=0 for v in numbers.values()) and numbers['total_wall_s']>0, 'finite transfer clocks')
                    for axis in ('cpu','wall'):
                        require(abs(numbers['submit_'+axis+'_s']+numbers['wait_'+axis+'_s']-numbers['total_'+axis+'_s'])<1e-7, 'submission/wait partition')
                    require(abs(numbers['total_GBps']-size/1e9/numbers['total_wall_s'])<max(.001, numbers['total_GBps']*.001), 'byte throughput arithmetic')
                    sample.update(numeric=numbers)
                record['cases'][label]=dict(host=kind,bytes=size,trace=tracing,setup=setup[0],samples=samples,
                    stdout=out,stderr_sha256=sha(root/('owned/'+label+'.stderr')),environment=current,
                    correct=True,model_performance_qualified=False)
                persist()
            for kind in ('ordinary','usm'): gpu_case('smoke-'+kind,kind,4 << 20,1,True)
            for size in (2 << 20,8192*10240*4):
                for i,kind in enumerate(ORDER): gpu_case('clean-'+str(size)+'-'+str(i)+'-'+kind,kind,size,3,False)
            require(all(sha(P/n)==h for n,h in record['source_pins'].items()), 'source frozen throughout component')
            record['summary']={}
            for size in record['matrix']['bytes']:
                arms={}
                for kind in ('ordinary','usm'):
                    process_samples=[]
                    for v in record['cases'].values():
                        if v['trace'] or v['host']!=kind or v['bytes']!=size:continue
                        process_samples.append({direction:{metric:statistics.median(s['numeric'][metric] for s in v['samples'] if s['direction']==direction and s['warmup']=='0')
                            for metric in ('submit_cpu_s','submit_wall_s','wait_cpu_s','wait_wall_s','total_cpu_s','total_wall_s','total_GBps')} for direction in ('H2D','D2H')})
                    require(len(process_samples)==3,'three independent processes per size/arm')
                    arms[kind]=dict(process_medians=process_samples,median={d:{m:statistics.median(v[d][m] for v in process_samples) for m in process_samples[0][d]} for d in ('H2D','D2H')})
                record['summary'][str(size)]=arms
            record.update(passed=True,complete=True)
        except BaseException as error:
            record['error']=repr(error)
            raise
        finally:
            try:
                if cursor and owner.active is None:
                    out,_=run('journal-final',['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json'])
                    entries=[json.loads(line) for line in out.splitlines() if line.strip()]
                    record.update(fresh_faults=[v for v in entries if FAULT.search(str(v.get('MESSAGE','')))],
                        boot_unchanged=boot_id()==record['boot_id'],driver_dump_exists=(DEVICE/'devcoredump').exists())
            finally:
                record.update(active=owner.active is not None,finished_utc=module.utc())
                if record['active'] or not all(module.closed(c) for c in owner.commands) or (record['GPU_executed'] and
                    ('fresh_faults' not in record or record.get('fresh_faults') or not record.get('boot_unchanged') or record.get('driver_dump_exists'))):
                    record.update(passed=False,complete=False)
                persist()
                with target.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
        print(json.dumps(dict(passed=record['passed'],cases=len(record['cases']),summary=record['summary'])),flush=True)


if __name__=='__main__':main()
