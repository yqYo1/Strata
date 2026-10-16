#!/usr/bin/env python3
"""Main-owned host/accounting gates, reviewed patch, SYCL build and small GPU gates."""
import fcntl
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
S = A / 'own-prefill-host-profile-v2'
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha, boot_id
FAULT = re.compile(r'(?i)(\bxe\b.*(fault|hang|reset|error|wedg|timeout|stuck)|GPU.*(fault|hang|wedg|reset)|GuC.*(fault|hang|reset|error))')


def main():
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        comparison = json.loads((A / 'own-selector-clean-comparison-v1.json').read_text())
        require(comparison['passed'] and comparison['complete'] and not comparison['active'], 'six process cohort must be closed')
        old = json.loads((A / 'own-prefill-cpu-wait-build-v1.json').read_text())
        require(old['passed'] and old['complete'] and not old['active'], 'preceding build gate')
        require(all(sha(P / n) == h for n, h in old['source_pins'].items() if n != 'cmake/sycl_backend.cmake'), 'compiled base stable except explicit Main fixture target')
        require(sha(S / 'source.diff') == '602f6dace013af8a167f17559e480d35d2dedff69d0800aef838f587062cbdee', 'Main reviewed exact patch')
        manifest = json.loads((S / 'manifest.json').read_text())
        for v in manifest['files']:
            n = v['path']
            require(sha(S / 'proposed' / n) == v['proposed_sha256'], 'proposal changed')
            require((not (P / n).exists()) if v['base_sha256'] is None else sha(P / n) == v['base_sha256'], 'exact source base')
        build = Path(old['build_path'])
        require(sha(build / 'strata') == old['binary_sha256'] == '12e1303047197a23f2184dd7232a9d43a32f1e5ad570b49de6f2a763de9ca8d6', 'prior native identity')
        require(sha(old['compiler']) == old['compiler_sha256'], 'compiler identity')
        root = B / 'main-own-prefill-host-profile-build-root-v2'
        target = A / 'own-prefill-host-profile-build-v2.json'
        require(not root.exists() and not target.exists(), 'fresh build/gates')
        root.mkdir(mode=0o700)
        record = dict(passed=False, complete=False, active=True, GPU_executed=False, commands=[],
            recipe_sha256=sha(Path(__file__)), source_diff_sha256=sha(S / 'source.diff'),
            host_fixture_sha256=sha(A / 'cpu-wait-profile-host-v2.cpp'), transformations=manifest['files'],
            source_pins=dict(old['source_pins']), build_path=str(build), environment=old['environment'],
            compiler=old['compiler'], compiler_sha256=old['compiler_sha256'], canonical_policy_passed=old['canonical_policy_passed'],
            replaced_native_sha256=old['binary_sha256'], boot_id=boot_id(), host_cases={}, gpu_cases={},
            performance_qualified=False, scope='Explicit host wait timing only; no arithmetic/queue/policy/placement change; default OFF. No CPU cause/performance claim from build.')
        def persist(): (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(root / 'owned', persist)
        record['commands'] = owner.commands
        persist()
        hostenv = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
        def run(label, argv, env=hostenv, wall=60, cpu=120, rss=2 << 30, file_cap=8 << 20):
            c, out, err = owner.run(label, argv, env, P, wall=wall, cpu=cpu,
                rss_cap=rss, text_cap=64 << 20, total_cap=1 << 30, file_cap=file_cap)
            require(module.closed(c) and c['exit_code'] == 0, label + ': ' + err.read_text()[-4000:] + '\n' + out.read_text()[-1000:])
            return out.read_text(), err.read_text()
        cursor = None
        try:
            import hashlib
            record['base_commit_membership']={}
            for item in manifest['files']:
                out=root/('base-'+str(len(record['base_commit_membership']))+'.bin')
                data,_=run('base-membership-'+str(len(record['base_commit_membership'])), ['git','show',manifest['base_commit']+':'+item['path']])
                require(hashlib.sha256(data.encode()).hexdigest()==item['base_sha256'], 'exact supplied commit membership')
                record['base_commit_membership'][item['path']]=item['base_sha256']
            record['source_pins'].update({n:sha(P/n) for n in ('cmake/sycl_backend.cmake','tools/sycl/gemm_device_probe.cpp')})
            run('host-compile', ['/usr/bin/g++', '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-pthread',
                '-I' + str(S / 'proposed/include'), str(A / 'cpu-wait-profile-host-v2.cpp'),
                '-Wl,--wrap=clock_gettime', '-o', str(root / 'host-test')])
            record['host_binary_sha256'] = sha(root / 'host-test')
            for mode in ('off', 'normal', 'false', 'unwind', 'nested-scope', 'fail-begin', 'fail-end', 'backward', 'host-normal', 'host-nested', 'host-inwait', 'host-throw', 'host-fail', 'host-nan', 'host-inf', 'overhead', 'overhead-off'):
                out, err = run('host-' + mode, [str(root / 'host-test'), mode])
                lines = [s for s in err.splitlines() if s.startswith('strata prefill cpu wait: ')]
                require((not out or mode.startswith('overhead')) and len(lines) == (0 if mode in ('off','overhead-off') else 2 if mode in ('normal', 'nested-scope') else 1), 'host report count')
                parsed = [dict(re.findall(r'(\w+)=([^ ]+)', s)) for s in lines]
                for v in parsed:
                    if mode in ('fail-begin', 'fail-end', 'backward', 'nested-scope','host-nested','host-inwait','host-fail','host-nan','host-inf'):
                        require(v['available'] == '0', 'invalid accounting must be unavailable')
                        require(all(float(value) == -1 for key, value in v.items() if key.endswith('_s')), 'invalid durations must be -1')
                    else:
                        require(v['available'] == '1' and v['complete'] == ('0' if mode in ('false', 'unwind') else '1'), 'host available/completion')
                        expected_wait='1' if mode in ('unwind','host-normal') else '0' if mode.startswith(('host-','overhead')) else '3'
                        require(v['wait_count'] == expected_wait and v['wait_throws'] == ('0' if mode.startswith(('host-','overhead')) else '1'), 'outer/throw counts')
                        require(v['compute_count'] == v['transfer_count'] == v['direct_count'] == '0', 'nested/worker exclusion')
                        require(abs(float(v['main_cpu_s']) - float(v['wait_cpu_s']) - float(v['outside_wait_cpu_s'])) < 3e-8, 'union accounting')
                        require(float(v['wall_s']) >= float(v['wait_wall_s']) >= 0, 'wall union')
                hosts=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in err.splitlines() if line.startswith('strata prefill cpu host: ')]
                require(len(hosts)==(0 if mode in ('off','overhead-off') else 2 if mode=='normal' else 1), 'boundedhost line count')
                for host in hosts:
                    if host['available']=='0':
                        require(all(float(value)==-1 for key,value in host.items() if key.endswith('_s')), 'invalid host durations')
                    else:
                        require(all(float(value)>=0 for key,value in host.items() if key.endswith('_s')), 'finite host times')
                        regions=('source_copy','host_group','gemm_submit')
                        wait=parsed[hosts.index(host)]
                        require(abs(sum(float(host[k+'_outside_wait_cpu_s']) for k in regions)+float(host['unclassified_cpu_s'])-float(wait['outside_wait_cpu_s']))<1e-7,'CPU partition')
                        if mode=='host-normal': require(all(host[k+'_count']=='1' for k in regions), 'allthree actual host scopes')
                        if mode=='host-throw': require(host['gemm_submit_throws']=='1','host unwind throw')
                        if mode=='overhead': require(host['gemm_submit_count']=='100000','actual calibration guards')
                record['host_cases'][mode]=dict(waits=parsed,hosts=hosts,calibration=out)

            run('patch-check', ['git', 'apply', '--check', str(S / 'source.diff')])
            run('patch-apply', ['git', 'apply', str(S / 'source.diff')])
            record['source_pins'].update({v['path']: v['proposed_sha256'] for v in manifest['files']})
            require(all(sha(P / n) == h for n, h in record['source_pins'].items()), 'exact applied source')
            record['source_applied'] = True
            persist()
            run('incremental-sycl-build', ['cmake', '--build', str(build), '--target', 'strata',
                'sycl_runtime_test', 'sycl_compat_test', 'sycl_gemm_device_probe', '--parallel', '2'], record['environment'],
                wall=1500, cpu=18000, rss=32 << 30, file_cap=512 << 20)
            require(all(sha(P / n) == h for n, h in record['source_pins'].items()), 'compiled source stable')
            record.update(binary=str(build / 'strata'), binary_sha256=sha(build / 'strata'),
                test_binaries={n: sha(build / n) for n in ('sycl_runtime_test', 'sycl_compat_test','sycl_gemm_device_probe')})
            require(record['binary_sha256'] != record['replaced_native_sha256'], 'new instrumentation actually linked')
            runtime = json.loads((A / 'own-selector-clean-0-r0-v1.json').read_text())
            env = dict(runtime['environment'], XDG_CACHE_HOME=str(root / 'cache'),
                UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
                UR_LOG_TRACING='level:info;flush:info;output:stderr',
                UR_LOG_LOADER='level:warning;flush:warning;output:stderr',
                UR_LOG_VALIDATION='level:warning;flush:warning;output:stderr',
                UR_LOG_LEVEL_ZERO='level:warning;flush:warning;output:stderr')
            record['small_gpu_environment'] = env
            device = Path('/sys/bus/pci/devices/0000:05:00.0')
            require((device / 'driver').resolve().name == 'xe' and not (device / 'devcoredump').exists(), 'GPU preflight')
            out, _ = run('journal-before', ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'])
            cursor = re.search(r'-- cursor: (\S+)', out)[1]
            for name in ('sycl_runtime_test', 'sycl_compat_test'):
                record['GPU_executed'] = True
                out, err = run(name, [str(build / name)], env, wall=180, cpu=1200, rss=8 << 30)
                marker = 'bounded completion wait and sticky asynchronous errors: PASS' if name == 'sycl_runtime_test' else 'and lifecycle passed'
                require(marker in out and 'strata prefill cpu wait:' not in err, 'existing small fixture/no scope')
                api_errors = [s for s in err.splitlines() if re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_', s)]
                require('UR_RESULT_SUCCESS' in err and not re.search(r'retirement failed|Strata SYCL cleanup:', err), 'trace configured/checked retirement')
                if name == 'sycl_compat_test': require(not api_errors, 'unexpected compat API failure')
                record['gpu_cases'][name] = dict(stdout=out, stdout_sha256=sha(root / ('owned/' + name + '.stdout')),
                    stderr_sha256=sha(root / ('owned/' + name + '.stderr')), API_error_lines=api_errors,
                    intentional_async_host_error_test=name == 'sycl_runtime_test')
            for label,args,cases in [('probe-smoke',['--smoke'],4),('probe-full',[],12)]:
                penv=dict(env,MKL_VERBOSE='1')
                out,err=run(label,[str(build/'sycl_gemm_device_probe'),*args],penv,wall=180,cpu=1200,rss=4<<30)
                require('DONE cases='+str(cases) in out and len(re.findall(r'^PASS type=',out,re.M))==cases,'all analytic/guard/input checks')
                require(not re.search(r'UR_RESULT_ERROR_|ZE_RESULT_ERROR_|FAIL |retirement failed|Strata SYCL cleanup:',out+err),'probe error')
                record.setdefault('GEMM_probe',{})[label]=dict(stdout=out,stderr_sha256=sha(root/('owned/'+label+'.stderr')),MKL_verbose_lines=[line for line in (out+'\n'+err).splitlines() if 'MKL_VERBOSE' in line],cases=cases,environment=penv,scope='Synthetic shapes via actualGemm wrapper; verbose target observation is not wholemodel dispatch proof or speed')
            record.update(passed=True, complete=True)

        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if cursor and owner.active is None:
                    out, _ = run('journal-final', ['journalctl', '-k', '-b', '--after-cursor', cursor, '--no-pager', '-o', 'json'])
                    entries = [json.loads(s) for s in out.splitlines() if s.strip()]
                    record.update(fresh_faults=[v for v in entries if FAULT.search(str(v.get('MESSAGE', '')))],
                        boot_unchanged=boot_id() == record['boot_id'], driver_dump_exists=device.joinpath('devcoredump').exists())
            finally:
                record.update(active=owner.active is not None, finished_utc=module.utc())
                if record['active'] or not all(module.closed(c) for c in owner.commands) or (
                    record['GPU_executed'] and ('fresh_faults' not in record or record.get('fresh_faults') or
                    not record.get('boot_unchanged') or record.get('driver_dump_exists'))):
                    record.update(passed=False, complete=False)
                persist()
                with target.open('x') as f: json.dump(record, f, indent=2); f.write('\n')
        print(json.dumps(dict(passed=record['passed'], host_cases=len(record['host_cases']),
            GPU_cases=list(record['gpu_cases']),GEMM_probe=list(record['GEMM_probe']), binary_sha256=record['binary_sha256'])), flush=True)


if __name__ == '__main__': main()
