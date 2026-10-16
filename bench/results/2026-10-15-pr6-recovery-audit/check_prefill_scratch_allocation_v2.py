#!/usr/bin/env python3
"""Main serial first GPU allocation check with complete bounded UR API logging."""
import fcntl
import json
import os
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
from analyze_prefill_window_profile_v2 import B, P, load_owner, require, sha
from run_prefill_window_profile_v4 import DEVICE, FAULT, boot_id


def main():
    root = B/'main-prefill-scratch-allocation-root-v2'
    target = A/'prefill-scratch-allocation-v2.json'
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not root.exists() and not target.exists(), 'fresh GPU component')
        failed_path = A/'prefill-scratch-allocation-v1.json'
        failed = json.loads(failed_path.read_text())
        require(not failed['passed'] and not failed['active'] and failed['component_exit_code'] == 88 and
            not failed['fresh_faults'] and failed['boot_unchanged'] and not failed['driver_dump_exists'],
            'preserved count-mismatch failure, no GPU fault')
        paths = {k:A/n for k,n in dict(build='prefill-scratch-allocation-build-v2.json',
            host='prefill-scratch-alias-build-host-v2.json',
            prior='prefill-clean-c8192-r0-v1.json').items()}
        refs = {k:json.loads(p.read_text()) for k,p in paths.items()}
        build, host, prior = [refs[k] for k in ('build','host','prior')]
        require(all(r['passed'] and r['complete'] and not r['active'] for r in refs.values()), 'actual closed gates')
        require(not prior['fresh_faults'] and prior['boot_unchanged'] and not prior['driver_dump_exists'] and
                prior['boot_id'] == boot_id(), 'fresh GPU fault/boot gate')
        require(all(sha(P/n) == h for n,h in build['source_pins'].items()), 'exact product source')
        require(all(sha(Path(n)) == h for n,h in build['linked_inputs'].items()), 'exact linked inputs')
        fixture = A/'prefill_scratch_allocation_fixture_v2.cpp'
        require(sha(fixture) == build['fixture_sha256'] and
                sha(Path(build['binary'])) == build['binary_sha256'], 'fixture/binary identity')
        module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        for r in [*refs.values(), failed]:
            for c in r['commands']:
                require(module.closed(c), 'previous command not closed')
                for old in c['owners']:
                    now = module.identity(old['pid'])
                    require(now is None or now['start_ticks'] != old['start_ticks'], 'previous owner present')
        require((DEVICE/'driver').resolve().name == 'xe' and not (DEVICE/'devcoredump').exists(), 'driver/dump gate')
        args = prior['engine_args']; pack = Path(args[args.index('--pack')+1])
        require(all(sha(Path(n)) == h for n,h in prior['pack_metadata_sha256'].items()), 'exact pack metadata')
        root.mkdir(mode=0o700)
        env = dict(prior['environment'])
        for key in ('STRATA_MEASUREMENT_ENGINE_START','STRATA_MEASUREMENT_ENGINE_EXIT','STRATA_PREFILL_RELEASE_HEADS'):
            env.pop(key, None)
        env.update(STRATA_SYCL_PREFILL_SCRATCH_ALIAS='1', STRATA_GR_UNFUSED='0',
            STRATA_PREFILL_BF16X2='0', STRATA_PF_FUSED='0', STRATA_PREFILL_MMQ='0',
            UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
            UR_LOG_TRACING='level:info;flush:info;output:stderr',
            XDG_CACHE_HOME=str(root/'cache'))
        require(not any('TOKEN' in k or 'API_KEY' in k or k == 'LD_PRELOAD' or
                k.startswith(('UNITRACE_','PTI_')) for k in env), 'bounded first-check environment')
        for k,v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', NEOReadDebugKeys='1',
            EnableDirectSubmission='0', STRATA_SYCL_PREFILL_SAME_QUEUE='1',
            SYCL_CACHE_PERSISTENT='0', STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0').items():
            require(env[k] == v, 'unchanged safety flag '+k)
        r = dict(passed=False, complete=False, active=True, GPU_executed=False, commands=[],
            recipe_sha256=sha(Path(__file__)), references={k:sha(v) for k,v in paths.items()}, prior_failed_receipt_sha256=sha(failed_path),
            binary_sha256=build['binary_sha256'], fixture_sha256=sha(fixture), environment=env,
            pack_metadata_sha256=prior['pack_metadata_sha256'], boot_id=boot_id(),
            limits=dict(wall_seconds=240, RSS_bytes=16 << 30, text_bytes=64 << 20, job_bytes=256 << 20),
            quality_qualified=False, performance_qualified=False, restore_qualified=False,
            corrected_fixture_assumption='bytes_needed=actual alias carve+8388608 slack; exact carve and carve-minus-one are tested separately; original failed receipt unchanged.',
            physical_model_262144_qualified=False,
            scope='Actual production Prefill C8192 owned and exact-carve-sized borrowed allocation, C4096 relayout, canaries and guarded rejections. Real262144 single-layer staging allocation, metadata-only session. No inference/model/state/math/quality or max-context qualification.',
            retention_consumer='First alias-owned/borrowed/relayout allocation/retirement comparison; after outcome extraction retire full successful API history.')
        def persist():
            (root/'record.json').write_text(json.dumps(r, indent=2)+'\n')
        owner = module.Owner(root/'owned', persist); r['commands'] = owner.commands; persist()
        host_env = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
        def utility(label, argv):
            c, out, err = owner.run(label, argv, host_env, P, wall=60, cpu=60,
                rss_cap=2 << 30, text_cap=4 << 20, total_cap=256 << 20, file_cap=4 << 20)
            require(module.closed(c) and c['exit_code'] == 0, label+': '+err.read_text()[-1000:])
            return out.read_text()
        cursor = None
        try:
            cursor = re.search(r'-- cursor: (\S+)', utility('journal-before',
                ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager']))[1]
            r['journal_before_cursor'] = cursor; r['GPU_executed'] = True; persist()
            c, out, err = owner.run('actual-scratch-allocation', [build['binary'], str(pack)], env, P,
                wall=240, cpu=240*len(os.sched_getaffinity(0)), rss_cap=16 << 30,
                text_cap=64 << 20, total_cap=256 << 20, file_cap=64 << 20)
            r['stdout'] = out.read_text(); stderr = err.read_text()
            r['component_exit_code'] = c.get('exit_code')
            r['runtime_errors'] = [line for line in stderr.splitlines() if
                re.search(r'(?:UR|ZE)_RESULT_ERROR_|scratch allocation fail-stop|terminate called|Strata SYCL cleanup:', line)]
            require(module.closed(c) and c['exit_code'] == 0, 'actual allocation check: '+stderr[-5000:])
            require(not r['runtime_errors'], 'first-check runtime error')
            require('scratch_allocation PASS owned borrowed relayout short_loan null_queue unordered_queue split no_inference' in r['stdout'], 'complete component checks')
            r['carve_lines'] = [line for line in stderr.splitlines() if line.startswith('strata prefill: FP16 MoE compact')]
            require(len(r['carve_lines']) >= 3 and
                all('chunk=8192' in line and 'accounted=2588395008' in line for line in r['carve_lines'][:2]) and
                'chunk=4096' in r['carve_lines'][2] and 'accounted=1503447296' in r['carve_lines'][2],
                'actual carve/owned/borrowed/relayout consistency')
            require('C8192=2596783616 C4096=1511835904 carve8192=2588395008 carve4096=1503447296 loan_slack=8388608' in r['stdout'], 'loan versus actual carve accounting')
            r['native_log_sha256'] = sha(err)
            r.update(passed=True, complete=True)
        except BaseException as e:
            r['error'] = repr(e)
            raise
        finally:
            try:
                if cursor and owner.active is None:
                    entries = [json.loads(s) for s in utility('journal-final',
                        ['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if s.strip()]
                    r.update(fresh_faults=[x for x in entries if FAULT.search(str(x.get('MESSAGE','')))],
                        boot_unchanged=boot_id() == r['boot_id'], driver_dump_exists=(DEVICE/'devcoredump').exists())
            finally:
                r.update(active=owner.active is not None, finished_utc=module.utc())
                if (r['active'] or not all(module.closed(c) for c in r['commands']) or
                        r.get('fresh_faults') or not r.get('boot_unchanged') or r.get('driver_dump_exists')):
                    r.update(passed=False, complete=False)
                persist(); target.write_text(json.dumps(r, indent=2)+'\n')
        print(json.dumps(dict(passed=r['passed'], exit_code=r.get('component_exit_code'),
                              fresh_faults=len(r.get('fresh_faults',[])))))


if __name__ == '__main__':
    main()
