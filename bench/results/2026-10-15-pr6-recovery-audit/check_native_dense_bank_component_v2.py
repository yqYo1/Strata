#!/usr/bin/env python3
"""Main serial native-bank gate with the corrected native pack loader."""
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

MODES = ('normal', 'oom-null', 'unknown-null', 'unknown-nonnull', 'oom-nonnull', 'success-null')


def main(mode):
    require(mode in MODES, 'declared component arm')
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = B/f'main-native-dense-bank-component-{mode}-root-v2'
        target = A/f'native-dense-bank-component-{mode}-v2.json'
        require(not root.exists() and not target.exists(), 'fresh component arm')
        paths = dict(build=A/'native-dense-phase-build-v2.json',
            prior=A/'qsa-device-batch-clean-1-r2-v1.json',
            seed=A/'qsa-device-batch-small-v1.json')
        refs = {k:json.loads(p.read_text()) for k,p in paths.items()}
        build,prior,seed = (refs[k] for k in ('build','prior','seed'))
        require(all(r['passed'] and r['complete'] and not r['active'] for r in refs.values()), 'closed build and GPU gates')
        require(not prior['fresh_faults'] and prior['boot_unchanged'] and not prior['driver_dump_exists']
            and prior['boot_id'] == boot_id(), 'healthy same-boot GPU predecessor')
        if mode != 'normal':
            paths['normal'] = A/'native-dense-bank-component-normal-v2.json'
            normal = json.loads(paths['normal'].read_text())
            require(normal['passed'] and normal['complete'] and not normal['active']
                and not normal['fresh_faults'] and normal['boot_unchanged']
                and not normal['driver_dump_exists'], 'normal payload/restore gate first')
            refs['normal'] = normal
        module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        for r in refs.values():
            for c in r['commands']:
                require(module.closed(c), 'preceding owned command closed')
                for old in c['owners']:
                    now = module.identity(old['pid'])
                    require(now is None or now['start_ticks'] != old['start_ticks'], 'preceding owner retired')
        require(all(sha(P/n) == h for n,h in build['source_pins'].items()), 'actual built source identity')
        name = 'sycl_native_dense_bank_test' if mode == 'normal' else 'sycl_native_dense_bank_fault_test'
        exe = Path(build['build_path'])/name
        require(sha(exe) == build['binaries'][name], 'actual linked fixture')
        model = json.loads((A/'selective-iq-clean-bench-full-0-v1.json').read_text())
        shards = sorted(model['model_identity'])
        require(len(shards) == 2, 'actual target shards')
        for n,v in model['model_identity'].items():
            st = Path(n).stat()
            require(st.st_size == v['bytes'] and st.st_mtime_ns == v['mtime_ns'], 'model identity')
        require(all(sha(Path(n)) == h for n,h in model['pack_metadata_sha256'].items()), 'pack metadata identity')
        pack = Path(model['engine_args'][model['engine_args'].index('--pack')+1])
        require((DEVICE/'driver').resolve().name == 'xe' and not (DEVICE/'devcoredump').exists(), 'driver preflight')
        root.mkdir(mode=0o700)
        env = dict(seed['environment'])
        env.update(XDG_CACHE_HOME=str(root/'cache'), STRATA_PREFILL_NATIVE_DENSE_BANK='1')
        if mode != 'normal':
            env.pop('UR_LOG_TRACING', None)
            env['UR_ENABLE_LAYERS'] = 'UR_LAYER_PARAMETER_VALIDATION'
        for k,v in dict(UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD='1', NEOReadDebugKeys='1',
            EnableDirectSubmission='0', STRATA_SYCL_PREFILL_SAME_QUEUE='1',
            SYCL_CACHE_PERSISTENT='0', STRATA_SYCL_VERIFY_NATIVE_CAPTURE='0').items():
            require(env[k] == v, 'unchanged safety option '+k)
        require(not any('TOKEN' in k or 'API_KEY' in k or k == 'LD_PRELOAD'
            or k.startswith(('UNITRACE_', 'PTI_', 'STRATA_MEASUREMENT_')) for k in env), 'bounded component environment')
        argv = [str(exe)] + ([] if mode == 'normal' else [mode]) + [str(pack), *shards]
        record = dict(passed=False, complete=False, active=True, GPU_executed=False, mode=mode,
            recipe_sha256=sha(Path(__file__)), commands=[], environment=env, argv=argv,
            references={p.name:sha(p) for p in paths.values()}, binary_sha256=sha(exe),
            source_pins=build['source_pins'], model_identity=model['model_identity'],
            pack_metadata_sha256=model['pack_metadata_sha256'], boot_id=boot_id(),
            limits=dict(wall_seconds=240, RSS_bytes=16<<30, text_bytes=64<<20, job_bytes=128<<20),
            quality_qualified=False, performance_qualified=False, restore_qualified=False,
            physical_model_262144_qualified=False,
            scope='Actual 300 native GGUF matrices, owned original packed RAM, all48 layer activations, exact packed payload/metadata before/bank/restore, repeated enter/restore, and one declared allocation outcome. No generator transaction, model state/task quality, full context, physical allocator recovery or speed proof.',
            retention_consumer='Extract complete result and runtime/owner/fault evidence, then retire successful full UR trace; retain only distinct failure mechanism excerpts.')
        def persist():
            (root/'record.json').write_text(json.dumps(record, indent=2)+'\n')
        owner = module.Owner(root/'owned', persist)
        record['commands'] = owner.commands
        persist()
        host_env = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
        def utility(label, args):
            c,out,err = owner.run(label, args, host_env, P, wall=60, cpu=60,
                rss_cap=2<<30, text_cap=4<<20, total_cap=128<<20, file_cap=4<<20)
            require(module.closed(c) and c['exit_code'] == 0, label+': '+err.read_text()[-2000:])
            return out.read_text()
        cursor = None
        try:
            cursor = re.search(r'-- cursor: (\S+)', utility('journal-before',
                ['journalctl','-k','-b','-n','0','--show-cursor','--no-pager']))[1]
            record.update(journal_before_cursor=cursor, GPU_executed=True)
            persist()
            c,out,err = owner.run('native-dense-bank-'+mode, argv, env, P, wall=240,
                cpu=240*len(os.sched_getaffinity(0)), rss_cap=16<<30,
                text_cap=64<<20, total_cap=128<<20, file_cap=64<<20)
            stdout,stderr = out.read_text(),err.read_text()
            record.update(stdout=stdout, stderr_sha256=sha(err), exit_code=c['exit_code'])
            record['runtime_errors'] = [line for line in stderr.splitlines() if
                re.search(r'(?:UR|ZE)_RESULT_ERROR_|terminate called|Strata SYCL cleanup:|FAIL', line)]
            require(module.closed(c) and not record['runtime_errors'], 'closed component without runtime error: '+stderr[-3000:])
            if mode in ('normal','oom-null'):
                require(c['exit_code'] == 0 and stdout.strip() ==
                    'PASS matrices=300 layers=48 snapshot=2116951040 bank=50918400',
                    'complete actual payload/restore: '+stderr[-3000:])
                record.update(all_300_payloads_and_metadata_checked=True, layers=48,
                    snapshot_bytes=2116951040, bank_bytes=50918400, repeated_enter_restore=True)
            else:
                reason = 'successful bank allocation returned null' if mode == 'success-null' else 'uncertain bank allocation failure'
                expected = 'native dense layer bank: fail-stop in '+reason
                require(c['exit_code'] == -6 and stderr.count(expected) == 1 and
                    'unexpected ordinary enter return' not in stderr and 'PASS matrices=' not in stdout,
                    'exact production allocation fail-stop mechanism: '+stderr[-3000:])
                record.update(expected_fail_stop=True, failure_operation=reason, mechanism_excerpt=expected)
            require(all(sha(P/n) == h for n,h in build['source_pins'].items()), 'source stability')
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            try:
                if cursor and owner.active is None:
                    entries = [json.loads(line) for line in utility('journal-final',
                        ['journalctl','-k','-b','--after-cursor',cursor,'--no-pager','-o','json']).splitlines() if line.strip()]
                    record.update(fresh_faults=[e for e in entries if FAULT.search(str(e.get('MESSAGE','')))],
                        boot_unchanged=boot_id() == record['boot_id'], driver_dump_exists=(DEVICE/'devcoredump').exists())
            finally:
                record.update(active=owner.active is not None, finished_utc=module.utc())
                if record['active'] or not all(module.closed(c) for c in record['commands']) or 'fresh_faults' not in record \
                    or record.get('fresh_faults') or not record.get('boot_unchanged') or record.get('driver_dump_exists'):
                    record.update(passed=False, complete=False)
                persist()
                with target.open('x') as output:
                    json.dump(record, output, indent=2)
                    output.write('\n')
        print(json.dumps(dict(passed=record['passed'], mode=mode,
            exit_code=record.get('exit_code'), fresh_faults=len(record.get('fresh_faults',[])))), flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2, 'one declared component arm')
    main(sys.argv[1])
