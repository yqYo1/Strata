#!/usr/bin/env python3
"""Incremental Main build for the short-prompt dispatcher and bounded progress."""
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from check_gemm_orientation import P, B, D, OWNER, sha

SEED = D/'layer-major-build-v2.json'
O = B/'main-primary-layer-major-dispatch-build-root-v1'
CHANGED = {'src/prefill/layer_major.inc'}


def main():
    seed = json.loads(SEED.read_text())
    assert seed['passed'] and seed['complete'] and not seed['active']
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ['layer-major-small-v1.json', 'layer-major-long-v1.json']:
            gate = json.loads((D/name).read_text())
            assert gate['passed'] and gate['complete'] and not gate['active']
            assert not gate['fresh_faults'] and not gate['driver_dump_exists']
            assert gate['boot_unchanged']
            assert all(c['direct_child_reaped'] and c['session_empty'] for c in gate['commands'])
        drift = {n for n, pin in seed['source_pins'].items() if sha(P/n) != pin}
        assert drift == CHANGED, drift
        build = Path(seed['build_path'])
        assert all(sha(build/n) == pin for n, pin in seed['binaries'].items())
        control = Path(seed['control_binary']['path'])
        assert sha(control) == seed['control_binary']['sha256']
        assert not O.exists()
        O.mkdir(mode=0o700)
        env = dict(seed['environment'])
        assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
        diff = subprocess.check_output(['git', 'diff', 'HEAD', '--', *sorted(CHANGED)], cwd=P)
        r = dict(active=True, passed=False, complete=False, GPU_executed=False,
            commands=[], environment=env, build_path=str(build),
            seed_receipt_sha256=sha(SEED), recipe_sha256=sha(Path(__file__)),
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),
            source_diff_sha256=hashlib.sha256(diff).hexdigest(),
            source_pins={n:sha(P/n) for n in seed['source_pins']},
            control_binary=seed['control_binary'], previous_binaries=seed['binaries'],
            host_contracts_passed=seed['host_contracts_passed'],
            host_contracts_reused_from=sha(SEED), host_test_rerun=False,
            performance_qualified=False, quality_qualified=False,
            physical_262144_qualified=False, restore_qualified=False, adopted=False,
            scope='Only exact-flag dispatcher now keeps n<=chunk on run_impl; '
                  'STRATA_TRACE adds two flushed progress lines per model layer. '
                  'Incremental build only; prior pure-CPU arithmetic test unchanged. '
                  'Separate GPU validation is required for the resulting executable.')

        def persist():
            (O/'record.json').write_text(json.dumps(r,indent=2)+'\n')

        owner = module.Owner(O/'owned',persist)
        r['commands'] = owner.commands
        persist()
        try:
            entry,out,err = owner.run('build-layer-major-dispatch',
                ['cmake','--build',str(build),'--target','strata','--parallel','2'],
                env,P,wall=1500,cpu=1500*len(os.sched_getaffinity(0)),
                text_cap=64 << 20,rss_cap=48 << 30,total_cap=1 << 30,file_cap=256 << 20)
            assert module.closed(entry) and entry['exit_code']==0, (
                entry.get('error'),err.read_text()[-6000:],out.read_text()[-2000:])
            r['binaries'] = {n:sha(build/n) for n in seed['binaries']}
            assert r['binaries']['prefill_layer_major_plan_test'] == seed['binaries']['prefill_layer_major_plan_test']
            assert r['binaries']['strata'] != seed['binaries']['strata']
            commands=json.loads((build/'compile_commands.json').read_text())
            proof=[c for c in commands if c['file']==str(P/'src/prefill/prefill.cpp')]
            assert len(proof)==1 and '-fsycl-id-queries-range=size_t' in proof[0]['command']
            r['compile_flag_proof'] = proof[0]
            r['compile_commands_sha256'] = sha(build/'compile_commands.json')
            assert all(sha(P/n)==pin for n,pin in r['source_pins'].items())
            assert sha(control)==seed['control_binary']['sha256']
            r.update(passed=True,complete=True)
        except BaseException as error:
            r['error']=repr(error)
            raise
        finally:
            r['active']=False
            persist()
            (D/'layer-major-dispatch-build-v1.json').write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps(dict(passed=True,GPU_executed=False,binaries=r['binaries'])),flush=True)


if __name__=='__main__':
    main()
