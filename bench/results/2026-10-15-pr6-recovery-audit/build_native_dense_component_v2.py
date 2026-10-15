#!/usr/bin/env python3
"""Correct the component's pack loader; rebuild only its two manual targets."""
import fcntl
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
from analyze_prefill_window_profile_v2 import B, P, load_owner, require, sha


def main():
    with (B/'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root = B/'main-native-dense-component-build-root-v2'
        target = A/'native-dense-phase-build-v2.json'
        require(not root.exists() and not target.exists(), 'fresh corrected fixture build')
        original_path = A/'native-dense-phase-build-v1.json'
        original = json.loads(original_path.read_text())
        failed_path = A/'native-dense-bank-component-normal-v1.json'
        failed = json.loads(failed_path.read_text())
        require(original['passed'] and original['complete'] and not original['active'], 'full native build passed')
        require(not failed['passed'] and not failed['active'] and failed['exit_code'] == -6 and
            failed['runtime_errors'] == ['FAIL: output.weight: this pack holds the tensor only in its GGUF form (run with --native SHARD1)'] and
            not failed['fresh_faults'] and failed['boot_unchanged'] and not failed['driver_dump_exists'], 'exact closed fixture loader failure')
        module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        for r in (original,failed):
            for c in r['commands']:
                require(module.closed(c), 'preceding commands closed')
                for old in c['owners']:
                    now = module.identity(old['pid'])
                    require(now is None or now['start_ticks'] != old['start_ticks'], 'preceding owner retired')
        changed = 'tests/sycl/native_dense_bank.cpp'
        require(all(sha(P/n) == h for n,h in original['source_pins'].items() if n != changed), 'only fixture source changed')
        build = Path(original['build_path'])
        require(sha(Path(original['binary'])) == original['binary_sha256'], 'unchanged product binary')
        targets = ('sycl_native_dense_bank_test','sycl_native_dense_bank_fault_test')
        replaced = [dict(path=str(build/n), bytes=(build/n).stat().st_size,
            sha256=sha(build/n), original_build=original_path.name,
            reason='Rebuild component-only executables with the actual native pack skip set; original failed receipt and source commit1025e0ed remain reachable.') for n in targets]
        require(all(x['sha256'] == original['binaries'][Path(x['path']).name] for x in replaced), 'original fixture identities')
        root.mkdir(mode=0o700)
        record = dict(original)
        record.update(passed=False, complete=False, active=True, commands=[],
            recipe_sha256=sha(Path(__file__)), references={original_path.name:sha(original_path),failed_path.name:sha(failed_path)},
            replaces_component_binaries=replaced, source_pins={n:sha(P/n) for n in original['source_pins']},
            scope='Original full product build reused byte-identically; only two real-model component targets recompiled for correct native pack loader admission. No GPU execution.')
        def persist():
            (root/'record.json').write_text(json.dumps(record, indent=2)+'\n')
        owner = module.Owner(root/'owned', persist)
        record['commands'] = owner.commands
        persist()
        try:
            c,out,err = owner.run('rebuild-native-dense-fixtures',
                ['cmake','--build',str(build),'--target',*targets,'--parallel','2'],
                record['environment'], P, wall=300, cpu=300*len(os.sched_getaffinity(0)),
                rss_cap=16<<30, text_cap=8<<20, total_cap=512<<20, file_cap=256<<20)
            require(module.closed(c) and c['exit_code'] == 0, 'fixture rebuild: '+err.read_text()[-3500:]+'\n'+out.read_text()[-2500:])
            require(sha(Path(original['binary'])) == original['binary_sha256'], 'product binary unchanged after fixture rebuild')
            require(all(sha(P/n) == h for n,h in record['source_pins'].items()), 'source stability')
            record['binaries'] = {n:sha(build/n) for n in ('strata',*targets)}
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record.update(active=owner.active is not None, finished_utc=module.utc())
            persist()
            with target.open('x') as output:
                json.dump(record, output, indent=2)
                output.write('\n')
        print(json.dumps(dict(passed=record['passed'], GPU_executed=False,
            product_binary_unchanged=True, binaries=record['binaries'])), flush=True)


if __name__ == '__main__':
    main()
