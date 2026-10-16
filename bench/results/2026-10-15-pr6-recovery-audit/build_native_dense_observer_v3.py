#!/usr/bin/env python3
"""Main serial generator rebuild for bounded dense-phase runtime observation."""
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
        root = B/'main-native-dense-observer-build-root-v3'
        target = A/'native-dense-phase-build-v3.json'
        require(not root.exists() and not target.exists(), 'fresh observer build')
        previous_path = A/'native-dense-phase-build-v2.json'
        previous = json.loads(previous_path.read_text())
        require(previous['passed'] and previous['complete'] and not previous['active'], 'actual build')
        refs = {previous_path.name:sha(previous_path)}
        module = load_owner(P/'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
        for mode in ('normal','oom-null','unknown-null','unknown-nonnull','oom-nonnull','success-null'):
            path = A/f'native-dense-bank-component-{mode}-v2.json'
            r = json.loads(path.read_text())
            require(r['passed'] and r['complete'] and not r['active'] and not r['fresh_faults']
                and r['boot_unchanged'] and not r['driver_dump_exists'], 'closed actual component '+mode)
            refs[path.name] = sha(path)
            for c in r['commands']:
                require(module.closed(c), 'preceding command closed')
                for old in c['owners']:
                    now = module.identity(old['pid'])
                    require(now is None or now['start_ticks'] != old['start_ticks'], 'preceding owner retired')
        require(all(sha(P/n) == h for n,h in previous['source_pins'].items()
            if n != 'src/program/generate.cpp'), 'only generator observer changed')
        build = Path(previous['build_path'])
        libraries = {str(p):sha(p) for p in sorted(build.glob('libstrata*.a'))}
        record = dict(previous)
        root.mkdir(mode=0o700)
        record.update(passed=False, complete=False, active=True, commands=[],
            recipe_sha256=sha(Path(__file__)), references=refs,
            replaced_product_binary=dict(path=previous['binary'], sha256=sha(Path(previous['binary'])),
                bytes=Path(previous['binary']).stat().st_size,
                reason='Rebuild generator only with diagnostic-only phase/free-memory observations; full source and original receipt remain reachable.'),
            unchanged_archives=libraries, source_pins={n:sha(P/n) for n in previous['source_pins']},
            scope='Actual generator rebuilt for STRATA_TRACE-only dense retire/bank/restore receipts. Component libraries and fixture binaries remain identical to the six passed GPU arms. No GPU execution.')
        def persist():
            (root/'record.json').write_text(json.dumps(record, indent=2)+'\n')
        owner = module.Owner(root/'owned', persist)
        record['commands'] = owner.commands
        persist()
        try:
            c,out,err = owner.run('rebuild-native-dense-observer',
                ['cmake','--build',str(build),'--target','strata','--parallel','2'],
                record['environment'], P, wall=300, cpu=300*len(os.sched_getaffinity(0)),
                rss_cap=16<<30, text_cap=8<<20, total_cap=512<<20, file_cap=256<<20)
            require(module.closed(c) and c['exit_code'] == 0, 'observer build: '+err.read_text()[-3000:]+'\n'+out.read_text()[-2000:])
            require(all(sha(Path(n)) == h for n,h in libraries.items()), 'component archive changed')
            require(all(sha(P/n) == h for n,h in record['source_pins'].items()), 'source stability')
            record['binaries'] = {n:sha(build/n) for n in previous['binaries']}
            require(all(record['binaries'][n] == h for n,h in previous['binaries'].items()
                if n != 'strata'), 'component binaries unchanged')
            record.update(binary_sha256=record['binaries']['strata'], passed=True, complete=True)
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
            binary_sha256=record['binary_sha256'], component_archives_unchanged=True)), flush=True)


if __name__ == '__main__':
    main()
