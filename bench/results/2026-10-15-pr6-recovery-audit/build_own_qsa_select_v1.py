#!/usr/bin/env python3
"""Main-reviewed exact radix4/cyclic candidate integration and build only."""
import fcntl
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
S = A / 'own-qsa-select-radix4-v1'
T = A / 'own-qsa-select-count-mapping-v1'
sys.path.insert(0, str(A / 'owned-stall-stack-probe-v1'))
from owned_stack_probe import load_owner, require, sha


def main():
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old_path = A / 'own-prompt-k-padded-build-v1.json'
        decision_path = A / 'own-prompt-k-padded-bench-comparison-v1.json'
        old = json.loads(old_path.read_text())
        decision = json.loads(decision_path.read_text())
        require(old['passed'] and old['complete'] and not old['active'] and decision['passed']
                and not decision['pitch40_component_positive'], 'closed previous negative screen')
        require(all(module.closed(c) for c in old['commands']), 'previous build owners closed')
        require(all(sha(P / n) == h for n, h in old['source_pins'].items()), 'actual compiled base stable')
        manifest = json.loads((S / 'manifest.json').read_text())
        addon = json.loads((T / 'manifest.json').read_text())
        require(sha(S / 'proposal.patch') == '2ef1034936bf5f51d50a6245cb71c014fe5323c94f6c3d24cbc67ace6778f15a', 'reviewed full Sol patch')
        require(sha(T / 'source.diff') == addon['diff_sha256'] ==
                '40cc437726299fc190e9aaa4a99dafbf20938dbf35212729f8659aefd8d5ce8d', 'reviewed full Main adjunct')
        for n in manifest['base_files']:
            require(sha(P / n) == manifest['files']['base/' + n]['sha256'], 'immutable patch base')
        for n in manifest['new_files']:
            require(not (P / n).exists(), 'fixture is new')
        changes = {}
        for n in manifest['base_files'] + manifest['new_files']:
            require(sha(S / 'files' / n) == manifest['files']['files/' + n]['sha256'], 'exact Sol bytes')
            changes[n] = sha(S / 'files' / n)
        for n, v in addon['files'].items():
            require(changes[n] == v['base_sha256'] and sha(T / 'files' / n) == v['proposal_sha256'], 'adjunct immutable chain')
            changes[n] = v['proposal_sha256']
        require(sha(old['compiler']) == old['compiler_sha256'], 'compiler pin')
        build = Path(old['build_path'])
        require(sha(build / 'strata') == old['binary_sha256'], 'previous native pin')
        carrier = build / 'sycl_prompt_k_cellmajor_test'
        require(sha(carrier) == old['component_binary_sha256'], 'existing prompt resource carrier')
        root = B / 'main-own-qsa-select-build-root-v1'
        target = A / 'own-qsa-select-build-v1.json'
        require(not root.exists() and not target.exists(), 'fresh build')
        root.mkdir(mode=0o700)
        record = dict(passed=False, complete=False, active=True, GPU_executed=False, commands=[],
            references={p.name: sha(p) for p in (old_path, decision_path, S / 'manifest.json', T / 'manifest.json')},
            recipe_sha256=sha(Path(__file__)), proposal_patch_sha256=sha(S / 'proposal.patch'),
            adjunct_diff_sha256=sha(T / 'source.diff'), source_pins=dict(old['source_pins']),
            environment=old['environment'], build_path=str(build), compiler=old['compiler'],
            compiler_sha256=old['compiler_sha256'], canonical_policy_passed=old['canonical_policy_passed'],
            retained_reference_binary=old['retained_reference_binary'], replaced_native_sha256=sha(build / 'strata'),
            retained_prompt_component=dict(path=str(carrier), sha256=sha(carrier), bytes=carrier.stat().st_size,
                consumer='Main R739 exact prompt JIT resource capture', budget_bytes=32 << 20,
                next_review='after actualJIT capture closes or is superseded'),
            component_qualified=False, performance_qualified=False, quality_qualified=False,
            restore_qualified=False, physical_262144_qualified=False,
            scope='Own exact selection candidates defaultOFF; reviewed Sol radix4 patch then Main count-only cyclic adjunct. Build only, no GPU or source from XeStrata.')
        def persist():
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(root / 'owned', persist)
        record['commands'] = owner.commands
        persist()
        hostenv = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
        try:
            for label, patch in [('radix4', S / 'proposal.patch'), ('cyclic', T / 'source.diff')]:
                for step, argv in [('check', ['git', 'apply', '--check', str(patch)]), ('apply', ['git', 'apply', str(patch)])]:
                    c, out, err = owner.run(label + '-' + step, argv, hostenv, P,
                        wall=60, cpu=60, rss_cap=2 << 30, text_cap=1 << 20, total_cap=1 << 30, file_cap=1 << 20)
                    require(module.closed(c) and c['exit_code'] == 0, label + ': ' + err.read_text()[-2000:])
            record['source_pins'].update(changes)
            require(all(sha(P / n) == h for n, h in record['source_pins'].items()), 'exact complete applied source')
            record['source_applied'] = True
            persist()
            c, out, err = owner.run('incremental-sycl-build',
                ['cmake', '--build', str(build), '--target', 'strata', 'sycl_qsa_select_radix_test', '--parallel', '2'],
                record['environment'], P, wall=1200, cpu=14400,
                rss_cap=32 << 30, text_cap=16 << 20, total_cap=1 << 30, file_cap=512 << 20)
            require(module.closed(c) and c['exit_code'] == 0, 'SYCL build: ' + err.read_text()[-3000:] + '\n' + out.read_text()[-3000:])
            require(all(sha(P / n) == h for n, h in record['source_pins'].items()), 'compiled source stable')
            require(sha(carrier) == record['retained_prompt_component']['sha256'], 'previous prompt component not relinked')
            exe = build / 'sycl_qsa_select_radix_test'
            record.update(passed=True, complete=True, binary=str(build / 'strata'), binary_sha256=sha(build / 'strata'),
                          component_binary=str(exe), component_binary_sha256=sha(exe))
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record.update(active=owner.active is not None, finished_utc=module.utc())
            if record['active'] or not all(module.closed(c) for c in owner.commands):
                record.update(passed=False, complete=False)
            persist()
            with target.open('x') as f:
                json.dump(record, f, indent=2)
                f.write('\n')
        print(json.dumps({k: record[k] for k in ('passed', 'binary_sha256', 'component_binary_sha256')}), flush=True)


if __name__ == '__main__':
    main()
