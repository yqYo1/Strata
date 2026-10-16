#!/usr/bin/env python3
"""Compact three closed NLL jobs, preserving original PASS and FAIL receipts."""
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
from owned_stack_probe import load_owner, require, sha


def main():
    module = load_owner(P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        compact = A / 'closed-own-dense-prefill-nll-compact-v1.json'
        target = A / 'closed-own-dense-prefill-nll-retention-v1.json'
        require(not compact.exists() and not target.exists(), 'fresh closure')
        roots = [
            ('own-dense-prefill-nll-build-v3.json', 'main-own-dense-prefill-nll-build-root-v3'),
            ('own-dense-prefill-nll-smoke-v3.json', 'main-own-dense-prefill-nll-smoke-root-v3'),
            ('own-dense-prefill-nll-smoke-v4.json', 'main-own-dense-prefill-nll-smoke-root-v4')]
        evidence = dict(original_result_status_unchanged=True, cases={}, compile_proof=[])
        deletions = []
        for receipt_name, root_name in roots:
            source = A / receipt_name
            root = B / root_name
            r = json.loads(source.read_text())
            require(not r['active'] and sha(root / 'record.json') == sha(source), 'exact closed durable receipt')
            for c in r['commands']:
                require(module.closed(c), 'owned command terminal')
                for old in c['owners']:
                    now = module.identity(old['pid'])
                    require(now is None or now['start_ticks'] != old['start_ticks'], 'owned process still present')
            e = dict(receipt_sha256=sha(source), passed=r['passed'], complete=r['complete'],
                     normal_command_exits=[dict(label=c['label'], exit_code=c['exit_code']) for c in r['commands']])
            if 'build_path' in r:
                require(r['passed'] and r['complete'] and not r['GPU_executed'], 'host/build scope')
                cc = json.loads((Path(r['build_path']) / 'compile_commands.json').read_text())
                proof = [x for x in cc if x['file'].endswith('dense_prefill_nll.cpp')]
                require(len(proof) == 3 and all('-fno-fast-math' in x['command'] and '-ffp-contract=off' in x['command'] and '-std=c++20' in x['command'] for x in proof), 'actual strict math compile commands')
                evidence['compile_proof'] = proof
                e.update(build_stdout=(root / 'owned/build-nll.stdout').read_text(),
                         build_stderr=(root / 'owned/build-nll.stderr').read_text(),
                         host_stdout=(root / 'owned/host-nll.stdout').read_text())
                require(e['host_stdout'] == r['host_fixture_stdout'], 'host PASS preserved')
                for i, name in enumerate(('src/program/generate.cpp', 'cmake/sycl_backend.cmake')):
                    require(sha(root / f'owned/base-membership-{i}.stdout') == r['base_commit_membership'][name], 'original source membership proof')
            else:
                require(not r['fresh_faults'] and r['boot_unchanged'] and not r['driver_dump_exists'], 'closed GPU health')
                if receipt_name.endswith('smoke-v3.json'):
                    text = (root / 'owned/native-off.stderr').read_text()
                    require(not r['passed'] and 'MTP expert release requires single-GPU native MTP4' in text and 'completed 2 chunks' not in text, 'distinct pre-prefill configuration failure')
                    e['failure_excerpt'] = text
                    e['failure_reason'] = 'spec0 plus inherited MTP-expert-release flag; observer was never reached; v4 restored actual MTP4 recipe.'
                else:
                    require(r['passed'] and r['complete'] and r['all_three_fullvocabulary_rows_byteexact'], 'paired observer transparency')
                    e['native'] = {}
                    for arm in ('off', 'on'):
                        runroot = root / arm
                        text = (root / f'owned/native-{arm}.stderr').read_text()
                        c = r['cases'][arm]
                        require(sha(runroot / 'selected.bin') == c['selected_sha256'] and (runroot / 'selected.bin.meta').read_text() == c['selected_meta'] and sha(root / f'owned/native-{arm}.stderr') == c['native_stderr_sha256'], 'selected capture and progress pins')
                        lines = [line for line in text.splitlines() if re.search(r'layer-major layer \d+ of 48: completed 2 chunks|strata dense prefill NLL:|strata prefill dense:|strata prefill MTP|strata prefill: layer-major candidate,', line)]
                        require(len(re.findall(r'layer-major layer \d+ of 48: completed 2 chunks', '\n'.join(lines))) == 48, 'all layer completions retained')
                        e['native'][arm] = dict(progress_and_lifecycle=lines, stdout=(root / f'owned/native-{arm}.stdout').read_text())
                    require((root / 'off/selected.bin').read_bytes() == (root / 'on/selected.bin').read_bytes(), 'reverified byte equality before retirement')
                    require(sha(root / 'on/nll.bin') == r['cases']['on']['nll_sha256'] and (root / 'on/nll.bin.meta').read_text() == r['cases']['on']['nll_metadata'], 'all128 NLL values and completion metadata retained in receipt')
                    e['retired_tensor_consumer'] = 'Paired observer-transparency gate and independent selectedrow oracle completed. No future raw consumer.'
            evidence['cases'][receipt_name] = e
            for path in root.rglob('*'):
                if not path.is_file():
                    continue
                require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), 'private raw artifact')
                deletions.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path), status='planned',
                                      reason='Closed diagnostic; original results, owners/health, all128 NLLs, exact selected comparison, strict compile proof and distinct first configuration failure preserved. Raw consumer completed.'))
        compact.write_text(json.dumps(evidence, indent=2) + '\n')
        manifest = dict(status='planned', owner='Main', recipe_sha256=sha(Path(__file__)),
                        compact_sha256=sha(compact), original_result_status_unchanged=True,
                        physical_free_space_claim=False, deletions=deletions)
        target.write_text(json.dumps(manifest, indent=2) + '\n')
        for item in deletions:
            path = Path(item['path'])
            require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'raw identity before unlink')
            path.unlink()
            item['status'] = 'removed'
        for _, root_name in roots:
            root = B / root_name
            for path in sorted((p for p in root.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                path.rmdir()
            root.rmdir()
        manifest.update(status='complete', files=len(deletions), removed_logical_bytes=sum(v['bytes'] for v in deletions))
        target.write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps({k: manifest[k] for k in ('status', 'files', 'removed_logical_bytes')}))


if __name__ == '__main__':
    main()
