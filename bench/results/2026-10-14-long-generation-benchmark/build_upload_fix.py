#!/usr/bin/env python3
"""Main-owned serial incremental build of the immutable MTP host upload fix."""
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent
BUILD = B / 'main-primary-native-build-root-v1/build'
O = B / 'main-primary-draft-upload-fix-build-root-v1'
OWNER = P / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
SEED = B / 'main-primary-routed-fp16-census-build-root-v1/record.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    seed = json.loads(SEED.read_text())
    assert seed['passed'] and seed['complete'] and not seed['active']
    env = seed['environment']
    assert not any('TOKEN' in k or 'API_KEY' in k for k in env)
    spec = importlib.util.spec_from_file_location('owner', OWNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not O.exists()
        closed = json.loads((D / 'suite-v4-summary.json').read_text())
        assert closed['owners_reaped'] and closed['session_empty'] and not closed['original_status']['active']
        assert sha(BUILD / 'strata') == seed['binaries']['strata']
        assert json.loads((D / 'draft-upload-cpu-validation-v1.json').read_text())['passed']
        O.mkdir()
        paths = set(seed['source_pins']) | {'include/strata/prefill/draft_uploads.hpp', 'tests/cpu/draft_uploads.cpp'}
        record = dict(active=True, passed=False, complete=False, GPU_executed=False, commands=[],
            environment=env, source_revision=subprocess.check_output(['git', '-C', str(P), 'rev-parse', 'HEAD'], text=True).strip(),
            source_pins={n: sha(P / n) for n in sorted(paths)}, old_binary_sha256=sha(BUILD / 'strata'),
            recipe_sha256=sha(Path(__file__)), seed_receipt_sha256=sha(SEED), build_path=str(BUILD),
            scope='Immutable MTP host token/bounds sources; successful path retains original final wait; SYCL early-exit drain/fail-stop guard. No device math/compiler/queue/graph/affinity change.',
            prior_build_path_superseded=True, performance_qualified=False, quality_qualified=False,
            physical_262144_qualified=False, build_byte_budget=4 << 30)
        def persist():
            (O / 'record.json').write_text(json.dumps(record, indent=2) + '\n')
        owner = module.Owner(O / 'owned', persist)
        record['commands'] = owner.commands
        persist()
        def run(label, argv, wall):
            entry, out, err = owner.run(label, argv, env, P, wall=wall,
                cpu=wall * len(os.sched_getaffinity(0)), text_cap=64 << 20,
                rss_cap=48 << 30, total_cap=1 << 30, file_cap=256 << 20)
            assert module.closed(entry) and entry['exit_code'] == 0, (label, err.read_text()[-3000:], out.read_text()[-3000:])
            print(json.dumps(dict(passed=label)), flush=True)
        try:
            run('build-upload-fix', ['cmake', '--build', str(BUILD), '--target', 'strata',
                'draft_uploads_host_test', '--parallel', '2'], 1500)
            run('configured-host-upload-fixture', [str(BUILD / 'draft_uploads_host_test')], 30)
            commands = json.loads((BUILD / 'compile_commands.json').read_text())
            selected = [c for c in commands if c['file'] == str(P / 'src/prefill/prefill.cpp')]
            assert len(selected) == 1 and '-fsycl-id-queries-range=size_t' in selected[0]['command']
            record['compile_flag_proof'] = dict(prefill=selected[0],
                compile_commands_sha256=sha(BUILD / 'compile_commands.json'))
            record['binaries'] = {n: sha(BUILD / n) for n in ['strata', 'draft_uploads_host_test']}
            assert all(sha(P / n) == h for n, h in record['source_pins'].items())
            record['build_logical_bytes'] = sum(p.stat().st_size for p in BUILD.rglob('*') if p.is_file())
            assert record['build_logical_bytes'] <= record['build_byte_budget']
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            record['active'] = False
            persist()
        (D / 'draft-upload-build-v1.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(dict(passed=True, GPU_executed=False, binaries=record['binaries'])), flush=True)


if __name__ == '__main__':
    main()
