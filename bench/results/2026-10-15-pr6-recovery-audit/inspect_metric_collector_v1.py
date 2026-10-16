#!/usr/bin/env python3
"""Main host-only inspection of the actually built collector; no library loading.
Repository LICENSE. Source and build seals are a separate reviewed decision.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys

sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def pin(p):
    p = Path(p)
    with p.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(p), realpath=str(p.resolve()), bytes=p.stat().st_size, sha256=digest)


def main():
    candidate = A / 'own-metric-collector-build-candidate-v4.json'
    assert pin(candidate)['sha256'] == '51da7297780813e89ab94abd11d3cb10ba99cc9d700d791bc5e0ea0b23085341'
    r = json.loads(candidate.read_text())
    assert r['passed'] and r['complete'] and r['ownership_closed'] and not r['active']
    build, source = Path(r['build_root']), Path(r['source_root'])
    assert pin(r['binary_candidate']['path']) == r['binary_candidate']
    root = B / 'main-metric-query-collector-inspection-v1'
    out = A / 'own-metric-collector-inspection-raw-v1.json'
    assert not root.exists() and not out.exists()
    owner_path = A.parents[2] / 'bench/results/2026-10-14-niko-main-service-port/direct_owner.py'
    assert pin(owner_path)['sha256'] == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f'
    spec = importlib.util.spec_from_file_location('metric_inspection_owner', owner_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root.mkdir(mode=0o700)
        record = dict(schema='strata-host-collector-inspection-v1', passed=False, complete=False,
            active=False, GPU_executed=False, library_loaded=False, commands=[],
            build_candidate=pin(candidate), binary=r['binary_candidate'], findings={},
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        owner = None

        def persist():
            record['active'] = bool(owner and owner.active is not None)
            (root / 'record.json').write_text(json.dumps(record, indent=2) + '\n')

        owner = helper.Owner(root / 'owned', persist)
        record['commands'] = owner.commands

        def run(label, argv):
            c, so, se = owner.run(label, argv, r['environment'], build, wall=45, cpu=90,
                rss_cap=2 << 30, text_cap=8 << 20, file_cap=8 << 20, total_cap=64 << 20)
            assert helper.closed(c) and c['exit_code'] == 0
            return so.read_text(), se.read_text()

        try:
            for c in r['commands']:
                assert helper.closed(c) and c['exit_code'] == 0
            deps, _ = run('ninja-dependencies', ['/usr/bin/ninja', '-C', str(build), '-t', 'deps'])
            commands, _ = run('ninja-commands', ['/usr/bin/ninja', '-C', str(build), '-t', 'commands'])
            dynamic, _ = run('collector-dynamic', ['/usr/bin/readelf', '-d', r['binary_candidate']['path']])
            strings, _ = run('collector-strings', ['/usr/bin/strings', '-a', r['binary_candidate']['path']])
            compiles = json.loads(Path(r['compile_commands']['path']).read_text())
            actual_deps = set()
            objects = []
            for i, item in enumerate(compiles):
                argv = shlex.split(item['command'])
                assert argv[-2] == '-c' and argv[-1] == item['file']
                output_at = argv.index('-o')
                stripped = argv[:output_at] + argv[output_at+2:-2] + [argv[-1]]
                dep_text, _ = run('all-headers-' + str(i), stripped + ['-M', '-MT', 'collector'])
                flat = dep_text.replace('\\\n', ' ')
                assert flat.startswith('collector: ')
                paths = shlex.split(flat[len('collector: '):])
                assert paths and len(paths) < 5000
                actual_deps.update(str(Path(p).resolve(strict=True)) for p in paths)
                obj = build / item['output']
                assert obj.exists() and obj.stat().st_size > 0
                assert item['file'] in commands and item['output'] in deps
                objects.append(dict(source=pin(item['file']), object=pin(obj), compile=item))
            expected = json.loads((A / 'own-metric-query-tiny-v3/patched-source-sha256.json').read_text())
            for rel, digest in expected.items():
                assert pin(source / rel)['sha256'] == digest
                assert str((source / rel).resolve()) in actual_deps
                assert str(source / rel) in deps
            needed = re.findall(r'\(NEEDED\).*?\[(.*?)\]', dynamic)
            assert needed
            assert 'OWN_METRIC_V2 ' in strings
            assert 'metric_query_audit' in strings
            log = build / '.ninja_log'
            ninja_rows = [line.split('\t') for line in log.read_text().splitlines() if not line.startswith('#')]
            for obj in objects:
                matches = [row for row in ninja_rows if row[3] == obj['compile']['output']]
                assert len(matches) == 1 and int(matches[0][1]) > int(matches[0][0])
                assert obj['object']['bytes'] > 0
            assert sum(row[3] == 'libunitrace_tool.so' for row in ninja_rows) == 1
            generated = [pin(p) for p in sorted(build.iterdir()) if p.is_file() and
                (p.suffix == '.gen' or p.name.endswith('_commit_hash.h'))]
            assert generated and all('COMMIT_HASH ""' in Path(p['path']).read_text()
                for p in generated if p['path'].endswith('_commit_hash.h'))
            record['findings'] = dict(compile_commands=compiles, objects=objects, DT_NEEDED=needed,
                compiler_consumed_inputs=[pin(p) for p in sorted(actual_deps)],
                generated_inputs=generated, ninja_log=pin(log),
                compile_headers_include_all_three_patch_files=True,
                patch_marker_present=True, fresh_object_rows=ninja_rows,
                commit_metadata='Actual generated COMMIT_HASH is empty because the private verified source is not a Git checkout. Independent complete baseline and patch inventories identify source.',
                no_compiler_cache=True, executable_launched=False)
            record.update(passed=True, complete=True)
        except BaseException as error:
            record['error'] = repr(error)
        finally:
            record['ownership_closed'] = owner.active is None and all(helper.closed(c) for c in owner.commands)
            for c in owner.commands:
                for old in c['owners'] + [c['direct_identity']]:
                    now = helper.identity(old['pid'])
                    assert now is None or now['start_ticks'] != old['start_ticks']
            persist()
            out.write_bytes((root / 'record.json').read_bytes())
        print(json.dumps(dict(receipt=pin(out), passed=record['passed'], error=record.get('error'),
            headers=len(record.get('findings', {}).get('compiler_consumed_inputs', [])),
            DT_NEEDED=record.get('findings', {}).get('DT_NEEDED', []))))
        return 0 if record['passed'] and record['ownership_closed'] else 1


if __name__ == '__main__':
    sys.exit(main())
