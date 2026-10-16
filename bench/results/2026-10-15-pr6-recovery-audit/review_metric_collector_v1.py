#!/usr/bin/env python3
"""Finish Main inspection from closed host commands; no process/device execution.
Repository LICENSE. A source filename need not survive Release optimization;
the actual dependency manifest plus audit bytes are the relevant proof.
"""
import hashlib
import json
from pathlib import Path
import shlex

A = Path(__file__).resolve().parent
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')


def pin(p):
    p = Path(p)
    with p.open('rb') as stream:
        h = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(p), realpath=str(p.resolve()), bytes=p.stat().st_size, sha256=h)


def main():
    previous = A / 'own-metric-collector-inspection-raw-v2.json'
    assert pin(previous)['sha256'] == '1c32fb30f9bdff9ff1f652a28fc9df5aedcfd7005a8ef2a9d485f0e622b6b4ab'
    old = json.loads(previous.read_text())
    assert not old['passed'] and old['ownership_closed'] and not old['active']
    assert 'assert \'metric_query_audit\' in strings' in old['traceback']
    root = B / 'main-metric-query-collector-inspection-v2'
    for c in old['commands']:
        assert c['exit_code'] == 0 and c['observation_complete'] and c['session_empty'] and c['direct_child_reaped']
        assert not c.get('error') and not c['errors'] and not c['cleanup']
        for name, value in c['logs'].items():
            p = pin(root / 'owned' / name)
            assert p['sha256'] == value['sha256'] and p['bytes'] == value['bytes']
    candidate = A / 'own-metric-collector-build-candidate-v5.json'
    assert pin(candidate) == old['build_candidate']
    r = json.loads(candidate.read_text())
    build, source = Path(r['build_root']), Path(r['source_root'])
    assert r['passed'] and r['complete'] and r['ownership_closed'] and not r['active']
    assert pin(r['binary_candidate']['path']) == r['binary_candidate']
    compiles = json.loads(Path(r['compile_commands']['path']).read_text())
    actual_deps = set()
    objects = []
    deps = (root / 'owned/ninja-dependencies.stdout').read_text()
    commands = (root / 'owned/ninja-commands.stdout').read_text()
    strings = (root / 'owned/collector-strings.stdout').read_text()
    for i, item in enumerate(compiles):
        text = (root / 'owned' / ('all-headers-' + str(i) + '.stdout')).read_text().replace('\\\n', ' ')
        assert text.startswith('collector: ')
        actual_deps.update(str(Path(p).resolve(strict=True)) for p in shlex.split(text[len('collector: '):]))
        obj = build / item['output']
        assert item['file'] in commands and item['output'] in deps
        objects.append(dict(source=pin(item['file']), object=pin(obj), compile=item))
    expected = json.loads((A / 'own-metric-query-tiny-v3/patched-source-sha256.json').read_text())
    for rel, digest in expected.items():
        assert pin(source / rel)['sha256'] == digest
        assert str((source / rel).resolve()) in actual_deps and str(source / rel) in deps
    assert 'OWN_METRIC_V2 ' in strings
    rows = [line.split('\t') for line in (build / '.ninja_log').read_text().splitlines() if not line.startswith('#')]
    for obj in objects:
        matches = [row for row in rows if row[3] == obj['compile']['output']]
        assert len(matches) == 1 and int(matches[0][1]) > int(matches[0][0])
    capture = r['generated_capture']['files']
    assert len(capture) == 6 and not r['generated_capture']['errors']
    for name, item in capture.items():
        # Official CMake declares tracing.gen; its generator also writes the
        # consumed tracing.gen.types byproduct in that same invocation.
        target = 'tracing.gen' if name == 'tracing.gen.types' else name
        matches = [row for row in rows if row[3] == target]
        assert len(matches) == 1 and int(matches[0][2]) == item['original_mtime_ns']
        assert item['original_path'] in deps and item['observed_during_actual_build']
        assert pin(item['snapshot']['path']) == item['snapshot']
        assert pin(root / 'generated-include' / name)['sha256'] == item['snapshot']['sha256']
        if name.endswith('_commit_hash.h'):
            assert 'COMMIT_HASH ""' in Path(item['snapshot']['path']).read_text()
    import re
    dynamic = (root / 'owned/collector-dynamic.stdout').read_text()
    needed = re.findall(r'\(NEEDED\).*?\[(.*?)\]', dynamic)
    assert needed
    out = A / 'own-metric-collector-compile-proof-v1.json'
    assert not out.exists()
    result = dict(schema='strata-actual-collector-compile-proof-v1', reviewer='Main',
        passed=True, complete=True, GPU_executed=False, library_loaded=False,
        original_inspection=pin(previous), original_failure_status_preserved=True,
        original_failure_reason='Unnecessary expectation that an included header filename survive in the optimized binary strings; all eight owned host inspection commands succeeded.',
        build_candidate=pin(candidate), binary=r['binary_candidate'], compile_commands=r['compile_commands'],
        objects=objects, compiler_consumed_inputs=[pin(p) for p in sorted(actual_deps)],
        generated_inputs=[item['snapshot'] for item in capture.values()], generated_capture=capture,
        ninja_log=pin(build / '.ninja_log'), DT_NEEDED=needed,
        all_three_patch_headers_in_actual_ninja_dependencies=True, audit_marker_in_binary=True,
        generated_snapshot_mtimes_match_actual_ninja_outputs=True, fresh_object_compile_and_link_rows=rows,
        scope='Fresh complete official source plus exact audit patch; actual generated input snapshots before official cleanup; original compiler dependencies and closed successful compile/link. No library constructor or GPU work executed.',
        commit_metadata='Actual generated COMMIT_HASH is empty; independent verified baseline/patch inventories, not that macro, identify source.')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(receipt=pin(out), input_count=len(actual_deps), DT_NEEDED=needed)))


if __name__ == '__main__':
    main()
