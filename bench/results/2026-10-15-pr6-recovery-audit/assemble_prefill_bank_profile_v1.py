#!/usr/bin/env python3
"""Assemble reviewed own-source proposals in memory; never edit product source."""
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess

A = Path(__file__).resolve().parent
P = A.parents[2]
BASE = '9177ad4fb7cc1587255db7235be474aca3eb4e2a'
CANONICAL = A/'prefill-canonical-dense-bank-v3'
PROFILE = A/'layermajor-coarse-profile-v2'
OUT = A/'prefill-bank-profile-main-v1'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def require(condition, why):
    if not condition:
        raise RuntimeError(why)


def original(name):
    result = subprocess.run(['git', 'show', BASE+':'+name], cwd=P,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        require(not (P/name).exists(), 'missing base but present live path '+name)
        return None
    return result.stdout.decode()


def parse(diff):
    lines = diff.splitlines(keepends=True)
    position = 0
    while position < len(lines):
        require(lines[position].startswith('--- '), 'expected file header')
        position += 1
        require(lines[position].startswith('+++ b/'), 'bounded candidate path')
        name = lines[position][6:].rstrip('\n')
        require('..' not in Path(name).parts and not Path(name).is_absolute(), 'safe source path')
        position += 1
        hunks = []
        while position < len(lines) and not lines[position].startswith('--- '):
            match = re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*\n', lines[position])
            require(match is not None, 'hunk header '+name)
            position += 1
            before, after = [], []
            while position < len(lines) and not lines[position].startswith(('@@ ', '--- ')):
                line = lines[position]
                require(line[:1] in (' ', '-', '+'), 'hunk line kind')
                if line[0] in (' ', '-'):
                    before.append(line[1:])
                if line[0] in (' ', '+'):
                    after.append(line[1:])
                position += 1
            require(len(before) == int(match[2] if match[2] is not None else 1) and
                    len(after) == int(match[4] if match[4] is not None else 1), 'hunk extent '+name)
            hunks.append((''.join(before), ''.join(after)))
        yield name, hunks


def main():
    require(not OUT.exists(), 'fresh source-only assembly')
    manifests = [json.loads((CANONICAL/'proposal-manifest.json').read_text()),
                 json.loads((PROFILE/'pins.json').read_text())]
    require(digest((CANONICAL/'source.diff').read_bytes()) == manifests[0]['source_diff_sha256'], 'canonical pin')
    require(digest((PROFILE/'source.diff').read_bytes()) == manifests[1]['source_diff_sha256'], 'profile pin')
    base, candidate, ledger = {}, {}, []
    for source in (CANONICAL, PROFILE):
        for name, hunks in parse((source/'source.diff').read_text()):
            if name not in base:
                base[name] = original(name)
                candidate[name] = base[name] or ''
            for number, (before, after) in enumerate(hunks):
                manual = False
                callback_context = False
                if source == PROFILE and name == 'src/program/generate.cpp' and 'host_profile.entry_end();' in after:
                    # Canonical entry must finish BEFORE the host entry stamp.
                    anchor = '        if(!phase.dense->enter_layer_bank({true},err)) return false;\n'
                    block = ('        if(phase.canonical) {\n'
                             '            phase.canonical->enter();\n'
                             '            if(!phase.canonical->bind_ple(ple,err)) fatal("persistent canonical PLE rebind failed");\n'
                             '        }\n')
                    require(before.count(anchor) == after.count(anchor) == 1, 'single explicit entry seam')
                    before = before.replace(anchor, anchor+block)
                    after = after.replace(anchor, anchor+block)
                    manual = True
                if source == PROFILE and name == 'src/program/generate.cpp' and 'host_profile.run_begin();' in after:
                    anchor = '            overlay=&phase.dense->layer_view(); return true;\n'
                    block = ('            if(phase.canonical) {\n'
                             '                if(!phase.canonical->activate(int(layer),stream,phase.dense->layer_view(),e)) return false;\n'
                             '                overlay=&phase.canonical->view();\n'
                             '            } else overlay=&phase.dense->layer_view();\n'
                             '            return true;\n')
                    require(before.count(anchor) == after.count(anchor) == 1, 'single callback context seam')
                    before = before.replace(anchor, block)
                    after = after.replace(anchor, block)
                    callback_context = True
                text = candidate[name]
                if before:
                    require(text.count(before) == 1, 'unique exact hunk '+name+':'+str(number))
                    candidate[name] = text.replace(before, after, 1)
                else:
                    require(not text and len(hunks) == 1, 'new file must be empty')
                    candidate[name] = after
                ledger.append(dict(source=source.name, path=name, hunk=number,
                                   manual_canonical_entry_context=manual,
                                   manual_canonical_callback_context=callback_context,
                                   before_sha256=digest(before.encode()), after_sha256=digest(after.encode())))
    for name, data in manifests[0]['files'].items():
        expected = data['base_sha256']
        require((digest(base[name].encode()) if base[name] is not None else None) == expected,
                'canonical base matches immutable9177 '+name)
    for name, expected in manifests[1]['base_source_sha256'].items():
        require(digest(base[name].encode()) == expected, 'profile base matches '+name)
    additions = {
        'tests/sycl/canonical_dense_bank.cpp': (A/'canonical_dense_bank_component_fixture_v4.cpp').read_text(),
        'tests/cpu/canonical_bank_policy.cpp': (CANONICAL/'host_fixture.cpp').read_text(),
    }
    for name, text in additions.items():
        base[name] = original(name)
        require(base[name] is None, 'new Main fixture '+name)
        candidate[name] = text
    name = 'cmake/sycl_backend.cmake'
    base[name] = original(name)
    anchor = '  add_executable(sycl_decode_attention_test tests/sycl/decode_attention.cpp)\n'
    require(base[name].count(anchor) == 1, 'manual target seam')
    targets = '''  # Main-only manual canonical ownership gates; no automatic GPU execution.
  add_executable(canonical_bank_policy_test EXCLUDE_FROM_ALL tests/cpu/canonical_bank_policy.cpp)
  target_include_directories(canonical_bank_policy_test PRIVATE "${CMAKE_SOURCE_DIR}/include")
  add_executable(sycl_canonical_dense_bank_test EXCLUDE_FROM_ALL tests/sycl/canonical_dense_bank.cpp)
  target_link_libraries(sycl_canonical_dense_bank_test PRIVATE strata_engine)
  target_compile_options(sycl_canonical_dense_bank_test PRIVATE -fno-fast-math -ffp-contract=off)
  add_executable(sycl_canonical_dense_bank_fault_test EXCLUDE_FROM_ALL tests/sycl/canonical_dense_bank.cpp)
  target_link_libraries(sycl_canonical_dense_bank_fault_test PRIVATE strata_engine)
  target_compile_definitions(sycl_canonical_dense_bank_fault_test PRIVATE STRATA_CANONICAL_ALLOC_FAULT=1)
  target_compile_options(sycl_canonical_dense_bank_fault_test PRIVATE -fno-fast-math -ffp-contract=off)
  add_executable(sycl_canonical_native_reserve_fault_test EXCLUDE_FROM_ALL
    tests/sycl/canonical_dense_bank.cpp tests/sycl/native_dense_bank_fault_object.cpp)
  target_link_libraries(sycl_canonical_native_reserve_fault_test PRIVATE strata_engine)
  target_compile_definitions(sycl_canonical_native_reserve_fault_test PRIVATE STRATA_CANONICAL_NATIVE_ALLOC_FAULT=1)
  target_compile_options(sycl_canonical_native_reserve_fault_test PRIVATE -fno-fast-math -ffp-contract=off)
'''
    candidate[name] = base[name].replace(anchor, targets+anchor)
    require(sum(x['manual_canonical_entry_context'] for x in ledger) == 1, 'one explicit manual context merge')
    require(sum(x['manual_canonical_callback_context'] for x in ledger) == 1, 'one explicit callback context merge')
    source = candidate['src/program/generate.cpp']
    expected_order = ('phase.canonical->enter();', 'persistent canonical PLE rebind failed',
                      'host_profile.entry_end();', 'observe("bank");')
    positions = [source.index(s) for s in expected_order]
    require(positions == sorted(positions), 'entry timer follows both owners')
    forbidden = ('DecodeRouteLedger', 'DecodeServiceBudget', 'CpuGuReuseObserver',
                 'MistVVK', 'decode_service_budget.hpp', 'decode_route_ledger.hpp')
    require(not any(word in text for text in candidate.values() for word in forbidden), 'retired derived symbols absent')
    patches = []
    for name in sorted(candidate):
        patches.extend(difflib.unified_diff((base[name] or '').splitlines(keepends=True),
                                           candidate[name].splitlines(keepends=True),
                                           fromfile='a/'+name if base[name] is not None else '/dev/null',
                                           tofile='b/'+name))
    patch = ''.join(patches).encode()
    pins = dict(base_commit=BASE, source_only=True, applied=False, compiled=False, tested=False,
                source_diff_sha256=digest(patch), ledger=ledger,
                references={str(p.relative_to(A)):digest(p.read_bytes()) for p in
                            (CANONICAL/'source.diff', PROFILE/'source.diff',
                             A/'canonical_dense_bank_component_fixture_v4.cpp', CANONICAL/'host_fixture.cpp')},
                files={name:dict(base_sha256=digest(base[name].encode()) if base[name] is not None else None,
                                 proposed_sha256=digest(text.encode())) for name,text in candidate.items()})
    OUT.mkdir()
    (OUT/'source.diff').write_bytes(patch)
    (OUT/'pins.json').write_text(json.dumps(pins,indent=2)+'\n')
    print(json.dumps(dict(source_only=True, files=len(candidate), hunks=len(ledger),
                          source_diff_sha256=digest(patch), product_modified=False)))


if __name__ == '__main__':
    main()
