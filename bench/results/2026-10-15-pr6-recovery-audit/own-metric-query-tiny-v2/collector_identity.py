"""Additive Main collector identity gates; repository LICENSE.
No building, enrollment, source mutation, runtime import or process launching.
The original inventory remains the reference; replacement identity has its own key.
"""
import hashlib
import json
import os
from pathlib import Path

PTI_COMMIT = '6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033'
BASELINE_COLLECTOR_SHA = '6ee0130e2b03918a5e4bb5d310dacb94e5ba06bf67c21a23d3134dc406aa9122'
D = Path(__file__).resolve().parent

def need(value, message):
    if not value:
        raise ValueError('HOLD_COLLECTOR_IDENTITY: ' + message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sealed_replacement(seal, manifest, base, owner, package_sha):
    """Accept only externally authored Main source/build/dependency evidence.

    Pins here are supplied by an independently reviewed seal. There is no helper
    that discovers a binary or creates a pin/approval from the current filesystem.
    Full receipts are durable inputs; their consumers are Main and this gate.
    """
    replacement = seal['replacement_collector']
    need(set(replacement) == {'binary','source_receipt','build_receipt'}, 'exact replacement identity schema')
    original = manifest['collector']
    binary = replacement['binary']
    need(binary['path'] != original['path'] and binary['realpath'] != original['realpath'] and
         binary['sha256'] != original['sha256'] and Path(binary['path']).name == 'libunitrace_tool.so',
         'separate named collector, no original allowlist replacement')
    for pin in replacement.values(): base.check_pin(pin['path'], pin)
    source = json.loads(Path(replacement['source_receipt']['path']).read_text())
    need(source['schema'] == 'strata-pti-query-source-seal-v2' and source['status'] == 'MAIN_REVIEWED_SEALED' and
         source['official_pti_commit'] == PTI_COMMIT and source['clean_baseline'] is True and
         source['reviewed_package_sha256'] == package_sha, 'actual Main baseline/patch source seal')
    # Baseline inventory: complete PTI source tree (and submodules) used by Main,
    # exact upstream commit receipt and full manifest pinned, never Git run here.
    for key in ('baseline_inventory','patched_inventory','toolchain_inputs','original_build_receipt','patch'):
        base.check_pin(source[key]['path'], source[key])
    need(source['patch']['realpath'] == str((D / 'pti-query-status.patch').resolve()) and
         source['patch']['sha256'] == sha(D / 'pti-query-status.patch'), 'exact reviewed patch bytes')
    original_build_path = str(Path(original['path']).parent.parent / 'record.json')
    need(source['original_build_receipt'] == manifest['file_pins'][original_build_path],
         'exact original reference build receipt remains pinned')
    original_build = json.loads(Path(original_build_path).read_text())
    need(original_build['passed'] is True and original_build['complete'] is True and
         original_build['active'] is False and original_build['source_revision'] == PTI_COMMIT,
         'original successful official PTI baseline build')
    before = json.loads(Path(source['baseline_inventory']['path']).read_text())
    after = json.loads(Path(source['patched_inventory']['path']).read_text())
    expected = json.loads((D / 'patched-source-sha256.json').read_text())
    need(before['official_pti_commit'] == after['official_pti_commit'] == PTI_COMMIT and
         before['complete'] is True and after['complete'] is True and
         before['files']['tools/unitrace/src/levelzero/ze_collector.h'] == BASELINE_COLLECTOR_SHA and
         source['patch_apply_fuzz'] == 0 and source['patch_apply_offset'] == 0,
         'Main full immutable source and exact no-fuzz/no-offset application receipt')
    need(set(after['files']) == set(before['files']) | set(expected) and
         after['files'] == {**before['files'], **expected}, 'only declared patch additions/changes')
    inputs = json.loads(Path(source['toolchain_inputs']['path']).read_text())
    need(inputs['complete'] is True and inputs['consumer'] == 'Main patched unitrace build' and
         inputs['files'] and inputs['dependencies'] and inputs['generated_inputs'] and inputs['recipe'],
         'Main actual toolchain/generated/dependency/recipe inventory')
    for pin in [*inputs['files'], *inputs['dependencies'], *inputs['generated_inputs']]:
        base.check_pin(pin['path'], pin)
    build = json.loads(Path(replacement['build_receipt']['path']).read_text())
    need(build['schema'] == 'strata-pti-query-build-v2' and build['mode'] == 'build' and
         build['passed'] is True and build['complete'] is True and build['active'] is False and
         build['ownership_closed'] is True and build['boot_id'] == seal['boot_id'] and
         build['reviewed_package_sha256'] == package_sha and build['binary_candidate'] == binary and
         build['source_receipt'] == replacement['source_receipt'] and
         build['toolchain_inputs'] == source['toolchain_inputs'] and build['recipe'] == inputs['recipe'],
         'Main actual separately closed patched collector build receipt')
    commands = build['commands']
    need(commands and all(owner.closed(c) and c['exit_code'] == 0 for c in commands), 'owned normal build commands')
    need([c['label'] for c in commands] == ['collector-configure','collector-build'] and
         [c['argv'] for c in commands] == inputs['recipe'] and all(c.get('environment') for c in commands),
         'actual configure/build argv and immutable child environments')
    # Preserve actual compile commands, source dependency file and build logs;
    # Main must prove no compiler cache/stale object substituted an old library.
    for key in ('compile_commands','build_log','dependency_inventory','binary_inspection'):
        base.check_pin(build[key]['path'], build[key])
    inspection = json.loads(Path(build['binary_inspection']['path']).read_text())
    need(inspection['binary'] == binary and inspection['reviewed'] is True and
         inspection['source_receipt'] == replacement['source_receipt'] and
         inspection['fresh_objects'] is True and inspection['actual_compile_proof'] is True,
         'actual Main binary/source/fresh compilation inspection')
    base.absence(commands)
    review = seal['additional_dependency_review']
    report = json.loads(Path(review['report']['path']).read_text())
    need(report['replacement_collector'] == binary and report['replacement_build_receipt'] == replacement['build_receipt'] and
         report['original_reference_collector'] == original and report['all_actual_dependencies_reviewed'] is True,
         'separate additional dependencies tied to actual replacement build')
    need(binary['realpath'] not in manifest['elf_pins'] and binary['path'] not in manifest['file_pins'] and
         binary['realpath'] not in review['elf_pins'] and binary['path'] not in review['file_pins'],
         'replacement has an explicit identity slot, never an allowlist overwrite')
    return replacement, [source[k] for k in ('baseline_inventory','patched_inventory','toolchain_inputs','original_build_receipt','patch')] + [
        *replacement.values(), build['compile_commands'],build['build_log'],build['dependency_inventory'],build['binary_inspection'],
        *inputs['files'], *inputs['dependencies'], *inputs['generated_inputs']]


def native_snapshot(base, pid, expected, env, pins, expected_sid, replacement, original):
    """Focused adaptation of pinned base native_snapshot b81a06f1, explicit identity.

    Same ownership/env/map device/inode/hash/source gates. The base helper's
    hard-coded original LIB must remain unchanged; we do not monkey-patch it.
    """
    before = base.identity(pid)
    need(before and before['sid'] == expected_sid, 'actual native session')
    proc = Path('/proc', str(pid)); raw = (proc / 'maps').read_text()
    required = ('libunitrace_tool.so','libxptifw.so','libsycl.so','libze_loader.so',
                'libze_tracing_layer.so','libze_intel_gpu.so','libur_adapter_level_zero')
    if not all(n in raw for n in required): return None
    args = [a.decode() for a in (proc / 'cmdline').read_bytes().split(b'\0') if a]
    effective = dict(a.decode().split('=',1) for a in (proc / 'environ').read_bytes().split(b'\0') if a)
    need(effective == env and args and str(Path(args[0]).resolve()) == expected['realpath'], 'actual argv/env')
    need(str((proc / 'exe').resolve(strict=True)) == expected['realpath'], 'actual executable path')
    with (proc / 'exe').open('rb') as stream:
        need(hashlib.file_digest(stream,'sha256').hexdigest() == expected['sha256'], 'actual open executable hash')
    mappings, seen = [], set()
    for line in raw.splitlines():
        fields = line.split(None,5)
        if len(fields) != 6 or not fields[5].startswith('/') or 'x' not in fields[1]: continue
        need(not fields[5].endswith(' (deleted)'), 'deleted mapped ELF')
        path = base.safe_path(fields[5]); name = str(path)
        if name in seen: continue
        seen.add(name)
        need(name in pins, 'unsealed actual loaded dependency: '+name)
        st = path.stat(); major,minor = [int(v,16) for v in fields[3].split(':')]
        need(st.st_ino == int(fields[4]) and os.major(st.st_dev) == major and os.minor(st.st_dev) == minor,
             'mapped ELF dev/inode')
        base.check_pin(path,pins[name]); mappings.append(dict(mapping=line,**base.artifact(path)))
    need(replacement['realpath'] in seen and original['realpath'] not in seen, 'actual replacement only; original reference stays pinned')
    after = base.identity(pid)
    need(after and after['start_ticks'] == before['start_ticks'], 'native PID/start stable')
    signature = '\n'.join(l for l in raw.splitlines() if len(l.split(None,5)) == 6 and
        'x' in l.split(None,5)[1] and l.split(None,5)[5].startswith('/'))
    return dict(native=before,argv=args,executable=expected,effective_environment=effective,executable_mappings=mappings,
                maps_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                executable_maps_sha256=hashlib.sha256(signature.encode()).hexdigest(),captured_utc=base.utc(),
                collector_identity='separately sealed replacement; original reference unchanged',
                anonymous_executable_mapping_scope='File-backed executable maps at capture; no JIT ISA/occupancy claim')
