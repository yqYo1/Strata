#!/usr/bin/env python3
"""MAIN ONLY after full source/inventory review; offline pin preparation, no tool execution.
Repository LICENSE. Never run automatically from controller; never refresh a pin
from live bytes in place of an original qualification. No symlink enrollment.
"""
import argparse
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
A = D.parent
P = A.parents[2]
BASE_SHA = '1e9fa3cb57e76405deda315dbdc91b8d43cbf0a98c1d62172661fa5d31da47c0'
BUILD_SHA = '3fca64de3664a74ac5a376e4e40cb8a3e2910113f29a4597a564c2f903381327'
SOURCE = '63326f98c36656cf3a3e1adcc54c3c25a02aa6ee'
ENGINE = '6334491b623ead7b83c7ff5aaee84583294b3ecd4b12c6d06b77d1fe7e7a691d'
COMPONENTS = {'own-prompt-softmax-components-check-v1.json':'a99643eed457c1bf141e770da44b548814c15e718c5694f41d5c2921568072ae',
              'own-prompt-softmax-components-smoke-v1.json':'f59bc11c027046d36defe2fba2ab496b1165671cca9f2eb79ad84cdab5093e4e'}


def require(v, m):
    if not v:
        raise RuntimeError(m)


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def safe(p):
    q = Path(p).resolve(strict=True)
    require('prefill-gemm-only-contract-root-v1' not in q.parts, 'protected artifact forbidden')
    require('XeStrata' not in q.parts or '/.tools/intel-llvm-v711/install/' in str(q), 'excluded source forbidden')
    return q


def check(p, pin):
    q = safe(p)
    require(str(q) == pin['realpath'] and q.stat().st_size == pin['bytes'] and sha(q) == pin['sha256'], 'original qualification drift: ' + str(p))


def pin(p):
    q = safe(p)
    return dict(path=str(q), realpath=str(q), bytes=q.stat().st_size, sha256=sha(q))


def read(p, digest):
    require(sha(p) == digest, 'sealed input identity: ' + str(p))
    return json.loads(Path(p).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviewed-package-sha256', required=True, help='External SHA256 of reviewed SHA256SUMS; no resampling source after review')
    parser.add_argument('--reviewed-inventory-sha256', required=True)
    parser.add_argument('--reviewed-source-sha256', required=True, help='SHA256 of source-provenance.json after full Main review')
    args = parser.parse_args()
    require(D == P/'bench/results/2026-10-15-pr6-recovery-audit/own-vllm-resource-timeline-v1', 'canonical source proposal')
    require(not (D/'manifest.json').exists(), 'no overwrite/automatic reprepare')
    require(sha(D/'SHA256SUMS') == args.reviewed_package_sha256, 'reviewed complete proposal snapshot')
    sealed = set()
    for line in (D/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        require(len(digest) == 64 and all(c in '0123456789abcdef' for c in digest) and
                name not in sealed and not Path(name).is_absolute() and '..' not in Path(name).parts,
                'strict source snapshot row')
        require(sha(D/name) == digest, 'reviewed proposal source drift: ' + name)
        sealed.add(name)
    actual = {str(p.relative_to(D)) for p in D.rglob('*') if p.is_file() and p.name != 'SHA256SUMS' and '__pycache__' not in p.parts}
    require(sealed == actual, 'complete reviewed source snapshot; no unreviewed files')
    prov = read(D/'source-provenance.json', args.reviewed_source_sha256)
    for v in prov['base_inputs'].values():
        check(v['path'], v)
    for v in prov['own_sources_read']:
        check(v['path'], v)
    for v in prov['official_inspected_sources'].values():
        check(v['path'], v)
    base = read(A/'own-vllm-unitrace-direct-v6/manifest.json', BASE_SHA)
    build = read(A/'own-prompt-softmax-build-v3.json', BUILD_SHA)
    inventory = read(D/'proposed_elf_inventory.json', args.reviewed_inventory_sha256)
    require(build['passed'] and build['complete'] and not build['active'] and build['ownership_closed'] and
            build['source_commit'] == SOURCE and build['binary_sha256'] == ENGINE and
            build['products']['strata']['sha256'] == ENGINE, 'original qualified source/build/product')
    require(inventory['base_manifest_sha256'] == BASE_SHA and inventory['build_receipt_sha256'] == BUILD_SHA and
            inventory['products'] == build['products'], 'reviewed inventory original product binding')
    expected_elf = dict(base['elf_pins'])
    expected_elf.update({v['realpath']:v for v in build['products'].values()})
    require(inventory['elf_pins'] == expected_elf, 'no newly sampled or relaxed ELF allowlist')
    file_pins = dict(base['file_pins'])
    # Preserve original input/build receipts. A conflicting old pin is an error,
    # never a request to hash whatever happens to be at that path now.
    for name, v in build['pins'].items():
        require(name not in file_pins or file_pins[name]['sha256'] == v['sha256'], 'conflicting original pin: ' + name)
        file_pins[name] = v
    for v in list(build['products'].values()) + list(expected_elf.values()) + list(prov['base_inputs'].values()):
        name = v.get('path', v['realpath'])
        require(name not in file_pins or file_pins[name]['sha256'] == v['sha256'], 'conflicting product/ELF pin')
        file_pins[name] = v
    components = []
    for name, digest in COMPONENTS.items():
        value = read(A/name, digest)
        require(value['passed'] and value['complete'] and not value['active'] and value['ownership_closed'] and
                value['source_commit'] == SOURCE and value['source_pins'] == build['source_pins'] and
                value['products'] == build['products'], 'original component qualification')
        components.append(pin(A/name)); file_pins[str((A/name).resolve())] = pin(A/name)
    diagnostic = json.loads((A/'own-prompt-softmax-long-diagnostic-0-r0-v1.json').read_text())
    require(diagnostic['build_receipt_sha256'] == BUILD_SHA and diagnostic['source_pins'] == build['source_pins'] and
            diagnostic['binary_sha256'] == ENGINE and diagnostic['passed'] and diagnostic['complete'], 'known safe actual model case')
    require(inventory['runtime_files'] == diagnostic['runtime_files'], 'original current runtime inventory')
    for v in diagnostic['runtime_files']:
        require(v['realpath'] in expected_elf and expected_elf[v['realpath']]['sha256'] == v['sha256'], 'runtime must match original v6 ELF inventory')
    for name, digest in build['source_pins'].items():
        require(sha(P/name) == digest, 'qualified source freeze: ' + name)
    for name, v in file_pins.items():
        check(name, v)
    for name, v in base['model_stat_pins'].items():
        st = safe(name).stat()
        require(st.st_size == v['bytes'] and st.st_mtime_ns == v['mtime_ns'], 'original model stat identity (not full shard SHA)')
    old_profile = json.loads((A/'own-vllm-profile-bf16-r0-v1.json').read_text())
    # Current safe model environment, plus existing coarse/QSA/main-thread CPU
    # instrumentation needed for cross-checks. All fixed before native exec.
    env = dict(diagnostic['environment'])
    for k in ('STRATA_MEASUREMENT_ENGINE_START','STRATA_MEASUREMENT_ENGINE_EXIT'):
        env.pop(k, None)
    env.update(STRATA_SYCL_PROMPT_SOFTMAX_SUBGROUP='0', STRATA_PREFILL_LAYER_MAJOR_PROFILE='1',
               STRATA_PREFILL_LAYER_MAJOR_QSA_PROFILE='1', STRATA_PREFILL_CPU_WAIT_PROFILE='1')
    expected_args = list(old_profile['config']['args'])
    actual_args = list(diagnostic['config']['args'])
    for xs in (actual_args, expected_args):
        xs[xs.index('--expert-profile')+1] = '<owned-private-profile-copy>'
    require(actual_args == expected_args and old_profile['benchmark_options'] == diagnostic['benchmark_options'], 'current safe exact CLI and request recipe')
    package = {str(p.relative_to(D)):pin(p) for p in sorted(D.rglob('*')) if p.is_file() and p.name != 'manifest.json' and '__pycache__' not in p.parts}
    manifest = dict(schema='strata-resource-timeline-source-v1',status='PREPARED_NOT_EXECUTED',source_commit=SOURCE,
                    source_pins=build['source_pins'],file_pins=file_pins,model_stat_pins=base['model_stat_pins'],
                    engine=build['products']['strata'],tiny=build['products']['sycl_prompt_native16_test'],
                    collector=base['collector'],products=build['products'],elf_pins=expected_elf,
                    component_receipts=components,package_files=package,
                    profile=dict(environment=env,config=old_profile['config'],benchmark_options=old_profile['benchmark_options']),
                    base_manifest_sha256=BASE_SHA,qualified_build_sha256=BUILD_SHA,
                    reviewed_package_sha256=args.reviewed_package_sha256,
                    reviewed_source_sha256=args.reviewed_source_sha256,reviewed_inventory_sha256=args.reviewed_inventory_sha256,
                    official_source_commit=base['official_source_commit'],official_source_pins=base['official_source_pins'],
                    original_receipts_preserved=True,execution=dict(offline_tests='MAIN_REQUIRED',tiny='UNRUN',model='UNRUN'),
                    inherited_model_limit='GGUF shard bytes/mtime identities retained unchanged, not full shard SHA; metadata exact SHA and actual request identity separately required')
    with (D/'manifest.json').open('x') as f:
        json.dump(manifest,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(manifest=str(D/'manifest.json'),sha256=sha(D/'manifest.json'),GPU_executed=False)))


if __name__ == '__main__':
    main()
