"""Freeze CPU-only launch-property audit and matched-wrapper preparation."""
from pathlib import Path
import hashlib
import json
import shutil

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
out = root / 'bench/results/2026-10-05-sycl-prefill-scaling/gpu-stall-diagnosis/dequant-launch-properties-20261007'
assert not out.exists(), 'Refuse to overwrite a frozen archive'
build = json.loads((base / 'dequant-no-root-build/record.json').read_text())
probe = json.loads((base / 'dequant-no-root-probe-build/record.json').read_text())
assert build['passed'] and probe['passed']
assert build['production_inputs_unchanged'] and probe['inputs_unchanged']
assert not build['gpu_tested'] and not probe['gpu_tested'] and not build['adopted']
assert sum(x['replaced'] for x in build['archive_members']) == 1
assert [x['member'] for x in build['archive_members'] if x['original_sha256'] != x['candidate_sha256']] == ['iq_kernels.dp.cpp.o']
out.mkdir(mode=0o755)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def copy(source, target):
    destination = out / target
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    assert digest(source) == digest(destination)


for name in ['build_dequant_no_root.py', 'build_dequant_no_root_probe.py',
             'run_dequant_actual_wrapper_probe.py', 'dequant_actual_wrapper_probe.cpp',
             'archive_dequant_no_root_preparation.py']:
    copy(base / name, Path('sources-used') / name)
for name in ['dequant-no-root-audit.json', 'dequant-no-root-control-api-snippets.json']:
    copy(base / name, Path(name))
for name in ['dequant-no-root-build', 'dequant-no-root-probe-build']:
    folder = base / name
    for path in sorted(folder.iterdir()):
        if path.is_file() and (path.suffix in ('.json', '.diff', '.cpp', '.stdout', '.stderr')):
            assert path.stat().st_size < 400000
            copy(path, Path(name) / path.name)
assert digest(root / 'sycl/src/kernels/cuda/iq_kernels.dp.cpp') == build['production_source_sha256']
assert digest(root / 'build-sycl-upstream-jit/strata') == build['production_binary_sha256']

readme = '''# Dequant launch-property preparation on 2026-10-07

These are CPU-only audit/build receipts on the existing B570 host with
oneAPI 2026.1.1. Neither new executable has run on a GPU. Production
sources, objects, archives and the executable remain unchanged.

The [2K profile](../profiling-20261007/README.md) recorded 100,060 flat/GU
expert dequant launches taking 4.628 seconds of device execution. Source
review finds `use_root_sync` on both wrappers and no root-group access in
this source file. The earlier short model's [kernel-specific API excerpts](dequant-no-root-control-api-snippets.json)
show both actual wrappers launching with `UR_KERNEL_LAUNCH_FLAG_COOPERATIVE`.
The [extension specification](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_root_group.asciidoc)
imposes a root-group-compatible launch and a work-group limit. The actual
device limit was not queried; this is not a demonstrated invalid range,
undefined behavior or a stall cause.

The [previous work-group probe](../../dequant-workgroup-probe/README.md)
already passed 432 full FP16 comparisons, but changed kernel names and
wrappers and skipped the original capability check. Its exploratory times
do not isolate this property.

The new private engine replaces only these two declarations with empty
property lists. Kernel names, formulas, ND-ranges, 32-item work-groups,
queue choice, capability checks, other properties and every other archive
member remain as before. The [source diff](dequant-no-root-build/source.diff)
and [build receipt](dequant-no-root-build/record.json) record the change;
offline compile/link passed in 28.277 seconds. This is not an engine runtime
measurement or an adopted optimization.

A [second build](dequant-no-root-probe-build/record.json) links one common
host fixture object separately against the original and changed kernel
objects. It passed offline compile/link in 9.134 seconds. The prepared
fixture will compare 144 full guarded FP16 outputs per executable across
nine types, two seeds, small/odd/640-row shapes, flat/GU layout and
zero/positive/negative scales. No comparisons have run yet.

The [deferred controller](sources-used/run_dequant_actual_wrapper_probe.py)
requires the separate full-context job's successful terminal receipt,
recorded absence of live jobs and disappearance of its pinned owners
before loading a GPU runtime. It captures API logs/environment through
owned GDB and bounds each binary to 180 seconds. Full-model head parity,
clean timings and the complete 256K gates remain separate requirements.
The [audit](dequant-no-root-audit.json) records provenance and limits;
the manifest hashes every archived file except itself. Executables,
objects, archives and unbounded logs are not included.
'''
(out / 'README.md').write_text(readme)
files = sorted(p for p in out.rglob('*') if p.is_file())
manifest = {'scope': 'CPU-only dequant launch-property audit/build preparation; no new GPU execution',
            'entries': [{'path': str(p.relative_to(out)), 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in files]}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
assert {str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()} == {x['path'] for x in manifest['entries']} | {'manifest.json'}
assert all((out / x['path']).stat().st_size == x['bytes'] and digest(out / x['path']) == x['sha256'] for x in manifest['entries'])
print(json.dumps({'archive': str(out), 'files': len(files) + 1,
                  'payload_bytes': sum(x['bytes'] for x in manifest['entries']),
                  'exact_manifest_checked': True, 'gpu_tested': False}, indent=2))
