"""Recover the target-snapshot build's recorded sources using its Git recipe."""

import fcntl
import json
from pathlib import Path
import sys
from types import ModuleType

sys.dont_write_bytecode = True
import dflash2_primary_prefill_build as primary

bank, builder, require = primary.bank, primary.builder, primary.require
REVISION = '4b4ac554cc604a9047da692523df7b336cd3cbcb'
KEY = primary.SNAPSHOT_KEY
ROOT = primary.SNAPSHOT_ROOT


def archived_module(name):
    """Run the two project-owned historical projection controllers in memory."""
    path = Path('bench/sycl') / (name + '.py')
    raw = bank.basis.validation.git('show', REVISION + ':' + str(path))
    module = ModuleType(name)
    module.__file__ = str(bank.P / path)
    sys.modules[name] = module
    exec(compile(raw, f'{REVISION}:{path}', 'exec'), module.__dict__)
    return module


def recover():
    row = bank.previous.read(ROOT / 'record.json')
    measured = bank.previous.historic(primary.PREFILL, 'dflash2_snapshot_direct_prefill_v1')
    frozen = measured['snapshot_reference']
    require(row['stage'] == 'built' and row['passed'] and row['complete'] and row['ownership_closed']
            and not row['active'] and row['native_source_commit'] == frozen['source'] == REVISION
            and row['binaries']['strata'] == frozen['binary']
            and bank.signature(builder.compact(row)) == frozen['native_build_sha256'],
            'the target snapshot receipt must match the original long-prefill comparison')
    archived_module('prefill_kv_upload_build')
    snapshot = archived_module('dflash2_target_snapshot_build')
    native, foundation, files, recipe, projection = snapshot.project()
    hashes = bank.stage.hashed(files)
    require(hashes == row['source_files'] and recipe == row['recipe']
            and projection == row['projection'] and foundation == row['foundation']
            and native['external_inputs'] == row['external_inputs'],
            'historical projection did not reproduce the exact snapshot recipe and bytes')
    source = ROOT / 'source'
    compiled = {Path(p).relative_to(source).as_posix(): h for p, h in row['actual_inputs']['compiled'].items()
                if Path(p).is_relative_to(source)}
    require(compiled and all(hashes.get(path) == h for path, h in compiled.items()),
            'reconstructed source does not bind all project compiler inputs')
    return dict(native_source_commit=REVISION, measured_reference_commit=primary.PREFILL,
                recovered_source_files=len(hashes), source_manifest_sha256=bank.signature(hashes),
                recipe_sha256=bank.signature(recipe), compiled_project_files=len(compiled),
                original_binary_sha256=frozen['binary']['sha256'], retained_artifacts=False)


if __name__ == '__main__':
    require(len(sys.argv) == 1, 'run only the fixed target snapshot recovery')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        builder.guards.no_model_owners([])
        print(json.dumps(recover(), indent=2))
