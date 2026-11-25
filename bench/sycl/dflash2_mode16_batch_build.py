"""Build the mode16 batch candidate using the exact primary SYCL recipe."""

import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import dflash2_primary_foundation_proof as foundation
import dflash2_primary_prefill_build as primary

bank, builder, require = primary.bank, primary.builder, primary.require
BASE, BASE_KEY = primary.ROOT, primary.KEY
BASE_SOURCE = '968e71988953dca3d98037d3b3fec9f903bdfd6f'
BASE_BUILD = '0ab86d36b'
PREFILL = 'bfc0c54aea58bc7f9e8a75103fa370606c54dd9d'
DEVICE_FIXTURE = '3552b9d7ad3e9271fe3044e365e965456d2dcd6d'
DECODE = primary.DECODE
KEY = 'dflash2_mode16_batch_native_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
NATIVE = ('CMakeLists.txt', 'cmake', 'src', 'include', 'third_party', 'tests')
CHANGES = ('src/prefill/layer_major.inc', 'tests/sycl/expert_bank.cpp')
REQUIRED_SOURCES = (*primary.REQUIRED_SOURCES, *CHANGES)


def references():
    build = bank.previous.read(BASE / 'record.json')
    prefill = bank.previous.historic(PREFILL, 'dflash2_primary_prefill_speed_v1')
    decode = bank.previous.historic(DECODE, 'dflash2_snapshot_regular_code_v1')
    controls = dict(prefill=prefill['binary'], decode=decode['binary'])
    require(build['stage'] == 'built' and prefill['stage'] == decode['stage'] == 'closed'
            and all(r['passed'] and r['complete'] and r['ownership_closed'] and not r['active']
                    for r in (build, prefill, decode))
            and builder.compact(build) == bank.previous.historic(BASE_BUILD, BASE_KEY)
            and build['native_source_commit'] == prefill['native_source_commit'] == BASE_SOURCE
            and build['binaries']['strata'] == prefill['binary']
            and bank.signature(builder.compact(build)) == prefill['native_build_sha256']
            and prefill['decision']['all_warm_requests_1000']['retain']
            and prefill['decision']['all_pairs_faster'] and decode['decision']['matched_work']
            and all(bank.sha(b['path']) == b['sha256'] and Path(b['path']).stat().st_size == b['bytes']
                    for b in controls.values()), 'the two verified phase reference builds are required')
    return build, controls


def recipe(build):
    original = build['recipe']
    command = [arg.replace(str(BASE), str(ROOT)) for arg in original['configure']]
    require(command[2:5] == [str(ROOT / 'source'), '-B', str(ROOT / 'build')]
            and original['targets'] == primary.TARGETS, 'preserve the selected primary build recipe')
    return dict(configure=command, targets=primary.TARGETS,
                environment={k: v.replace(str(BASE), str(ROOT)) for k, v in original['environment'].items()})


def plan(revision=None):
    git = bank.basis.validation.git
    head = git('rev-parse', 'HEAD').decode().strip()
    revision = head if revision is None else revision
    dirty = subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=bank.P, timeout=15)
    require(not dirty and not git('diff', '--name-only', revision, head, '--', *primary.PREFIXES)
            and tuple(git('diff', '--name-only', BASE_SOURCE, revision, '--', *NATIVE)
            .decode().splitlines()) == CHANGES
            and not git('diff', '--name-only', DEVICE_FIXTURE, revision, '--', *NATIVE),
            'preserve the frozen build sources and the two device-checked native changes')
    build, controls = references()
    return dict(source_commit=revision, phase_references=controls, native_changes=list(CHANGES), recipe=recipe(build))


def activate():
    primary.ROOT, primary.KEY, primary.THIS = ROOT, KEY, THIS
    primary.references, primary.recipe = references, recipe
    primary.foundation_reader = foundation.historical_foundation
    primary.REQUIRED_SOURCES = REQUIRED_SOURCES


def recorded_plan(record):
    frozen = plan(record['native_source_commit'])
    require(record['mode16_batch_plan'] == frozen, 'candidate build plan changed')
    return frozen


def admitted_build():
    record = bank.previous.read(ROOT / 'record.json')
    require(record['stage'] == 'built' and record['passed'] and record['complete']
            and record['ownership_closed'] and not record['active'], 'finish the coherent candidate build')
    recorded_plan(record)
    activate()
    primary.check(record)
    return record


def main(action):
    require(action in ('plan', 'prepare', 'configure', 'compile'), 'use plan|prepare|configure|compile')
    if action == 'plan':
        print(json.dumps(dict(plan(), produces_artifacts=False), indent=2))
        return
    if action == 'prepare':
        frozen = plan()
        legacy_roots = (primary.SNAPSHOT_ROOT, foundation.ORIGINAL.ROOT)
        require(not any((root / 'build' / 'strata').exists() for root in legacy_roots),
                'close the live legacy build consumers before adding the candidate generation')
    else:
        frozen = recorded_plan(bank.previous.read(ROOT / 'record.json'))
    save = primary.save

    def publish(record):
        record['mode16_batch_plan'] = frozen
        record['scope'] = 'Fresh SYCL model engine for a demand-versus-batched mode16 upload comparison.'
        record['retention'] = 'Retire the candidate after the comparison closes or a qualification gate fails.'
        save(record)

    activate()
    primary.PREFILL = PREFILL
    primary.save = publish
    primary.main(action)


if __name__ == '__main__':
    main(*sys.argv[1:])
