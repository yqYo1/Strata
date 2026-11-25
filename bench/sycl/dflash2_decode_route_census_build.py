"""Build the isolated DFlash2 decode census with the retained primary recipe."""

import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import dflash2_primary_prefill_build as primary

bank, builder, require = primary.bank, primary.builder, primary.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
BASE = primary.ROOT
BASE_KEY = primary.KEY
BASE_EVIDENCE = '0ab86d36b'
BASE_SOURCE = '968e71988953dca3d98037d3b3fec9f903bdfd6f'
CHECKED_SOURCE = '88215efa1b45b1f7383cda91ce05775f76da1dba'
BASE_BINARY = '5153182852b607e781570b606d4b2a444b143027425da90c8d05f1a31e85522e'
KEY = 'dflash2_decode_route_census_native_v1'
ROOT = bank.B / KEY.replace('_', '-')
FIXTURE = 'tests/core/dflash2_decode_route_census_test.cpp'
UNITS = ('src/program/generate.cpp', 'include/strata/program/dflash2_decode_route_census.hpp', FIXTURE)
TARGETS = [*primary.TARGETS, 'dflash2_decode_route_census_test']
NATIVE_SCOPE = ('CMakeLists.txt', 'cmake', 'src', 'include', 'third_party', 'tests')
CHANGES = frozenset(('CMakeLists.txt', *UNITS))


def reference():
    row = bank.previous.read(BASE / 'record.json')
    original = bank.previous.historic(BASE_EVIDENCE, BASE_KEY)
    engine = row['binaries']['strata']
    require(row['stage'] == 'built' and row['passed'] and row['complete']
            and not row['active'] and row['ownership_closed']
            and builder.compact(row) == original and row['native_source_commit'] == BASE_SOURCE
            and engine['sha256'] == BASE_BINARY and bank.sha(engine['path']) == BASE_BINARY
            and Path(engine['path']).stat().st_size == engine['bytes'],
            'recover the actual retained primary build and preserve its engine')
    return row


def changed_sources():
    revision = bank.basis.validation.git('rev-parse', 'HEAD').decode().strip()
    status = subprocess.check_output(['git', 'status', '--porcelain=v1', '--untracked-files=all'],
                                     cwd=bank.P, timeout=30)
    require(not status, 'commit exact build sources, recipe and fixture before any stage')
    names = bank.basis.validation.git('diff', '--name-only', BASE_SOURCE, revision, '--', *NATIVE_SCOPE)
    require(set(names.decode().splitlines()) == CHANGES, 'another native source change is bundled')
    untested = bank.basis.validation.git('diff', '--name-only', CHECKED_SOURCE, revision, '--', *NATIVE_SCOPE)
    require(not untested, 'the source differs from the checked native entrypoint or fixture')
    return revision


def recipe(row):
    original = row['recipe']
    args = [arg.replace(str(BASE), str(ROOT)) for arg in original['configure']]
    require(args[2:5] == [str(ROOT / 'source'), '-B', str(ROOT / 'build')]
            and original['targets'] == TARGETS[:-1],
            'primary build root or targets changed')
    return dict(configure=args, environment={k: v.replace(str(BASE), str(ROOT))
                                             for k, v in original['environment'].items()}, targets=TARGETS)


def main(action):
    require(action in ('plan', 'prepare', 'configure', 'compile'), 'use plan|prepare|configure|compile')
    reference_row = reference()
    revision = changed_sources()
    evidence = dict(native_source_commit=BASE_SOURCE, binary=reference_row['binaries']['strata'],
                    build_record_sha256=bank.sha(BASE / 'record.json'), control_evidence=BASE_EVIDENCE)

    primary.ROOT, primary.KEY, primary.THIS = ROOT, KEY, THIS
    primary.TARGETS = TARGETS
    primary.REQUIRED_SOURCES = (*primary.REQUIRED_SOURCES, *UNITS[1:])
    primary.recipe = lambda _: recipe(reference_row)
    original_save = primary.save

    def save(row):
        row['route_census_reference'] = evidence
        row['scope'] = ('Fresh coherent SYCL engine with only the request-local decode observer. '
                        'Use the retained primary engine for no-observer work parity.')
        row['retention'] = ('Preserve the primary prefill and independent decode references. '
                            'Retire this isolated build at the first failed gate or after the comparison.')
        original_save(row)

    primary.save = save
    if action == 'plan':
        print(json.dumps(dict(source_commit=revision, primary_reference=evidence, new_root=str(ROOT),
                              native_changed_sources=sorted(CHANGES), recipe_configure=recipe(reference_row)['configure'],
                              targets=TARGETS, produces_artifacts=False), indent=2))
        return
    if action != 'prepare':
        row = bank.previous.read(ROOT / 'record.json')
        require(row['route_census_reference'] == evidence and row['native_source_commit'] == revision,
                'candidate source or retained primary build changed')
    primary.main(action)


if __name__ == '__main__':
    action, = sys.argv[1:]
    main(action)
