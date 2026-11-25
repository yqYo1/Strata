"""Recover the exact primary build foundation without reading old executables."""

import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_primary_prefill_build as primary

bank, builder, require = primary.bank, primary.builder, primary.require
ORIGINAL = builder.original
COHERENT = ORIGINAL.coherent
EVIDENCE = '0ab86d36b'
PRIMARY_KEY = primary.KEY
PRIMARY_ROOT = primary.ROOT


def historical_foundation(*, expected=None):
    """Use the committed primary proof and recheck source and live SDK inputs.

    Old executable identities continue to bind the original closed outcomes.
    The current probe, runtime files and toolchain still require live checks.
    The original build's source and recipe metadata remain required inputs.
    """
    baseline = bank.previous.historic(EVIDENCE, PRIMARY_KEY)
    primary_row = bank.previous.read(PRIMARY_ROOT / 'record.json')
    require(primary_row['stage'] == 'built' and primary_row['complete'] and primary_row['passed']
            and primary_row['ownership_closed'] and not primary_row['active']
            and builder.compact(primary_row) == baseline,
            'recover the exact committed primary build')
    proof = primary_row['foundation']
    require(expected is None or proof == expected, 'the pinned primary foundation changed')
    require(proof['original_build_evidence'] == ORIGINAL.EVIDENCE
            and proof['original_native_source'] == ORIGINAL.SOURCE
            and proof['reader_sha256'] == bank.sha(ORIGINAL.__file__),
            'the original historical reader identity changed')

    record = ORIGINAL.ROOT / 'record.json'
    require(bank.sha(record) == proof['original_record_sha256'], 'the original foundation receipt changed')
    row = bank.previous.read(record)
    original = bank.previous.historic(ORIGINAL.EVIDENCE, ORIGINAL.KEY)
    require(COHERENT.build.compact(row) == original and row['native_source_commit'] == ORIGINAL.SOURCE
            and bank.signature(row['recipe']) == proof['original_recipe_sha256']
            and row['binaries'] == proof['same_build_executables'],
            'the original executable identities or build recipe changed')
    recovered = COHERENT.recover(row['project'])
    for pin in row['dependencies']:
        recovered.update(COHERENT.recover(pin))
    require(len(recovered) == proof['original_source_dependency_files'],
            'the original project and dependency source set changed')
    for key, digest in proof['closed_evidence_sha256'].items():
        evidence = bank.previous.historic(EVIDENCE, key)
        require(bank.signature(evidence) == digest, 'an original closed qualification changed: ' + key)

    for item in (row['recipe_seed'], row['external_inputs'], proof['runtime_manifest'],
                 *row['native_validation_probe'].values()):
        if not isinstance(item, dict) or 'path' not in item or 'sha256' not in item:
            continue
        require(bank.sha(item['path']) == item['sha256'], 'a live foundation input changed: ' + item['path'])
    require(all(bank.sha(p) == h for p, h in row['toolchain'].items()), 'the original toolchain changed')
    for manifest in (row['external_inputs'], proof['runtime_manifest']):
        inputs = bank.previous.read(Path(manifest['path']))
        require(all(bank.sha(p) == h for p, h in inputs.items()), 'a live SDK or runtime input changed')
    configured = row['configured_files'] | row['generated_inputs']
    require(all((recovered[p] if p in recovered else bank.sha(p)) == h for p, h in configured.items()),
            'the retained configurable/generated source metadata changed')
    census = row['actual_inputs']
    require(bank.sha(census['path']) == census['sha256'] == proof['actual_compiler_census_sha256']
            and census['source_commit'] == ORIGINAL.SOURCE and census['uncovered'] == 0,
            'the original compiler census changed')
    actual = bank.previous.read(Path(census['path']))
    allowed = recovered | bank.previous.read(Path(row['external_inputs']['path'])) | row['generated_inputs']
    require(all(allowed.get(p) == h for group in ('compiled', 'generated', 'links')
                for p, h in actual[group].items()),
            'original compiler inputs do not bind the recovered sources and pinned SDK')
    return row, proof
