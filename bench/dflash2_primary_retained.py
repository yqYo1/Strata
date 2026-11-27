"""Verify the selected primary DFlash2 build and its closed phase evidence."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_primary_prefill_speed as speed

bank, basis, checks, require = speed.bank, speed.basis, speed.checks, speed.require
PRIMARY = checks.primary
BUILD = '0ab86d36b7ac37465a4f577a25bedbf7e66fb2b2'
TIMING = 'bfc0c54aea58bc7f9e8a75103fa370606c54dd9d'
CONTEXT = '137fb933f0edb2a7e3c3f27bb1ee172399e5e7c9'
THIS = str(Path(__file__).resolve().relative_to(bank.P))


def closed(row, stage):
    return (row['stage'] == stage and row['complete'] and row['passed']
            and row['ownership_closed'] and not row['active'])


def baseline():
    native = bank.previous.read(PRIMARY.ROOT / 'record.json')
    measured = bank.previous.read(speed.ROOT / 'record.json')
    full = basis.validation.committed(bank.RESULT, (speed.physical.KEY,))
    timing = bank.previous.historic(TIMING, speed.KEY)
    compact = PRIMARY.builder.compact(native)
    identity = bank.signature(compact)
    revision = native['native_source_commit']
    engine = native['binaries']['strata']
    require(closed(native, 'built') and closed(measured, 'closed') and closed(full, 'closed')
            and compact == bank.previous.historic(BUILD, PRIMARY.KEY)
            and basis.compact(measured) == timing
            and full == bank.previous.historic(CONTEXT, speed.physical.KEY)
            and full['outcomes']['full-copy']['final_slot_consumed']
            and full['native_source_commit'] == measured['native_source_commit'] == revision
            and full['native_build_sha256'] == measured['native_build_sha256'] == identity
            and measured['binary'] == engine and timing['decision'] == speed.decision(timing['processes'])
            and timing['decision']['all_pairs_faster'] and timing['decision']['all_warm_requests_1000']['retain'],
            'require the exact selected primary build, timing and physical-context evidence')

    build_sources = native['source_files']
    recover = dict(bank.stage.source.blobs(bank.P, revision, build_sources))
    census = native['actual_inputs']
    require(bank.stage.hashed(recover) == build_sources
            and not census['partial_archive_reuse'] and census['uncovered'] == 0
            and all(bank.sha(path) == digest for path, digest in native['configured_files'].items())
            and all(bank.sha(path) == digest for group in bank.previous.ACTUAL for path, digest in census[group].items())
            and all(bank.sha(artifact['path']) == artifact['sha256']
                    and Path(artifact['path']).stat().st_size == artifact['bytes']
                    for artifact in native['binaries'].values()),
            'the selected source, compiler/link inputs, executable or fixtures changed')
    external = native['external_inputs']
    runtime = measured['runtime_manifest']
    require(bank.sha(external['path']) == external['sha256']
            and bank.sha(runtime['path']) == runtime['sha256']
            and runtime['sha256'] == native['foundation']['runtime_manifest']['sha256']
            and bank.sha(measured['template']['path']) == measured['template']['sha256']
            and bank.sha(bank.P / 'data/expert-profile.bin') == measured['sources']['data/expert-profile.bin']
            and measured['boot'] == checks.BOOT.read_text().strip()
            and measured['driver'] == native['driver'] == PRIMARY.builder.guards.driver_guard(),
            'selected runtime manifests, template or platform differs')
    providers = bank.previous.read(Path(runtime['path']))
    require(len(providers) == runtime['files']
            and all(bank.sha(path) == digest for path, digest in providers.items())
            and all(bank.sha(path) == digest for path, digest in bank.previous.read(Path(external['path'])).items())
            and all(bank.sha(item['path']) == item['sha256'] for item in measured['native_validation_probe'].values())
            and all(bank.sha(path) == digest for path, digest in measured['dflash2_assets'].items()),
            'live SDK, runtime, validation or draft input differs')
    basis.comparison.shared.models.model_pins(measured['configuration'], measured['model_assets'], full=False)
    contract, manifest = basis.core.frontend.python_manifest()
    require(contract == measured['python_contract'] and manifest['selected'] == measured['python_selected'],
            'the selected benchmark runtime changed')
    fields = ('boot', 'driver', 'environment_template', 'native_validation_probe', 'configuration', 'model_assets',
              'runtime_manifest', 'template', 'ordinary_control', 'dflash2_assets', 'validated_dflash2_args',
              'binary', 'native_source_commit', 'native_build_sha256', 'physical_context_sha256', 'reference_evidence')
    require(measured['physical_context_sha256'] == bank.signature(full), 'physical-context receipt changed')
    return dict({k: measured[k] for k in fields}, python_contract=contract, python_selected=manifest['selected'],
                selected_arm='retain', selection_sha256=bank.signature(timing['decision']))
