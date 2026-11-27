"""Build the first-gather scheduling change on the measured PLE source."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_cold_ple_measure as ple

window, bank, wait, require = ple.window, ple.bank, ple.wait, ple.require
base_build = window.build
builder, source = base_build.builder, base_build.source
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_ple_early_native_v2'
ROOT = bank.B / KEY.replace('_', '-')
TREE = ROOT / 'source'
PREIMAGE = '24ea4aff0c17f38e1cd5efbe0b0ce371bab2dd28'
PATCH = '8a2a6c4b2fc70b3465bee13555b2f83e961dfb57'
REJECTED = '6db247db794b3f4dd7cdd002c5ef66bf67552a21'
V1_CHECK = 'dflash2_prefill_ple_early_check_v1'
V1_NATIVE = 'dflash2_prefill_ple_early_native_v1'
OVERLAYS = ('include/strata/prefill/prefill.hpp', 'src/prefill/layer_major.inc', 'src/prefill/prefill.cpp')


def observed_build():
    row = bank.previous.read(window.BUILD_ROOT / 'record.json')
    evidence = ple.reference()
    proof = wait.basis.validation.committed(bank.RESULT, (ple.KEY,))
    compact = builder.compact(row)
    require(wait.retained.closed(row, 'built')
            and compact == wait.basis.validation.committed(bank.RESULT, (window.BUILD_KEY,))
            and wait.retained.closed(proof, 'closed')
            and proof['binary'] == row['binaries']['strata']
            and proof['native_source_commit'] == window.SOURCE
            and proof['native_build_sha256'] == bank.signature(compact)
            and proof['window_evidence_sha256'] == bank.signature(evidence)
            and proof['decision'] == ple.decision(proof['processes'])
            and proof['decision']['historical_warm_gate'],
            'require two closed PLE observations on the exact predecessor')
    delays = proof['decision']['first_minus_subsequent']
    require(len(delays) == 2 and all(v['exposed_wait_s'] > abs(v['unclassified_after_wait_s'])
                                    for v in delays.values()), 'require repeated exposed PLE wait')
    return row, proof


def recover_predecessor(row):
    fields, payload, original, _ = window.inputs(row['source_commit'])
    require(all(row[k] == v for k, v in fields.items()),
            'recover the original observed source, compiler and recipe')
    return fields, payload, original


def inputs(revision):
    predecessor, evidence = observed_build()
    fields, payload, original = recover_predecessor(predecessor)
    before = dict(source.blobs(bank.P, PREIMAGE, OVERLAYS))
    after = dict(source.blobs(bank.P, PATCH, OVERLAYS))
    require(all(payload[p] == before[p] and after[p] != before[p] for p in OVERLAYS),
            'apply the scheduling patch only to its exact observed preimages')
    payload.update(after)
    hashes = bank.stage.hashed(payload)
    controls = source.pin(bank.P, [*predecessor['controller_sources'], ple.THIS, THIS])['source_sha256']
    require(bank.stage.hashed(dict(source.blobs(bank.P, revision, controls))) == controls,
            'commit the early PLE controller and its complete inputs')

    old, new = str(window.BUILD_ROOT), str(ROOT)
    recipe = predecessor['recipe']
    args = [arg.replace(old, new) for arg in recipe['configure']]
    require(args[2:5] == [str(TREE), '-B', str(ROOT / 'build')]
            and recipe['targets'] == base_build.primary.TARGETS, 'relocate only the observed build paths')
    fields.update(source_commit=revision, overlay_commit=PATCH, source_files=hashes,
                  overlays={p: hashes[p] for p in OVERLAYS}, controller_sources=controls,
                  observed_build_sha256=bank.signature(builder.compact(predecessor)),
                  ple_evidence_sha256=bank.signature(evidence),
                  fresh_object_paths_sha256=bank.signature(
                      sorted(str(Path(p).relative_to(window.BUILD_ROOT / 'build'))
                             for p in predecessor['actual_inputs']['objects'])),
                  recipe=dict(configure=args, targets=recipe['targets'],
                              environment={k: v.replace(old, new) for k, v in recipe['environment'].items()}))
    return fields, payload, original, predecessor


def prepare(ctx):
    require(not ROOT.exists() and KEY not in bank.basis.validation.committed(bank.RESULT),
            'inspect the existing candidate generation before preparing')
    require({p.name for p in window.BUILD_ROOT.iterdir()} == {'record.json'},
            'retire the closed diagnostic generation before preparing the candidate')
    previous_check = bank.previous.historic(REJECTED, V1_CHECK)
    previous_build = bank.previous.historic(REJECTED, V1_NATIVE)
    for key in (V1_CHECK, V1_NATIVE):
        path = bank.B / key.replace('_', '-')
        require({p.name for p in path.iterdir()} == {'record.json'}, 'retire the failed V1 candidate')
    require(previous_check['stage'] == 'failed' and previous_check['ownership_closed']
            and not previous_check['active']
            and previous_check['plan']['binary'] == previous_build['binaries']['strata']
            and previous_check['outcomes']['direct']['error'] == "RuntimeError('first gather did not reach layer 1')",
            'preserve the original missed scheduling outcome')
    revision = source.git(bank.P, 'rev-parse', 'HEAD').decode().strip()
    fields, payload, _, _ = inputs(revision)
    ROOT.mkdir(mode=0o700)
    for name in ('owned', 'tmp', 'cache'):
        (ROOT / name).mkdir(mode=0o700)
    for name, raw in payload.items():
        target = TREE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    row = dict(fields, stage='prepared', commands=[], active=False, passed=False, complete=False,
               ownership_closed=True, scope='Fresh observed-engine build with the dense-bank-safe early PLE inputs. '
                   'Verify actual issuance before STOP and consumption at layer 1 before comparing off/on.',
               retention='Keep for dispatch, cancellation, handoff and matched prefill measurements. '
                   'Review at the first failed gate or comparison closure.')
    ctx.check(row, committed=False)
    ctx.save(row)


def context():
    window.select_build()
    ctx = bank.basis.core.source.device.load_module('early_ple_build', base_build.__file__)
    ctx.KEY, ctx.ROOT, ctx.TREE, ctx.OVERLAYS = KEY, ROOT, TREE, OVERLAYS
    ctx.inputs = inputs
    inherited_check = ctx.check

    def check(row, committed=True):
        original = inherited_check(row, committed=committed)
        if row['stage'] == 'built':
            observed = sorted(str(Path(p).relative_to(ROOT / 'build')) for p in row['actual_inputs']['objects'])
            require(bank.signature(observed) == row['fresh_object_paths_sha256'],
                    'rebuild every original translation unit')
        return original

    ctx.check = check
    ctx.prepare = lambda: prepare(ctx)
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    context().main(action)
