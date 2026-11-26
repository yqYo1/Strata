"""Rebuild the exact primary DFlash2 engine with the indexed head option."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_decode_handoff_memory as handoff


CANDIDATE = '9866bec75d379342653a82f0d21523a1ae70af13'
CHANGES = ('src/core/dflash2.cpp', 'include/strata/core/dflash2.hpp',
           'src/kernels/sycl/native_mmvq.cpp', 'include/strata/kernels/native_mmvq.hpp',
           'tests/sycl/dflash2.cpp')
THIS = str(Path(__file__).resolve().relative_to(handoff.bank.P))


def build_controller():
    path = handoff.bank.P / 'bench/dflash2_cold_host_build.py'
    ctx = handoff.basis.core.source.device.load_module('dflash2_indexed_head_native_build', path)
    root = ctx.bank.B / 'dflash2-decode-indexed-head-native-v1'
    ctx.ROOT, ctx.KEY, ctx.TREE = root, 'dflash2_decode_indexed_head_native_v1', root / 'source'
    ctx.OVERLAY, ctx.OVERLAYS = CANDIDATE, (handoff.GENERATOR, *CHANGES)
    ctx.CONTROLLERS = (*ctx.CONTROLLERS, handoff.THIS, THIS)
    original_save = ctx.save

    def projection(base):
        recovery = ctx.bank.previous.read(handoff.BUILD_ROOT / 'record.json')
        ctx.require(handoff.retained.closed(recovery, 'built')
                    and {p.name for p in handoff.BUILD_ROOT.iterdir()} == {'record.json'}
                    and not (ctx.bank.B / 'dflash2-cold-layer-native-v1/build/strata').exists(),
                    'retire the finished memory observer before this coherent build')
        measured = ctx.bank.basis.validation.committed(ctx.bank.RESULT, (handoff.KEY,))
        ctx.require(handoff.retained.closed(measured, 'closed')
                    and measured['base_native_source_commit'] == recovery['native_base_commit'] == base['native_source_commit']
                    and measured['native_source_commit'] == recovery['overlay_commit'] == handoff.NATIVE
                    and measured['binary'] == recovery['binaries']['strata']
                    and measured['decision']['observed_arithmetic_margin_bytes'] < 0,
                    'require the exact closed stock headroom experiment')
        sources = base['source_files']
        project = ctx.bank.P
        payload = dict(ctx.source.blobs(project, base['native_source_commit'], sources))
        ctx.require(ctx.bank.stage.hashed(payload) == sources, 'recover the exact primary source inputs')
        for revision, names in ((handoff.NATIVE, (handoff.GENERATOR,)), (CANDIDATE, CHANGES)):
            before = dict(ctx.source.blobs(project, revision + '^', names))
            after = dict(ctx.source.blobs(project, revision, names))
            ctx.require(before == {name: payload[name] for name in names}
                        and all(before[name] != raw and (project / name).read_bytes() == raw
                                for name, raw in after.items()), 'indexed head overlay or preimages differ')
            payload.update(after)
        recipe = base['recipe']
        prior = str(ctx.primary.ROOT)
        configure = [arg.replace(prior, str(root)) for arg in recipe['configure']]
        ctx.require(configure[2:5] == [str(ctx.TREE), '-B', str(root / 'build')]
                    and recipe['targets'] == ctx.primary.TARGETS, 'relocate only the primary recipe paths')
        return payload, dict(configure=configure, targets=recipe['targets'],
                             environment={k: v.replace(prior, str(root)) for k, v in recipe['environment'].items()})

    def save(row):
        row.update(scope='Coherent primary engine with the proved memory observer and optional indexed DFlash2 head.',
                   retention='Keep the two validated phase builds. Retain this temporary comparison generation '
                             'through its indexed-head admission and matched comparison or stop on a failed gate.')
        original_save(row)

    ctx.projection, ctx.save = projection, save
    return ctx


if __name__ == '__main__':
    build_controller().main(*sys.argv[1:])
