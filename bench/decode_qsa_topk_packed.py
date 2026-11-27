"""Compare exact two-bit QSA selection with the existing binary threshold search."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import decode_qsa_score_grid as shared
import decode_qsa_topk_packed_source as packed

bank, basis, require = shared.bank, shared.basis, shared.require
KEY = 'decode_qsa_topk_packed_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
GENERATOR = str(Path(packed.__file__).resolve().relative_to(bank.P))


def experiment():
    ctx = basis.core.source.device.load_module('decode_qsa_topk_packed_fixture', shared.__file__)
    ctx.KEY, ctx.ROOT, ctx.BUILD, ctx.TARGET = KEY, ROOT, ROOT / 'fixture-build', 'decode_qsa_topk_packed'
    ctx.THIS, ctx.source, ctx.GENERATOR, ctx.GENERATED = THIS, packed, GENERATOR, 'decode_qsa_topk_packed_kernel.cpp'
    ctx.PLAN = dict(ctx.PLAN,
        scope='Same retained QSA scorer and output order. Change only the verifier selector threshold scan. '
              'Time selector-only replay followed by original scorer plus selector replay.',
        fields_per_lane=3, bits_per_field=packed.FIELD_BITS, radix_bits=2, capacity=packed.CAPACITY,
        gates='Before timing, check exact selected IDs and inactive guards for the actual score producer and '
              'for selector-only raw keys covering ties, signed zero, NaNs, infinities and negative scores. '
              'Exercise every tail remainder, invalid steps and large/small reuse. '
              'Keep 10 ms per timing batch and at most 1% empty-clock overhead.',
        decision='A shape advances when selector-only and combined completion improve in every independent '
                 'process. Keep both arms and setup times. Any graph device time is separate attribution evidence.')
    ctx.PLAN.pop('grid')
    old_inputs, old_decision = ctx.inputs, ctx.decision

    def inputs(revision):
        original = ctx.validation.git('show', revision + ':' + shared.THIS)
        require((bank.P / shared.THIS).read_bytes() == original, 'the shared graph controller changed')
        fields, parent, sources = old_inputs(revision)
        sources[ctx.FIXTURE] = packed.generate_fixture(sources[ctx.FIXTURE])
        fields.update(overlay_sha256=bank.stage.hashed(sources), shared_controller_sha256=ctx.sha(shared.__file__))
        return fields, parent, sources

    def decision(processes):
        rows = old_decision(processes)
        for row in rows:
            row['selector_ratios'] = row.pop('scorer_ratios')
        return rows

    ctx.inputs, ctx.decision = inputs, decision
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    ctx = experiment()
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'build', 'profile', *ctx.STAGES),
            'use the pinned Python and one planned selector stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx.main(action)
