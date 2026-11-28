"""Compare sustained DFlash2 after verified dedicated-cache RESTORE."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_ple_phase_cache_restore as restore
import dflash2_prefill_ple_record_window as grouped

phase, bank, wait = grouped.phase, grouped.bank, grouped.wait
validate, require = grouped.validate, grouped.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))


def experiment():
    source = wait.basis.core.source.device
    peer = source.load_module('ple_phase_handoff_from_receipt', phase.__file__)
    # Keep the measured controller's original source and workload while
    # verifying its older timing references through the committed receipts.
    peer.baseline = grouped.retained_phase_baseline
    clock, ctx = peer.handoff_experiment()
    old_baseline, old_pins = ctx.baseline, ctx.pins

    def baseline():
        row = old_baseline()
        physical = validate.committed(bank.RESULT, (restore.context.KEY,))
        continued = validate.committed(bank.RESULT, (restore.KEY,))
        require(wait.retained.closed(physical, 'closed')
                and physical['outcomes']['full-copy']['final_slot_consumed']
                and physical['native_build_sha256'] == row['native_build_sha256']
                and wait.retained.closed(continued, 'closed')
                and continued['physical_sha256'] == bank.signature(physical)
                and continued['timing_sha256'] == row['phase_prefill_proof_sha256']
                and all(continued[k] == row[k] for k in ('binary', 'native_build_sha256', 'native_source_commit')),
                'require the corresponding physical-context and live RESTORE results')
        return dict(row, phase_physical_sha256=bank.signature(physical),
                    phase_restore_sha256=bank.signature(continued))

    def pins(revision):
        values = old_pins(revision)
        for name in (THIS, grouped.THIS, restore.THIS, restore.context.THIS):
            path = bank.P / name
            require(path.read_bytes() == validate.git('show', revision + ':' + name),
                    'commit the live phase handoff controller: ' + name)
            values[name] = bank.sha(path)
        return values

    ctx.THIS, ctx.baseline, ctx.pins = THIS, baseline, pins
    ctx.protocol['eligibility'] = 'Require same-build physical final-slot and matched live RESTORE receipts.'
    ctx.protocol['decision'] = ('Require the same native prefill and sustained decode work and at least stock '
                                'geometric mean TG on both workloads. Retain individual PP, TG, TTFT and draft '
                                'counts. Weak acceptance and agent sessions need separate evidence.')
    return clock, ctx


if __name__ == '__main__':
    require(sys.executable == bank.basis.reference.model.PYTHON, 'use the pinned controller interpreter')
    clock, campaign = experiment()
    clock.main(campaign)
