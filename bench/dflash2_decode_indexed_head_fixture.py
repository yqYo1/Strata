"""Qualify the indexed DFlash2 Q6 head on the coherent comparison build."""

import fcntl
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_decode_indexed_head_build as indexed


handoff = indexed.handoff
bank, basis, require = handoff.bank, handoff.basis, handoff.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_decode_indexed_head_fixture_v1'
TARGET = 'sycl_dflash2_test'


def controller():
    build_ctx = indexed.build_controller()
    path = bank.P / 'bench/dflash2_primary_checks.py'
    fixture = basis.core.source.device.load_module('dflash2_indexed_head_fixture', path)
    sources = (THIS, fixture.THIS, indexed.THIS, handoff.THIS)
    parameters, save = fixture.parameters, fixture.save
    fixture.KEY, fixture.ROOT, fixture.THIS = KEY, bank.B / KEY.replace('_', '-'), THIS
    fixture.ORDER = ('indexed',)
    fixture.BUDGETS = {'indexed': fixture.device.SMALL}
    fixture.COMMANDS = {'indexed': ((TARGET, ('--indexed-q6-head',)),)}

    def build():
        row = bank.previous.read(build_ctx.ROOT / 'record.json')
        require(handoff.retained.closed(row, 'built') and row['overlay_commit'] == indexed.CANDIDATE,
                'complete the coherent indexed head build first')
        original = build_ctx.check(row)
        return row, original['native_validation_probe']

    def envelope(row, probe):
        view = dict(row, native_source_commit=indexed.CANDIDATE)
        result = parameters(view, probe)
        return dict(result, build_sha256=bank.signature(build_ctx.builder.compact(row)),
                    controller_inputs=bank.stage.source.pin(bank.P, sources)['source_sha256'])

    def coverage(target, output):
        require(target == TARGET and output.splitlines().count('dflash2 indexed Q6_K head cases=14 PASS') == 1,
                'indexed Q6 fixture output, bounds or parity checks are incomplete')
        return dict(cases=14, scope='bitwise gathered/indexed output, invalid rows and canaries')

    def save_result(row):
        row['scope'] = 'Bounded traced GPU parity for the indexed DFlash2 head on its coherent build.'
        save(row)

    fixture.build, fixture.parameters, fixture.coverage = build, envelope, coverage
    fixture.save = save_result
    return fixture


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(action in ('prepare', 'indexed'), 'select indexed fixture preparation or execution')
    fixture = controller()
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fixture.builder.guards.no_model_owners([])
        fixture.prepare() if action == 'prepare' else fixture.execute(action)
