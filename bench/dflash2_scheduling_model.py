"""Screen the pinned scheduling workload through Strata's DFlash2 API."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_reconciliation_model as model
from dflash2_reconciliation_client import Workload
from dflash2_scheduling_fixture import build, validate_answer


THIS = str(Path(__file__).resolve().relative_to(model.bank.P))
CASE = Workload('scheduling', 52055,
                'a7b6c72e9a06280eb216d5d34340a451b5deb07b363a3d0da4b554b45cc4874b',
                'bench/dflash2_scheduling_fixture.py', build, validate_answer)
PLAN = dict(model.PLAN, input_tokens=CASE.input_tokens, wire_sha256=CASE.wire_sha256,
            scope='One independent cold server on the selected primary engine. Screen the scheduling '
                  'packet through the production OpenAI SSE frontend with no benchmark warm-up.',
            acceptance='Bind the live input count, wire body, native segments and API history. '
                       'Verify the visible optimal schedule and record client TTFT, native generation and '
                       'draft coverage. A repeated weak trial requires at least 1024 native outputs, '
                       'at least 256 offers and acceptance at most 40 percent on this request.')


def decision(processes):
    result = model.decision(processes)
    process, = processes
    info = process['evidence']
    sample, = info['samples']
    generated = info['observation']['native_generated']
    return dict(result, draft_offers_per_native_output=sample['drafts_offered'] / generated if generated else None)


if __name__ == '__main__':
    if sys.argv[1:2] == ['client']:
        model.collect(Path(sys.argv[2]), int(sys.argv[3]), case=CASE)
    else:
        model.clock.main(model.experiment(CASE, source=THIS, protocol=PLAN, decide=decision))
