"""Screen DFlash2 draft acceptance on the existing 96K prose task."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_primary_continuation as primary


bank, basis, require = primary.bank, primary.basis, primary.require
continuation = primary.continuation
client = continuation.suite.official.d.client
KEY = 'dflash2_sampled_continuation_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ORDER = ('sampled-prose',)
WORKLOADS = ('varied-prose',)
SEED = 1
SAMPLING = client.long_generation.sampling_plan('qwen-thinking', SEED)
PLAN = dict(primary.PLAN, order=list(ORDER), sampling=SAMPLING,
    scope='One independent process on the retained DFlash2 engine. Use the existing 96K prose task and warmups '
          'with thinking enabled and the Qwen sampling profile with a fixed seed.',
    evidence='Check the actual native long-prefill route and reconcile the client and engine counts. '
             'Screen draft acceptance against the existing 40 percent weak-case band.',
    stop='Stop on the first source, runtime, workload, route, ownership or validation failure. '
         'A successful sample only screens the workload. Qualification needs repeated independent processes.')


def command(plan):
    return [*continuation.command(plan), '--long-generation-workload', WORKLOADS[0],
            '--long-generation-sampling', SAMPLING['profile'], '--long-generation-seed', str(SEED)]


def read_complete_suite(root, workload_id):
    evidence = continuation.read_complete_suite(root, workload_id, workloads=WORKLOADS)
    path, = (root / 'bench').glob('*/results.json')
    result = bank.previous.read(path)
    require(result['params']['long_generation_sampling']
            == result['comparison_identity']['long_generation_sampling']
            == result['results']['long_generation']['sampling'] == SAMPLING,
            'the recorded sampling profile or fixed seed changed')
    return dict(evidence, sampling=SAMPLING)


def decision(processes):
    process, = processes
    require(process['arm'] == ORDER[0] and process['passed'] and process['complete']
            and process['ownership_closed'] and process['native_exit_code'] == 0 and not process['fresh_faults'],
            'require one normal sampled DFlash2 process')
    sample, = [s for s in process['evidence']['samples'] if s.get('workload') == WORKLOADS[0]]
    require(sample['controller_eligible'] and process['evidence']['sampling'] == SAMPLING,
            'the sustained sampled request must qualify for agreement accounting')
    accepted, offered = sample['drafts_accepted'], sample['drafts_offered']
    return dict(read_tokens=sample['read_tokens'], generated_tokens=sample['generated_tokens'],
                accepted=accepted, offered=offered, acceptance=accepted / offered,
                weak_band_screen=5 * accepted <= 2 * offered, weak_case_confirmed=False,
                prefill_tps=sample['prompt_tps'], ttft_s=sample['ttft_s'],
                decode_tps=process['evidence']['decode_rates'][WORKLOADS[0]],
                comparison_qualified=False, adopted=False)


def experiment():
    ctx = primary.experiment()
    base_pins = ctx.pins
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.command, ctx.protocol = command, PLAN
    ctx.routes = lambda log, samples, _arm: primary.routes(log, samples, 'dflash2', workloads=WORKLOADS)
    ctx.evidence_reader.read_complete_suite, ctx.evidence_reader.decide = read_complete_suite, decision

    def pins(revision):
        values = base_pins(revision)
        file = bank.P / THIS
        require(file.read_bytes() == basis.validation.git('show', revision + ':' + THIS),
                'commit the sampled continuation controller before using it')
        values[THIS] = bank.sha(file)
        return values

    ctx.pins = pins
    return ctx


if __name__ == '__main__':
    primary.clock.main(experiment())
