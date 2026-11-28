"""Compare two-chunk and whole-request PLE without timing observers."""

from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_ple_record_window as grouped

bank, wait, require = grouped.bank, grouped.wait, grouped.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ORDER = grouped.FULL_ORDER


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER
            and all(p['passed'] and p['complete'] and p['ownership_closed']
                    and p['clean_speed'] and p['native_exit_code'] == 0 and not p['fresh_faults']
                    for p in processes)
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                     for p in processes}) == len(ORDER),
            'require four independently closed clean processes')
    forward = grouped.request_time_ratios(processes[0], processes[1])
    reverse = grouped.request_time_ratios(processes[3], processes[2])
    pairs = [forward, reverse]
    return dict(reference='two', candidate='full', pairs=pairs,
                repeated_first_gain=all(p['first_pp_time'] < 1 and p['first_ttft'] < 1 for p in pairs),
                subsequent_ratios={name: statistics.median(p[name] for p in pairs)
                                   for name in ('subsequent_pp_time', 'subsequent_ttft')},
                first_1000=[p['samples'][0]['prompt_ms'] <= wait.READ
                            for p in processes if p['arm'] == 'full'], adopted=False)


def experiment():
    ctx = grouped.experiment(clean='full')
    ctx.ORDER = ORDER
    ctx.extra_sources = (*ctx.extra_sources, THIS)
    ctx.decision = decision
    ctx.protocol.update(
        order=list(ORDER), reference='two', candidate='full',
        scope='Use the same executable, fresh prose input, ordinary row cache and native work. '
              'Change only the PLE record horizon and reverse process order in the second pair. '
              'Disable the PLE observer and shared storage counters in both arms.',
        decision='Require faster first prefill and client TTFT in both pairs. Keep individual subsequent '
                 'samples and report 1000 tok/s separately. Do not adopt from this timing alone.',
        limits='One technical-prose workload. The earlier legacy and two-chunk comparisons, '
               'cache interaction, physical-context and decode qualification are separate.')
    return ctx


if __name__ == '__main__':
    require(sys.executable == bank.basis.reference.model.PYTHON, 'use the pinned controller interpreter')
    wait.speed.clock.main(experiment())
