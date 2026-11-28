"""Locate first-use time outside PLE waits on the measured cache build."""

from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_prefill_ple_phase_cache as phase
import dflash2_prefill_ple_record_window as grouped
import dflash2_cold_window_measure as windows

bank, wait, require = phase.bank, phase.wait, phase.require
layer, coarse = windows.layer, windows.layer.coarse
cpu = coarse.wait
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_prefill_ple_phase_cache_residual_v1'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = ('large', 'large')
PREFIXES = (*cpu.OBSERVER_PREFIXES, coarse.PREFIX, layer.PREFIX, windows.PREFIX)
SELECTORS = (cpu.SELECTOR, coarse.SELECTOR, layer.SELECTOR, windows.SELECTOR)


def baseline():
    selected = grouped.retained_phase_baseline()
    timing = phase.validate.committed(bank.RESULT, (phase.KEY,))
    require(wait.retained.closed(timing, 'closed')
            and timing['decision'] == phase.capacity.decision(timing['processes'])
            and timing['decision']['warm_cache_mechanism_advances']
            and all(timing[k] == selected[k] for k in (
                'binary', 'native_source_commit', 'native_build_sha256', 'host_fixture_sha256')),
            'finish the same-binary cache timing before attributing first use')
    controls = [p['subsequent_median_ms'] for p in timing['processes'] if p['arm'] == 'large']
    require(len(controls) == len(ORDER) and min(controls) > 0, 'missing dedicated-cache speed controls')
    return dict(selected, timing_proof_sha256=bank.signature(timing), historical_warm_control_ms=controls)


def route(text, plan):
    plain = ''.join(line for line in text.splitlines(keepends=True) if not line.startswith(PREFIXES))
    frame = phase.route(plain, plan)

    def records(prefix):
        return [line.removeprefix(prefix) for line in text.splitlines() if line.startswith(prefix)]

    count = int(frame['work']['windows'])
    rows = plan['read'] - 1
    require(count == 48 and frame['work']['maximum_rows'] == str(rows), 'unexpected layer or window geometry')
    v4 = cpu.cpu_partition(text, count)
    host = coarse.phase_accounting(records(coarse.PREFIX), v4, enabled=True)
    layers = layer.layer_accounting(records(layer.PREFIX), host['layers'], count, enabled=True)
    focus = windows.accounting(text, layers['run'], rows, 1, enabled=True)
    exposed_s = frame['ple_wait']['future_get_ms'] / 1000
    rest = focus['other']['unclassified_wall_s'] - exposed_s
    other_layer_run = layers['run']['wall_s'] - focus['run']['wall_s']
    require(rest >= -coarse.TOLERANCE_S and other_layer_run >= -coarse.TOLERANCE_S,
            'the layer-1 split or its PLE wait does not fit the enclosing clocks')
    frame['outside_ple'] = dict(
        observed_scope_s=v4['wall_s'], caller_cpu_s=v4['cpu_s'],
        caller_cpu_inside_gpu_wait_s=v4['wait_cpu_s'], gpu_wait_s=v4['wait_wall_s'],
        caller_cpu_outside_gpu_wait_s=v4['cpu_s'] - v4['wait_cpu_s'],
        caller_unclassified_cpu_s=v4['unclassified'][0],
        phase_wall_s={name: part['wall_s'] for name, part in host.items()},
        layer_segments=layers, layer1_outside_ple_unclassified_s=max(0, rest),
        other_layers_run_wall_s=max(0, other_layer_run),
        layer1_moe_wall_s=focus['moe']['wall_s'], layer1_handoff_wall_s=focus['handoff']['wall_s'])
    return frame


def after_execution(outcome, row, plan):
    observations = [frame.pop('outside_ple') for frame in outcome['routes']]
    phase.after_execution(outcome, row, plan)
    for frame, sample, observation in zip(outcome['routes'], outcome['samples'], observations, strict=True):
        require(0 < observation['observed_scope_s'] <= sample['prompt_ms'] / 1000 + coarse.TOLERANCE_S,
                'host attribution exceeds the timed prompt')
        observation['outside_observed_scope_s'] = max(0, sample['prompt_ms'] / 1000 - observation['observed_scope_s'])
        frame['outside_ple'] = observation
    outcome['observer_warm_ratio'] = outcome['subsequent_median_ms'] / row['historical_warm_control_ms'][outcome['index']]


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                     for p in processes}) == len(ORDER)
            and all(p['complete'] and p['passed'] and p['ownership_closed']
                    and p['native_exit_code'] == 0 and not p['fresh_faults'] for p in processes),
            'require independent completed attribution runs')
    rows = []
    for p in processes:
        values = [dict(frame['outside_ple'], prompt_without_ple_s=
                       (sample['prompt_ms'] - frame['ple_wait']['future_get_ms']) / 1000)
                  for sample, frame in zip(p['samples'], p['routes'], strict=True)]
        first, *later = values
        fields = ('prompt_without_ple_s', 'caller_cpu_s', 'caller_cpu_inside_gpu_wait_s',
                  'caller_cpu_outside_gpu_wait_s', 'caller_unclassified_cpu_s', 'gpu_wait_s',
                  'outside_observed_scope_s', 'layer1_outside_ple_unclassified_s',
                  'other_layers_run_wall_s', 'layer1_moe_wall_s', 'layer1_handoff_wall_s')
        require(all(v['prompt_without_ple_s'] > 0 for v in values), 'invalid prompt remainder after removing PLE')
        rows.append(dict(index=p['index'], observer_warm_ratio=p['observer_warm_ratio'],
                         first_without_ple_s=first['prompt_without_ple_s'],
                         first_without_ple_tok_s=p['samples'][0]['actual_read_tokens'] / first['prompt_without_ple_s'],
                         first_budget_delta_s=first['prompt_without_ple_s'] - p['samples'][0]['actual_read_tokens'] / 1000,
                         first_minus_subsequent={name: first[name] - statistics.median(v[name] for v in later)
                                                for name in fields},
                         phase_deltas_s={name: value - statistics.median(v['phase_wall_s'][name] for v in later)
                                         for name, value in first['phase_wall_s'].items()}))
    return dict(processes=rows, attribution_eligible=all(.95 <= r['observer_warm_ratio'] <= 1.05 for r in rows),
                adopted=False)


def experiment():
    ctx = phase.geometry.experiment()
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER = KEY, ROOT, THIS, ORDER
    base_save, base_current, base_command = ctx.save, ctx.current, ctx.command
    payload = ROOT / 'corpus.txt'
    name = phase.capacity.WORKLOAD
    digest, paths = phase.corpus.CORPORA[name]
    selected = phase.experiment()

    def save(row):
        if row['stage'] == 'prepared':
            with payload.open('xb') as output:
                output.write(phase.corpus.corpus_payload(name))
            payload.chmod(0o400)
        base_save(row)

    def current(row):
        require(bank.sha(payload) == digest, 'the frozen corpus changed')
        return base_current(row)

    def environment(row, plan):
        env = selected.environment(row, plan)
        require(all(env.get(name, '0') == '0' for name in SELECTORS), 'unexpected inherited host observer')
        return dict(env, **dict.fromkeys(SELECTORS, '1'))

    ctx.baseline, ctx.save, ctx.current, ctx.environment = baseline, save, current, environment
    ctx.command = lambda plan: [*base_command(plan), '--prefill-corpus', str(payload)]
    ctx.request_route, ctx.after_execution, ctx.decision = route, after_execution, decision
    ctx.extra_sources = (*ctx.extra_sources, THIS, phase.THIS, phase.capacity.THIS, phase.corpus.THIS, grouped.THIS,
                         windows.THIS, layer.THIS, coarse.THIS, cpu.THIS, cpu.OBSERVER)
    ctx.selector = cpu.SELECTOR
    ctx.protocol = dict(order=list(ORDER), read=wait.READ, requests_per_process=wait.speed.RUNS,
        corpus=name, corpus_sha256=digest, corpus_revision=phase.corpus.CORPUS_REVISION, corpus_paths=list(paths),
        scope='Enable the existing disjoint caller CPU, GPU wait, layer and first-window observers on the '
              'selected cache executable with the same three prose requests.',
        decision='Require both subsequent-duration ratios within five percent of the recorded PLE-only '
                 'observer controls. Subtract exposed PLE wait once. Record the remaining first-use budget gap.',
        limits='Observer attribution on one prose extent. First-use observer cost remains unknown. '
               'The PLE-free rate is a fixed-remainder estimate. GPU wait time measures the caller.',
        stop=phase.geometry.PLAN['stop'])
    return ctx


if __name__ == '__main__':
    require(sys.executable == bank.basis.reference.model.PYTHON, 'use the pinned controller interpreter')
    wait.speed.clock.main(experiment())
