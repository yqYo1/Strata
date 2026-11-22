"""Measure the committed-prefix MTP selector in the official thinking suite."""

from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import decode_mtp_prefix_restore as handoff
import decode_profile_generation as profiles

clock, suite, require = profiles.clock, profiles.suite, profiles.require
bank, basis = handoff.bank, handoff.bank.basis
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'decode_mtp_prefix_generation_v1'
ROOT = bank.B / 'decode-mtp-prefix-generation-v1'
ORDER = suite.ORDER
WORKLOADS = suite.evidence_reader.SUSTAINED_WORKLOADS
SELECTOR = handoff.physical.SELECTOR
PROTOCOL = dict(order=list(ORDER),
    comparison='Three off/on, on/off, off/on pairs of independent processes with the fixed width-four engine. '
               'Keep the stock expert ranking, ordinary prefill and uncoupled MTP. Change the prefix selector '
               'before exec. Run the official long prompt, six-turn agent and both v5 thinking workloads.',
    acceptance='Require matched fresh prompt bodies, completed thinking outputs, qualified strong/weak MTP '
               'and at least two faster pairs with geometric mean TG ratio above one per sustained workload. '
               'Report the work counts and full request time. Compare fixed-input PP and client TTFT separately. '
               'Report agent decode only as matched work when both histories and counters match. Dedicated '
               'prefill preservation and broader quality/qualification remain separate gates.',
    stop='Stop on changed inputs/dispatch, a failed request or a runtime/ownership violation. Keep every '
         'sample. No early speed stops, replacement processes or automatic adoption.')


def baseline():
    native, speed, build_sha, physical_sha = handoff.inputs()
    gate = basis.validation.committed(bank.RESULT, (handoff.KEY,))
    require(gate['stage'] == 'closed' and gate['complete'] and gate['passed']
            and gate['ownership_closed'] and not gate['active'] and gate['plan'] == handoff.PLAN
            and gate['binary'] == native['binaries']['strata']
            and gate['native_source_commit'] == native['native_source_commit']
            and gate['build_sha256'] == build_sha and gate['device_sha256'] == physical_sha
            and list(gate['outcomes']) == list(handoff.ORDER)
            and all(o['complete'] and o['passed'] and o['ownership_closed']
                    and o['native_exit_code'] == 0 and not o['fresh_faults'] for o in gate['outcomes'].values()),
            'complete the short and near-capacity prefix RESTORE comparisons first')
    for size in ('short', 'near'):
        off, on = (gate['outcomes'][size + '-' + arm] for arm in ('off', 'on'))
        require(sum(w['catchup_rows'] for w in on['verification'])
                < sum(w['catchup_rows'] for w in off['verification']), 'missing live prefix shortening')
    reference = bank.previous.read(basis.reference.ROOT / 'record.json')
    require(all(speed[k] == reference[k] for k in clock.model.FIELDS)
            and all(gate[k] == speed[k] for k in ('configuration', 'runtime_manifest', 'native_validation_probe',
                                                  'boot', 'driver'))
            and all(gate[k + '_sha256'] == bank.signature(speed[k])
                    for k in ('environment_template', 'model_assets')), 'frozen suite base changed')
    fixed = {k: reference[k] for k in (*clock.model.FIELDS, 'template', 'python_contract', 'python_selected')}
    return dict(fixed, binary=native['binaries']['strata'], native_source_commit=native['native_source_commit'],
                native_build_sha256=build_sha, restore_sha256=bank.signature(gate),
                ordinary_control=clock.model.ordinary_control())


def configuration(row, plan):
    config = profiles.configuration(row, dict(plan, arm='original'))
    args = config['args']
    require(all(args.count(a) == 1 for a in ('--mtp', '--no-coupled-draft', '--no-spec-split'))
            and args.count('--suffix-draft') == 1 and args[args.index('--suffix-draft') + 1] == '0'
            and not any(a in args for a in ('--mtp-max-t', '--spec-min-p', '--draft-model')),
            'require ordinary uncoupled MTP with no competing drafter')
    return dict(config, args=[*args, '--mtp-max-t', '4'])


def environment(row, plan):
    env = profiles.environment(row, plan)
    require(SELECTOR not in env and env['STRATA_SYCL_DECODE_Q6_DPAS_HEAD'] == '2',
            'require the tested width-four head and an independent selector')
    return dict(env, **{SELECTOR: str(int(plan['arm'] == 'candidate'))})


def omission_bounds(sample):
    offered, accepted = (sample[k] for k in ('drafts_offered', 'drafts_accepted'))
    require(type(offered) is int and type(accepted) is int and 0 <= accepted <= offered,
            'invalid native draft counts')
    # Each rejected proposal is one surplus catch-up row in the stock route.
    # The last verifier window may skip its draft call and account for at most three.
    rejected = offered - accepted
    return [max(0, rejected - 3), rejected]


def routes(log, samples, _arm):
    profiles.check_profile_startup(log, 'original')
    require(profiles.profile.census.PREFIX not in log and 'strata loan audit:' not in log
            and 'strata mtp: decode catchup' not in log, 'unexpected timed observer')
    result = clock.routes(log, samples, 'clean')
    heads = list(re.finditer(clock.footer.HEADER, log, re.M))
    result.update(prefill_dispatch={}, prefix_rows_eligible={}, mtp_recapture={})
    for i, (head, s) in enumerate(zip(heads, samples, strict=True)):
        if not s['phase'].startswith('long-generation/'):
            continue
        frame = log[heads[i-1].end() if i else 0:head.start()]
        read, name = s['read_tokens'], s['workload']
        bounds = omission_bounds(s)
        require(read >= 32768 and s['reused_tokens'] == 0 and bounds[0] > 0,
                'sustained workload must have fresh input and rejected nonterminal drafts')
        result['prefill_dispatch'][name] = dict(
            kv=profiles.state.kv.check_clean_request(frame, 0, read-1, 8192, reuse=False),
            gather=profiles.state.gather_counts(frame, False))
        result['prefix_rows_eligible'][name] = bounds
        captures = re.findall(r'^strata mtp: expert decode recapture kind=(round|step) '
                              r'coupled=0 ok=1 ms=([0-9.]+) cumulative_ms=[0-9.]+$', frame, re.M)
        timings = {}
        for kind in ('round', 'step'):
            durations = [float(value) for label, value in captures if label == kind]
            timings[kind] = dict(events=len(durations), total_ms=sum(durations))
        result['mtp_recapture'][name] = timings
    require(set(result['prefill_dispatch']) == set(WORKLOADS), 'incomplete sustained route evidence')
    previous = bank.previous.read(ROOT / 'record.json')['processes']
    if previous:
        require(result['prefill_dispatch'] == previous[0]['routes']['prefill_dispatch'],
                'fixed-input prefill dispatch differs from the first process')
    return result


def decision(processes):
    result = profiles.decision(processes)
    for pair_index, (a, b) in enumerate(zip(processes[::2], processes[1::2], strict=True)):
        reference, candidate = (a, b) if a['arm'] == 'original' else (b, a)
        for name in WORKLOADS:
            old, new = (next(s for s in p['evidence']['samples'] if s.get('workload') == name)
                        for p in (reference, candidate))
            result['fixed_prompt_pairs'][name][pair_index]['request_time_ratio'] = new['total_s'] / old['total_s']
    for name in WORKLOADS:
        reported = result['decode'][name]
        reported['faster_in_two_pairs'] = sum(r > 1 for r in reported['pair_rate_ratios']) >= 2
        reported['workload_gain_screen'] = (result['MTP_pair_qualified'] and reported['faster_in_two_pairs']
                                            and reported['geometric_mean_rate_ratio'] > 1)
        reported['matched_work'] = all(result['matching_work_counts_by_metric'][name])
    result['fixed_prompt_ttft_preserved'] = all(statistics.median(p['ttft_ratio'] for p in pairs) <= 1
                                                for pairs in result['fixed_prompt_pairs'].values())
    result['agent_matched_history'] = all(result['matched_agent_request_bodies'])
    return result


def experiment():
    ctx = clock.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.configuration, ctx.environment, ctx.routes = baseline, configuration, environment, routes
    ctx.protocol, ctx.evidence_reader.decide = PROTOCOL, decision

    def pins(revision):
        values = suite.pins(revision)
        required = (THIS, clock.THIS, profiles.THIS, handoff.THIS, handoff.physical.THIS,
                    profiles.profile.PROFILES['stock'], *profiles.profile.SOURCES)
        for name in required:
            require((bank.P / name).read_bytes() == suite.gen.git('show', revision + ':' + name),
                    'changed prefix speed input: ' + name)
            values[name] = bank.sha(bank.P / name)
        return values

    ctx.pins = pins
    return ctx


if __name__ == '__main__':
    clock.main(experiment())
