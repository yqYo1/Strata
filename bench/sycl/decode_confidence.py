"""Qualify and measure confidence-gated MTP on the same ordinary prefill and decode placement."""
import fcntl
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import ordinary_decode_round_cost as clock
import prefill_kv_gather_dispatch as dispatch

bank, basis, require = dispatch.bank, dispatch.basis, dispatch.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
PILOT = 'decode_confidence_windows_v1'
SUITE = 'decode_confidence_generation_v1'
READ = 65537
FLOOR = '0.9'
ORDER = clock.suite.ORDER
PILOT_PLAN = dict(read=READ, runs=2, threshold=FLOOR,
    compare='Same first and subsequent 65537-token bodies, ordinary copy/reload/scalar prefill. Reuse the '
            'closed same-binary fixed-width trace as control. Count actual widths, verified and accepted '
            'positions from the native trace. Require at least one shortened later window before timing.')
SPEED_PLAN = dict(order=list(ORDER), threshold=FLOOR,
    compare='Three independent 0/0.9, 0.9/0, 0/0.9 pairs, fixed stock expert layout. Official long, agent '
            'and both v5 thinking generations. Classify inputs on the reference only. Keep changed lengths, '
            'candidate acceptance, rate, time to answer, PP and TTFT in each pair. For a workload gain '
            'require shortened windows on that workload, two favorable TG pairs and geometric mean above one. '
            'No exclusions based on speed or candidate acceptance. If fewer than 256 drafts are offered, '
            'the full-suite agreement contract stops collection; keep the outcome and use the separate '
            'official speed-only workload for a throughput decision. No automatic adoption.')


def confidence_args(args):
    require('--spec-min-p' not in args and '--mtp-max-t' not in args
            and '--no-coupled-draft' in args and '--mtp' in args
            and all(args.count(f) == 1 and args[args.index(f)+1] == v for f,v in
                    (('--spec','4'),('--suffix-draft','0'),('--pcie-frac','0'),('--adapt-swaps','0'))),
            'expected the qualified fixed-width stock MTP configuration')
    return [*args, '--spec-min-p', FLOOR]


def engine_info(root, floor):
    path, = (root/'bench').glob('*/results.json')
    info = json.loads(path.read_bytes())['machine']['engine_info']
    require(all(info.get(k) == v for k,v in dict(spec=4, mtp_max=0, lookup=0, drafting='mtp',
                context=262144, kv_resident=20480, expert_slots=1256, spec_min_p=format(float(floor), '.2f')).items()),
            'native INFO does not match the declared MTP threshold or placement')


def decode_windows(text, floor=None):
    headers = re.findall(clock.footer.HEADER, text, re.M)
    require(len(headers) == 1, 'expected one native request in this trace')
    prompt, reused, read, output, accepted, offered = map(int, headers[0])
    pairs = [tuple(map(int,v)) for v in re.findall(r'^strata trace: window (\d+) (\d+)$', text, re.M)]
    require(pairs and pairs[0] == (prompt-1, 1) and prompt == READ == read and reused == 0
            and all(1 <= w <= 4 and p + w <= 262144 for p,w in pairs)
            and (floor != 0 or all(w == 4 for _,w in pairs[1:]))
            and sum(w-1 for _,w in pairs) == offered, 'invalid verification widths or offered drafts')
    # The next starting position reveals exactly how many tokens were committed.
    # At EOS or the output limit the last accepted tokens can exceed the displayed output.
    successes = [q-p-1 for (p,_),(q,_) in zip(pairs,pairs[1:])]
    successes.append(accepted-sum(successes))
    require(0 < output <= 24 and all(0 <= a < w for a,(_,w) in zip(successes,pairs,strict=True))
            and pairs[-1][0] - pairs[0][0] < output <= len(pairs)+accepted,
            'native positions or final accepted prefix do not conserve output')
    return dict(windows=len(pairs), widths=[sum(w==i for _,w in pairs) for i in range(1,5)],
                offered_by_depth=[sum(w>i for _,w in pairs) for i in range(1,4)],
                accepted_by_depth=[sum(a>=i for a in successes) for i in range(1,4)],
                generated=output, accepted=accepted, offered=offered)


def pilot():
    ctx = basis.core.source.device.load_module('decode_confidence_windows', basis.core.__file__)
    traced = dispatch.experiment()
    ctx.THIS, ctx.KEY, ctx.ROOT = THIS, PILOT, bank.B/'decode-confidence-windows-v1'
    ctx.READS, ctx.ORDER, ctx.RUNS, ctx.CLEAN_FAST = (READ,), [(READ,1)], 2, True
    ctx.check_sample, ctx.environment = traced.check_sample, traced.environment
    old_pins, old_current, old_save = ctx.pins, ctx.current, ctx.save

    def admission():
        native, prior, providers, _ = dispatch.admission()
        ref = basis.validation.committed(bank.RESULT, (dispatch.KEY,))
        require(ref['stage'] == 'closed' and ref['complete'] and ref['passed'] and ref['ownership_closed']
                and not ref['active'] and ref['native_source_commit'] == native['native_source_commit']
                and ref['binary'] == native['binaries']['strata'] and ref['plans'][0]['arm'] == 'base'
                and ref['plans'][0]['read'] == READ, 'missing same-engine ordinary width-four control')
        before = ref['processes'][0]
        require(before['passed'] and before['ownership_closed'] and not before['fresh_faults']
                and before['native_exit_code'] == 0 and len(before['routes']) == ctx.RUNS, 'control did not close')
        require([a for a in prior['argv'][1:] if a != '--serve'] == ref['configuration']['args'],
                'actual ordinary control configuration differs')
        prior['argv'] = confidence_args(prior['argv'])
        prior['observer_admission']['control_routes'] = [bank.signature(r) for r in before['routes']]
        return native, prior, providers, None

    def save(row):
        if row['stage'] == 'prepared':
            row['plans'][0].update(trace_progress=True, arm='base')
            row['plan'] = PILOT_PLAN
        old_save(row)

    def current(row):
        require(row['plan'] == PILOT_PLAN and row['plans'][0]['trace_progress']
                and row['configuration']['args'][-2:] == ['--spec-min-p', FLOOR], 'pilot plan changed')
        return old_current(row)

    def pins(revision):
        values = old_pins(revision)
        require({THIS, clock.THIS, dispatch.THIS, *dispatch.state.SCRIPTS} <= values.keys(), 'missing controller inputs')
        return values

    def route(text, plan):
        return dict(traced.request_route(text, plan), decode_windows=decode_windows(text))

    def check(out, row, plan):
        engine_info(Path(plan['root']), FLOOR)
        for i,(r,s) in enumerate(zip(out['routes'], out['samples'], strict=True)):
            w = r['decode_windows']
            require(not out['clean_speed'] and bank.signature({k:v for k,v in r.items() if k!='decode_windows'})
                    == row['observer_admission']['control_routes'][i] and
                    s['request_sha256'] == row['observer_admission']['parent_work'][str(READ)][i]['request']
                    and (w['generated'],w['accepted'],w['offered']) ==
                    (s['generated_tokens'],s['drafts_accepted'],s['drafts_offered']),
                    'request body, prefill dispatch, routes or API/native MTP counts differ')

    def decision(processes):
        p, = processes
        return dict(shortened=any(sum(r['decode_windows']['widths'][:3]) > 1 for r in p['routes']),
                    sustained_speed_tested=False)

    ctx.basis, ctx.pins, ctx.save, ctx.current = admission, pins, save, current
    ctx.request_route, ctx.after_execution, ctx.decision = route, check, decision
    return ctx


def window_bounds(sample):
    # W+A counts commits. The last window alone can have 0..3 undisplayed positions.
    generated, accepted, offered = (sample[k] for k in ('generated_tokens','drafts_accepted','drafts_offered'))
    low = max(1, generated-accepted, (offered+2)//3+1)
    high = min(generated, generated-accepted+3)
    require(0 <= accepted <= offered and low <= high, 'inconsistent MTP work')
    return dict(min_windows=low, max_windows=high, max_mean_width=1+offered/low,
                shortened_proven=offered < 3*(low-1))


def sustained():
    ctx = clock.experiment()
    ctx.THIS, ctx.KEY, ctx.ROOT, ctx.ORDER = THIS, SUITE, bank.B/'decode-confidence-generation-v1', ORDER
    ctx.protocol = SPEED_PLAN

    def baseline():
        native, speed, build, _ = dispatch.state.inputs()
        gate = basis.validation.committed(bank.RESULT, (PILOT,))
        require(gate['stage'] == 'closed' and gate['complete'] and gate['passed'] and gate['ownership_closed']
                and not gate['active'] and gate['decision']['shortened'] and gate['plan'] == PILOT_PLAN
                and gate['binary'] == native['binaries']['strata'] and gate['native_source_commit'] == native['native_source_commit']
                and gate['processes'][0]['native_exit_code'] == 0 and not gate['processes'][0]['fresh_faults'],
                'complete the same-engine traced confidence pilot first')
        common = {k:speed[k] for k in (*clock.model.FIELDS,'python_contract','python_selected','template')}
        require(confidence_args(common['configuration']['args']) == gate['configuration']['args']
                and clock.build.sha(native['binaries']['strata']['path']) == native['binaries']['strata']['sha256']
                and clock.build.sha(speed['runtime_manifest']['path']) == speed['runtime_manifest']['sha256'],
                'pilot configuration, executable or runtime differs')
        return dict(common, binary=gate['binary'], native_source_commit=gate['native_source_commit'],
                    native_build_sha256=build, ordinary_control=clock.model.ordinary_control(),
                    pilot_sha256=bank.signature(gate))

    def configuration(row, plan):
        config = clock.suite.configuration(row, plan)
        args = confidence_args(config['args'])
        return dict(config, args=args[:-1]+['0'] if plan['arm']=='original' else args)

    def environment(row,plan):
        return dict(clock.environment(row, dict(plan,arm='clean')), **{dispatch.state.KV:'0', dispatch.state.GATHER:'0'})

    def routes(log, samples, arm):
        evidence = clock.routes(log,samples,'clean')
        row = clock.build.previous.read(ctx.ROOT/'record.json')
        plan = row['plans'][len(row['processes'])]
        require(plan['arm']==arm, 'wrong confidence arm')
        engine_info(Path(plan['root']), 0 if arm=='original' else FLOOR)
        evidence['work'] = {s['workload']:window_bounds(s) for s in samples if s['phase'].startswith('long-generation/')}
        evidence['prefill_work'] = {}
        offset = 0
        for header, sample in zip(re.finditer(clock.footer.HEADER, log, re.M), samples, strict=True):
            if sample['phase'].startswith('long-generation/'):
                name = sample['workload']
                frame = log[offset:header.start()]
                evidence['prefill_work'][name] = dict(gather=dispatch.state.gather_counts(frame, False),
                    kv=dispatch.state.kv.check_clean_request(frame, 0, sample['read_tokens']-1, 8192, reuse=False))
                if arm == 'original':
                    work = evidence['work'][name]
                    offered = sample['drafts_offered']
                    require(offered % 3 == 0 and work['min_windows'] <= 1+offered//3 <= work['max_windows'],
                            'control contradicts fixed-width MTP')
            offset = header.end()
        return evidence

    def decide(processes):
        require(tuple(p['arm'] for p in processes)==ORDER and len({(p['native_identity']['pid'],
                p['native_identity']['start_ticks']) for p in processes})==len(ORDER) and all(
                p['complete'] and p['passed'] and p['ownership_closed'] and p['native_exit_code']==0
                and not p['fresh_faults'] for p in processes), 'six independent closed timings required')
        result = clock.suite.evidence_reader.decide(processes)
        pairs = [{p['arm']:p for p in processes[i:i+2]} for i in range(0,6,2)]
        def agent_history(process):
            return [s['request_sha256'] for s in process['evidence']['samples'] if s['phase']=='agent']
        result['matched_agent_requests'] = [agent_history(p['original']) == agent_history(p['candidate']) for p in pairs]
        result['decode_preservation_with_matching_counts'] &= all(result['matched_agent_requests'])
        result['workload_screen'] = {}
        for name in clock.suite.evidence_reader.SUSTAINED_WORKLOADS:
            observations = []
            for pair in pairs:
                a,b = (next(s for s in pair[k]['evidence']['samples'] if s.get('workload')==name)
                       for k in ('original','candidate'))
                require(a['request_sha256']==b['request_sha256'] and a['read_tokens']==b['read_tokens']>=32768
                        and a['reused_tokens']==b['reused_tokens']==0 and pair['original']['routes']['prefill_work'][name]
                        == pair['candidate']['routes']['prefill_work'][name], 'different long prompts or prefill dispatch')
                observations.append(dict(prompt_ratio=b['prompt_ms']/a['prompt_ms'], ttft_ratio=b['ttft_s']/a['ttft_s'],
                    answer_time_ratio=b['total_s']/a['total_s'], shortened=pair['candidate']['routes']['work'][name]['shortened_proven']))
            metric = result['decode'][name]
            label = 'strong_case_confirmed' if name == 'regular-code' else 'weak_case_confirmed'
            result['workload_screen'][name]=dict(pairs=observations, gain=result['MTP']['original'][label]
                and all(s['shortened'] for s in observations) and sum(v>1 for v in metric['pair_rate_ratios'])>=2
                and metric['geometric_mean_rate_ratio']>1)
        return result

    ctx.baseline, ctx.configuration, ctx.environment, ctx.routes = baseline, configuration, environment, routes
    original_pins = ctx.pins
    def pins(revision):
        values = original_pins(revision)
        require({THIS, clock.THIS, dispatch.THIS, *dispatch.state.SCRIPTS} <= values.keys(), 'unfrozen suite sources')
        return values
    ctx.pins = pins
    ctx.evidence_reader.decide = decide
    return ctx


if __name__ == '__main__':
    mode, action = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and mode in ('pilot','sustained'), 'use pinned Python and mode')
    if mode == 'sustained':
        sys.argv = [__file__, action]
        clock.main(sustained())
    else:
        require(action in ('prepare','host','0'), 'unplanned pilot stage')
        with (bank.B/'owned-v0141-measurement.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            bank.builder.guards.no_model_owners([])
            ctx = pilot()
            if action == 'prepare':
                ctx.prepare()
            elif action == 'host':
                basis.comparison.host(ctx)
            else:
                basis.comparison.run(0, ctx)
