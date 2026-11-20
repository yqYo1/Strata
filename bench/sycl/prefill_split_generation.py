"""Check sustained decode preservation after selecting a long-prefill KV/gather path."""

from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import decode_profile_generation as profiles
import prefill_split_length_curve as curve

clock, suite, state, require = profiles.clock, profiles.suite, profiles.state, profiles.require
THIS = str(Path(__file__).resolve().relative_to(clock.PROJECT))
KEY = 'prefill_split_generation_v1'
ROOT = clock.model.B / 'prefill-split-generation-v1'


def baseline():
    native, prior, _, _ = curve.admission()
    binary = native['binaries']['strata']
    speed = clock.build.previous.read(clock.model.basis.reference.ROOT / 'record.json')
    fixed = {k: speed[k] for k in (*clock.model.FIELDS, 'template', 'python_contract', 'python_selected')}
    require(prior['observer_admission']['curve_arm'] != 'base'
            and clock.build.sha(binary['path']) == binary['sha256']
            and clock.build.sha(fixed['runtime_manifest']['path']) == fixed['runtime_manifest']['sha256'],
            'need a measured faster prefill arm and the qualified executable and runtime')
    return dict(fixed, binary=binary, native_source_commit=native['native_source_commit'],
                selected_arm=prior['observer_admission']['curve_arm'],
                timing_sha256=prior['observer_admission']['timing_result_sha256'],
                ordinary_control=clock.model.ordinary_control())


def experiment():
    ctx = clock.experiment()
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER = KEY, ROOT, THIS, suite.ORDER
    ctx.baseline = baseline
    ctx.protocol = dict(order=list(ctx.ORDER), compare='Three AB, BA, AB independent-process pairs on the '
        'ordinary stock layout. Change only the winning 117489-token prefill dispatch. Full official long, '
        'agent and two thinking-enabled sustained workloads. No early speed stop or replacement.',
        decision='Apply official-suite matched-work decode preservation per workload, require identical agent '
        'request histories and the observed strong/weak MTP classes. Report fixed-body prompt and TTFT ratios, '
        'keep individual paired TG and acceptance. Stop on any wrong dispatch, incomplete run or fault; no automatic adoption.')

    def environment(row, plan):
        selected = row['selected_arm'] if plan['arm'] == 'candidate' else 'base'
        kv, vec8 = state.ARMS[selected]
        env = profiles.environment(row, plan)
        return dict(env, **{state.KV: str(int(kv)), state.GATHER: str(int(vec8))})

    def routes(text, samples, arm):
        row = clock.build.previous.read(ROOT / 'record.json')
        selected = row['selected_arm'] if arm == 'candidate' else 'base'
        kv, vec8 = state.ARMS[selected]
        profiles.check_profile_startup(text, 'original')
        result = clock.routes(text, samples, 'clean')
        headers = list(re.finditer(clock.footer.HEADER, text, re.M))
        result['fixed_work'] = {}
        for i, (header, sample) in enumerate(zip(headers, samples, strict=True)):
            if not sample['phase'].startswith('long-generation/'):
                continue
            frame = text[headers[i-1].end() if i else 0:header.start()]
            count = sample['read_tokens'] - 1
            require(count >= 32767 and sample['reused_tokens'] == 0, 'expected a fixed fresh long prompt')
            copies = state.kv.check_clean_request(frame, 0, count, 8192, reuse=kv)
            result['fixed_work'][sample['workload']] = dict(gather=state.gather_counts(frame, vec8),
                prefix_pages=copies['reloaded_pages'] + copies['reused_pages'])
        return result

    def decision(processes):
        require(tuple(p['arm'] for p in processes) == ctx.ORDER and len({
                (p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == 6
                and all(p['complete'] and p['passed'] and p['native_exit_code'] == 0 and p['ownership_closed']
                        and not p['fresh_faults'] for p in processes), 'six independent closed processes required')
        result = suite.evidence_reader.decide(processes)
        matched = []
        ratios = {n: [] for n in suite.evidence_reader.SUSTAINED_WORKLOADS}
        for a, b in zip(processes[::2], processes[1::2], strict=True):
            base, candidate = (a, b) if a['arm'] == 'original' else (b, a)
            require(base['routes']['fixed_work'] == candidate['routes']['fixed_work']
                    and set(base['routes']['fixed_work']) == set(ratios), 'fixed prefill work differs')
            left, right = base['evidence']['samples'], candidate['evidence']['samples']
            matched.append([s['request_sha256'] for s in left if s['phase'] == 'agent']
                           == [s['request_sha256'] for s in right if s['phase'] == 'agent'])
            for name in ratios:
                x, y = (next(s for s in samples if s.get('workload') == name) for samples in (left, right))
                require(x['request_sha256'] == y['request_sha256'] and x['read_tokens'] == y['read_tokens']
                        and min(x['generated_tokens'], y['generated_tokens']) >= 1024
                        and x['controller_eligible'] and y['controller_eligible'], 'ineligible sustained pair')
                ratios[name].append(dict(prompt=y['prompt_ms']/x['prompt_ms'], ttft=y['ttft_s']/x['ttft_s']))
        result.update(fixed_prompt_time_ratios=ratios, matched_agent_bodies=matched,
                      selected_prefill_preserves_decode=result['decode_preservation_with_matching_counts']
                      and all(matched) and result['MTP_pair_qualified'], adopted=False)
        return result

    ctx.environment, ctx.routes, ctx.evidence_reader.decide = environment, routes, decision
    ordinary_pins = ctx.pins

    def pins(revision):
        values = ordinary_pins(revision)
        paths = {THIS, clock.THIS, curve.THIS, curve.speed.THIS, profiles.THIS,
                 *profiles.profile.SOURCES, curve.speed.capacity.THIS, curve.speed.dispatch.THIS,
                 curve.speed.dispatch.split.THIS, curve.speed.dispatch.split.bank.THIS}
        for path in paths:
            data = suite.gen.git('show', revision + ':' + path)
            require((clock.PROJECT / path).read_bytes() == data, 'changed suite input: ' + path)
            values[path] = clock.build.sha(clock.PROJECT / path)
        return values
    ctx.pins = pins
    return ctx


if __name__ == '__main__':
    clock.main(experiment())
