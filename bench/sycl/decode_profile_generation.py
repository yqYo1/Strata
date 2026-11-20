"""Compare the two shipped decode expert rankings on the ordinary, thinking-enabled server."""

import hashlib
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import ordinary_decode_round_cost as clock
import decode_profile_restore as profile

state, suite, require = profile.state, clock.suite, clock.require
THIS = str(Path(__file__).resolve().relative_to(clock.PROJECT))
KEY = 'decode_expert_profile_generation_v1'
ROOT = clock.model.B / 'decode-expert-profile-generation-v1'
RANK = {'original': 'stock', 'candidate': 'intel'}
ORDER = suite.ORDER
PROTOCOL = dict(order=list(ORDER), ranking=RANK,
    compare='Three independent stock/Intel, Intel/stock, stock/Intel pairs. Same 1256 per-layer slots, '
            'copy input/output, scalar gather and ordinary KV reload. Change only --expert-profile. '
            'Full official long, six-turn agent and both v5 thinking workloads in every process.',
    acceptance='Report actual fresh reads, generation, MTP acceptance, decode rate, prompt time and TTFT. '
               'For both sustained workloads require the intended observed MTP classes, a higher TG in at least '
               'two pairs and geometric mean TG ratio above one. Screen agent follow-up TG and both fixed '
               'prompts for preservation. Compare agent work only for matching histories. Preserve all pairs. '
               'A rate improvement with different generated work is workload throughput only. No automatic adoption.',
    stop='Stop on an incomplete profile RESTORE gate, altered inputs or dispatch, ownership/runtime/fault failure, '
         'or an ineligible suite. No speed-based stopping, retries or replacement.')


def baseline():
    native, speed, build_hash, device_hash = state.inputs()
    gate = profile.bank.basis.validation.committed(profile.bank.RESULT, (profile.KEY,))
    binary = native['binaries']['strata']
    require(gate['stage'] == 'closed' and gate['complete'] and gate['passed'] and gate['ownership_closed']
            and not gate['active'] and gate['binary'] == binary and gate['native_source_commit'] == native['native_source_commit']
            and gate['build_sha256'] == build_hash and gate['device_sha256'] == device_hash
            and gate['plan']['profile_paths'] == profile.PROFILES and list(gate['outcomes']) == list(profile.ORDER),
            'complete both profiles on the same 32K/near-capacity state and RESTORE paths first')
    membership = {arm: set() for arm in profile.PROFILES}
    for case, done in gate['outcomes'].items():
        require(done['complete'] and done['passed'] and done['ownership_closed'] and done['native_exit_code'] == 0
                and not done['fresh_faults'] and len(done['decode_placement']) == 3, 'incomplete placement validation')
        membership[case.split('-')[1]].update(r['membership_sha256'] for r in done['decode_placement'])
    for arm, filename in profile.PROFILES.items():
        path = clock.PROJECT / filename
        require(path.read_bytes() == suite.gen.git('show', gate['source_commit'] + ':' + filename),
                'the ranking file changed since the state and RESTORE test')
        masks = profile.census.expected_membership(path)
        digest = hashlib.sha256(b''.join(mask.to_bytes(64, 'little') for mask in masks)).hexdigest()
        require(membership[arm] == {digest}, 'generation and RESTORE membership differ')
    require(membership['intel'] != membership['stock']
            and clock.build.sha(binary['path']) == binary['sha256']
            and clock.build.sha(speed['runtime_manifest']['path']) == speed['runtime_manifest']['sha256'],
            'actual profile membership, executable or runtime differs')
    fixed = {k: speed[k] for k in (*clock.model.FIELDS, 'python_contract', 'python_selected', 'template')}
    require(all(gate[k] == fixed[k] for k in ('configuration', 'native_validation_probe', 'runtime_manifest', 'boot', 'driver'))
            and gate['model_assets_sha256'] == clock.build.signature(fixed['model_assets'])
            and gate['environment_template_sha256'] == clock.build.signature(fixed['environment_template']),
            'official suite and restored placement must use the same base')
    return dict(fixed, binary=binary, native_source_commit=native['native_source_commit'], native_build_sha256=build_hash,
                ordinary_control=clock.model.ordinary_control(), restore_sha256=clock.build.signature(gate))


def configuration(row, plan):
    config = suite.configuration(row, plan)
    args = list(config['args'])
    require(args.count('--expert-cache-per-layer') == 1, 'per-layer layout required')
    for flag, expected in (('--expert-profile', str(clock.PROJECT / profile.PROFILES['stock'])),
                           ('--expert-cache', '1256'), ('--max-context', '262144'), ('--kv-resident', '20480'),
                           ('--spec', '4'), ('--prefill', '8192'), ('--pcie-frac', '0'), ('--adapt-swaps', '0')):
        require(args.count(flag) == 1 and args[args.index(flag)+1] == expected, 'unexpected fixed option ' + flag)
    args[args.index('--expert-profile')+1] = str(clock.PROJECT / profile.PROFILES[RANK[plan['arm']]])
    return dict(config, args=args)


def environment(row, plan):
    env = clock.environment(row, dict(plan, arm='clean'))
    require(env[clock.model.INPUT] == env[clock.model.OUTPUT] == '0'
            and all(env.get(key, '0') == '0' for key in ('STRATA_CPU_WORK_CENSUS', 'STRATA_SYCL_CACHE_LOAN_AUDIT')),
            'clean ordinary copy placement required')
    return dict(env, **{state.KV: '0', state.GATHER: '0'})


def check_profile_startup(log, arm):
    path = clock.PROJECT / profile.PROFILES[RANK[arm]]
    require(log.splitlines().count(f'strata generate: profile {path}: 24576 ranked pairs, built for 24576 slots') == 1
            and log.count('strata generate: profile ') == 1
            and log.count('pre-filled 1256 of 1256 slots from the profile; slot 0 verified') == 1
            and log.count('R4.2g PER-LAYER: each layer owns 26 slots (0..25).') == 1,
            'actual startup ranking or per-layer capacity differs')


def routes(log, samples, arm):
    check_profile_startup(log, arm)
    require(profile.census.PREFIX not in log and 'strata loan audit:' not in log, 'unexpected native observer')
    result = clock.routes(log, samples, 'clean')
    heads = list(re.finditer(clock.footer.HEADER, log, re.M))
    result['prefill_dispatch'] = {}
    for i, (head, sample) in enumerate(zip(heads, samples, strict=True)):
        if sample['phase'].startswith('long-generation/'):
            frame = log[heads[i-1].end() if i else 0:head.start()]
            rows = sample['read_tokens'] - 1
            require(sample['reused_tokens'] == 0 and rows >= 32767, 'fixed long prompt must be fresh')
            result['prefill_dispatch'][sample['workload']] = dict(
                gather=state.gather_counts(frame, False), kv=state.kv.check_clean_request(frame, 0, rows, 8192, reuse=False))
    previous = clock.build.previous.read(ROOT / 'record.json')['processes']
    if previous:
        require(result['prefill_dispatch'] == previous[0]['routes']['prefill_dispatch'],
                'fixed-input prefill dispatch differs from the ordinary ranking')
    return result


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER and len({
        (p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ORDER)
        and all(p['passed'] and p['complete'] and p['ownership_closed'] and p['native_exit_code'] == 0
                and not p['fresh_faults'] for p in processes), 'six independent completed processes required')
    result = suite.evidence_reader.decide(processes)
    by_metric = {name: [] for name in suite.evidence_reader.SUSTAINED_WORKLOADS}
    agent_histories = []
    for a, b in zip(processes[::2], processes[1::2], strict=True):
        reference, candidate = (a, b) if a['arm'] == 'original' else (b, a)
        left, right = (p['evidence']['samples'] for p in (reference, candidate))
        agent_histories.append([s['request_sha256'] for s in left if s['phase'] == 'agent'] ==
                               [s['request_sha256'] for s in right if s['phase'] == 'agent'])
        for name in by_metric:
            x, y = [next(s for s in samples if s.get('workload') == name) for samples in (left, right)]
            require(x['request_sha256'] == y['request_sha256']
                    and x['read_tokens'] == y['read_tokens'] >= 32768 and x['reused_tokens'] == y['reused_tokens'] == 0
                    and min(x['generated_tokens'], y['generated_tokens']) >= 1024 and x['controller_eligible'] and y['controller_eligible']
                    and reference['routes']['prefill_dispatch'][name] == candidate['routes']['prefill_dispatch'][name],
                    'fixed prompt, prefill dispatch or sustained completion differs')
            by_metric[name].append(dict(prompt_time_ratio=y['prompt_ms']/x['prompt_ms'], ttft_ratio=y['ttft_s']/x['ttft_s']))
    result['fixed_prompt_pairs'] = by_metric
    result['matched_agent_request_bodies'] = agent_histories
    result['decode_preservation_with_matching_counts'] &= all(agent_histories)
    result['prefill_preservation_screen'] = all(statistics.median(p['prompt_time_ratio'] for p in pairs) <= 1
                                                for pairs in by_metric.values())
    result['sustained_gain_screen'] = result['MTP_pair_qualified'] and all(
        sum(ratio > 1 for ratio in result['decode'][name]['pair_rate_ratios']) >= 2
        and result['decode'][name]['geometric_mean_rate_ratio'] > 1 for name in by_metric)
    return result


def experiment():
    ctx = clock.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.environment, ctx.configuration, ctx.routes = baseline, environment, configuration, routes
    ctx.protocol, ctx.evidence_reader.decide = PROTOCOL, decision
    base_pins = ctx.pins

    def pins(revision):
        values = base_pins(revision)
        for path in profile.PROFILES.values():
            data = suite.gen.git('show', revision + ':' + path)
            require((clock.PROJECT / path).read_bytes() == data, 'shipped ranking differs from committed bytes')
            values[path] = clock.build.sha(clock.PROJECT / path)
        require({THIS, clock.THIS, *profile.SOURCES} <= values.keys(), 'missing profile or transitive controller')
        return values

    ctx.pins = pins
    return ctx


if __name__ == '__main__':
    clock.main(experiment())
