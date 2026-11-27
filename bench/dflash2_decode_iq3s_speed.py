"""Compare decode-only IQ3_S Gate/Up against GGML on the same DFlash2 binary."""

from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_decode_iq3s_model as admission

bank, primary, ranked = admission.bank, admission.primary, admission.ranked
build, require, validation = admission.build, admission.require, admission.validation
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_decode_iq3s_speed_v1'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = ('ggml', 'iq3s', 'iq3s', 'ggml', 'ggml', 'iq3s')
PLAN = dict(
    primary.PLAN, order=list(ORDER),
    scope='Run the two pinned 96K thinking workloads in three counterbalanced pairs through bench.sh. '
          'Keep one coherent executable and the stock global 1256-slot map. Change only the decode-only '
          'IQ3_S NT1 Gate/Up selector before exec.',
    acceptance='Require higher decode throughput in at least two pairs and a positive geometric mean gain '
               'on each workload. Require median paired prompt-time and client TTFT ratios at most one '
               'on each workload. Keep every sample and preserve separate first/subsequent scopes. '
               'Record actual output, drafts, expert work and whether those matched in each pair. '
               'A pass permits physical 262144, RESTORE, agent and weak-draft qualification.'
)


def baseline():
    base = primary.retained.baseline()
    native = bank.previous.read(build.ROOT / 'record.json')
    build.check(native)
    binary = native['binaries']['strata']
    identity = bank.signature(build.builder.compact(native))
    gate = validation.committed(bank.RESULT, (admission.KEY,))
    profile = ranked.qualified_profile('stock', ranked.restore_proof(base, ranked.restore.KEY, ranked.restore.ORDER), None)
    require(primary.retained.closed(gate, 'closed') and gate['binary'] == binary
            and gate['build_sha256'] == identity and gate['native_source_commit'] == native['native_source_commit']
            and all(gate[k] == base[k] for k in ('boot', 'driver', 'runtime_manifest'))
            and gate['stock'] == profile and gate['plan'] == admission.PLAN
            and tuple(gate['outcomes']) == admission.ORDER, 'finish both exact same-binary IQ3_S route checks')
    reference, candidate = (gate['outcomes'][arm] for arm in admission.ORDER)
    require(all(o['passed'] and o['ownership_closed'] and o['native_exit_code'] == 0 and not o['fresh_faults']
                for o in (reference, candidate)) and reference['request'] == candidate['request']
            and reference['cpu_work'] == candidate['cpu_work'] and candidate['cpu_work']['iq3s_single_jobs'] > 0,
            'the bounded work, emitted IDs or draft counters differ')
    base['base_physical_context_sha256'] = base.pop('physical_context_sha256')
    return dict(base, binary=binary, native_source_commit=native['native_source_commit'],
                native_build_sha256=identity, native_context_qualified=False,
                qualified_profiles={'stock': profile}, iq3s_admission_sha256=bank.signature(gate))


def configuration(row, plan):
    return ranked.configuration(row, dict(plan, arm='stock'))


def environment(row, plan):
    env = primary.curve.environment(row, plan)
    require(env.get(admission.FLAG, '0') == '0' and 'STRATA_NO_IQ256' not in env
            and env.get('STRATA_IQ_MT_MIN', '2') == '2'
            and all(env.get(k, '0') == '0' for k in ('STRATA_CPU_SERVICE', 'STRATA_CPU_WORK_CENSUS'))
            and 'STRATA_DECODE_TIMING' not in env and 'STRATA_TRACE' not in env,
            'keep the qualified CPU dispatch and disable timing observers')
    return dict(env, **{admission.FLAG: 'iq3s' if plan['arm'] == 'iq3s' else '0'})


def routes(text, samples, arm):
    require('strata cpu work:' not in text and 'strata IQ3_GU_' not in text,
            'unexpected CPU observer during the timing run')
    profile = Path(bank.previous.read(ROOT / 'record.json')['qualified_profiles']['stock']['path'])
    return primary.routes(text, samples, 'dflash2', profile_check=lambda log:
                          ranked.restore.check_loader(log, profile, per_layer=False))


def decision(processes):
    result = ranked.decision(processes, ORDER)
    ttft_kept = all(statistics.median(p['ttft_ratio'] for p in pairs) <= 1 for pairs in result['pairs'].values())
    stable = {name: [] for name in primary.WORKLOADS}
    fields = ('generated_tokens', 'drafts_accepted', 'drafts_offered', 'output_sha256', 'reasoning_sha256')
    for first, second in zip(processes[::2], processes[1::2], strict=True):
        for name, matches in stable.items():
            a, b = (next(s for s in p['evidence']['samples'] if s.get('workload') == name)
                    for p in (first, second))
            matches.append(all(a[k] == b[k] for k in fields)
                           and first['routes'][name]['expert_work'] == second['routes'][name]['expert_work'])
    return dict(result, matched_decode_work=stable, ttft_preserved=ttft_kept,
                eligible_for_qualification=result['eligible_for_qualification'] and ttft_kept)


def experiment():
    placement = ranked.experiment('blend')
    ctx = primary.experiment()
    initial_current = ctx.current
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.protocol, ctx.configuration, ctx.environment = baseline, PLAN, configuration, environment
    ctx.routes, ctx.evidence_reader.decide = routes, decision

    def pins(revision):
        values = placement.pins(revision)
        for name in (THIS, admission.THIS, build.THIS):
            path = bank.P / name
            require(path.read_bytes() == validation.git('show', revision + ':' + name),
                    'commit the exact IQ3_S timing input: ' + name)
            values[name] = bank.sha(path)
        return values

    def current(row):
        common = []
        for case in row['plans']:
            env = environment(row, case)
            common.append({k: v.replace(case['root'], '<process>') for k, v in env.items() if k != admission.FLAG})
        require(all(env == common[0] for env in common), 'another immutable setting differs between arms')
        return initial_current(row)

    ctx.pins, ctx.current = pins, current
    return ctx


if __name__ == '__main__':
    primary.clock.main(experiment())
