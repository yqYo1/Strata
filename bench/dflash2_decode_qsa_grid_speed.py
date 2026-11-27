"""Compare the reference and bounded QSA scorers through the DFlash2 server."""

from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_decode_qsa_grid_admission as admission

ranked, bank, basis, require = admission.ranked, admission.bank, admission.basis, admission.require
primary, build = admission.primary, admission.build
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_decode_qsa_grid_speed_v1'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = ('reference', 'grid', 'grid', 'reference', 'reference', 'grid')
PLAN = dict(primary.PLAN, order=list(ORDER),
    scope='Same coherent executable, stock global expert placement and two pinned 96K thinking requests. '
          'Switch only the decode scorer before exec. Check finalized verifier captures and preserve prompt routes.',
    acceptance='Require a positive decode gain in at least two pairs and a geometric mean gain on each workload. '
               'Require the median paired prefill and TTFT ratios to be at most one for each workload. '
               'Preserve individual first/subsequent clocks and native work. A successful screen permits the next '
               'same-build physical-context, restore and quality checks. The original executable remains the fallback.')


def baseline():
    base, native, profile = admission.baseline()
    outcome = basis.validation.committed(bank.RESULT, (admission.KEY,))
    identity = bank.signature(build.builder.compact(native))
    require(build.retained.closed(outcome, 'closed') and outcome['build_sha256'] == identity
            and outcome['native_source_commit'] == native['native_source_commit']
            and all(outcome['outcomes'][arm]['passed'] and not outcome['outcomes'][arm]['fresh_faults']
                    for arm in admission.ORDER)
            and outcome['stock'] == profile, 'finish the same-build short-prompt and host controls')
    base['base_physical_context_sha256'] = base.pop('physical_context_sha256')
    return dict(base, binary=native['binaries']['strata'], native_source_commit=native['native_source_commit'],
                native_build_sha256=identity, native_context_qualified=False,
                bounded_grid_admission_sha256=bank.signature(outcome), qualified_profiles={'stock': profile})


def configuration(row, plan):
    return ranked.configuration(row, dict(plan, arm='stock'))


def environment(row, plan):
    env = primary.curve.environment(row, plan)
    require(admission.FLAG not in env, 'unexpected scorer setting before comparison selection')
    return dict(env, **{admission.FLAG: str(int(plan['arm'] == 'grid'))})


def decision(processes):
    result = ranked.decision(processes, ORDER)
    ttft_kept = all(statistics.median(item['ttft_ratio'] for item in pairs) <= 1
                    for pairs in result['pairs'].values())
    return dict(result, ttft_preserved=ttft_kept,
                eligible_for_qualification=result['eligible_for_qualification'] and ttft_kept)


def experiment():
    placement = ranked.experiment('blend')
    ctx = primary.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.protocol, ctx.configuration, ctx.environment = baseline, PLAN, configuration, environment
    ctx.evidence_reader.decide = decision

    def pins(revision):
        values = placement.pins(revision)
        for name in (THIS, admission.THIS, build.THIS):
            require((bank.P / name).read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit the exact grid comparison input: ' + name)
            values[name] = bank.sha(bank.P / name)
        return values

    def routes(text, samples, arm):
        prepared = bank.previous.read(ROOT / 'record.json')
        profile = Path(prepared['qualified_profiles']['stock']['path'])
        report = primary.routes(text, samples, 'dflash2', profile_check=lambda log:
                                ranked.restore.check_loader(log, profile, per_layer=False))
        return dict(report, score_captures=admission.score_routes(text, grid=arm == 'grid'))

    ctx.routes, ctx.pins = routes, pins
    return ctx


if __name__ == '__main__':
    primary.clock.main(experiment())
