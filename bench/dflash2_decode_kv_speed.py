"""Compare scalar and cooperative KV resolution on the same DFlash2 executable."""

from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_decode_kv_admission as kv
import dflash2_decode_qsa_grid_speed as scorer

ranked, bank, basis, require = kv.ranked, kv.bank, kv.basis, kv.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_decode_kv_speed_v1'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = ('scalar', 'cooperative', 'cooperative', 'scalar', 'scalar', 'cooperative')
PLAN = dict(scorer.PLAN, order=list(ORDER),
            scope='Same DFlash2 executable, stock placement, scalar QSA scorer and exact 96K thinking workloads. '
                  'Switch only cooperative KV before exec and check finalized prompt/decode captures.',
            acceptance='Require decode gains in at least two pairs and a positive geometric mean gain for both workloads. '
                       'Require median paired prefill and TTFT ratios at most one in each workload. '
                       'Keep first/subsequent clocks and recorded native work. A pass permits dedicated physical-context, '
                       'RESTORE and model quality checks.')


def baseline():
    base = scorer.baseline()
    admission = basis.validation.committed(bank.RESULT, (kv.KEY,))
    require(kv.build.retained.closed(admission, 'closed') and admission['binary'] == base['binary']
            and admission['build_sha256'] == base['native_build_sha256']
            and admission['scalar_qsa_admission_sha256'] == base['bounded_grid_admission_sha256']
            and list(admission['outcomes']) == list(kv.ORDER)
            and all(o['passed'] and not o['fresh_faults'] for o in admission['outcomes'].values())
            and admission['stock'] == base['qualified_profiles']['stock'],
            'finish both same-executable short cooperative KV controls')
    return dict(base, cooperative_kv_admission_sha256=bank.signature(admission))


def environment(row, plan):
    env = scorer.environment(row, dict(plan, arm='reference'))
    require(kv.FLAG not in env and env.get('STRATA_SYCL_KV_COPY_GROUPS') == '0', 'unexpected inherited KV selector')
    return dict(env, **{kv.FLAG: str(int(plan['arm'] == 'cooperative'))})


def decision(processes):
    result = ranked.decision(processes, ORDER)
    ttft_kept = all(statistics.median(p['ttft_ratio'] for p in pairs) <= 1 for pairs in result['pairs'].values())
    return dict(result, ttft_preserved=ttft_kept,
                eligible_for_qualification=result['eligible_for_qualification'] and ttft_kept)


def experiment():
    ctx = scorer.experiment()
    original_pins = ctx.pins
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.protocol, ctx.environment = baseline, PLAN, environment
    ctx.evidence_reader.decide = decision

    def pins(revision):
        values = original_pins(revision)
        for name in (THIS, kv.THIS):
            path = bank.P / name
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit the exact cooperative KV comparison input: ' + name)
            values[name] = bank.sha(path)
        return values

    def routes(text, samples, arm):
        prepared = bank.previous.read(ROOT / 'record.json')
        profile = Path(prepared['qualified_profiles']['stock']['path'])
        report = kv.primary.routes(text, samples, 'dflash2', profile_check=lambda log:
                                  ranked.restore.check_loader(log, profile, per_layer=False))
        return dict(report, kv_captures=kv.kv_routes(text, cooperative=arm == 'cooperative'))

    ctx.routes, ctx.pins = routes, pins
    return ctx


if __name__ == '__main__':
    kv.primary.clock.main(experiment())
