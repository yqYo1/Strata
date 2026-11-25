"""Measure a single width-five Q6 dense projection in sustained DFlash2 decode."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True
import dflash2_c5_dense_probe as probe
import dflash2_c5_head_generation as head

bank, basis, require, width = head.bank, head.basis, head.require, head.width
KEY = 'dflash2_c5_dense_generation_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ORDER = ('reference', 'dense', 'dense', 'reference', 'reference', 'dense')
PROTOCOL = dict(head.PROTOCOL, order=list(ORDER),
    only_factor='Keep the width-five head enabled in both arms. Change only the exact decode Q6 '
                '2560x10240 dense selector in one executable.',
    decision='Advance when all three ordered pairs have matched verifier and expert work and each '
             'candidate decodes faster. Width four remains the operational reference. Measure weak '
             'draft acceptance, agent sessions, full context and RESTORE separately.')


def admission():
    outcome = basis.validation.committed(bank.RESULT, (probe.KEY,))
    require(outcome['stage'] == 'closed' and outcome['complete'] and outcome['passed']
            and outcome['ownership_closed'] and tuple(outcome['outcomes']) == probe.ORDER
            and all(r['passed'] and r['native_exit_code'] == 0 and r['ownership_closed'] and not r['fresh_faults']
                    for r in outcome['outcomes'].values())
            and [r['dense_dpas_calls'] for r in outcome['outcomes'].values()] == [0, probe.CALLS],
            'complete both same-build dense route probes first')
    return outcome


def baseline():
    row = width.baseline()
    gate = admission()
    compiled = bank.previous.read(probe.build.ROOT / 'record.json')
    probe.build.context().check(compiled)
    plan = gate['plan']
    require(plan['binary'] == compiled['binaries']['strata']
            and plan['native_source_commit'] == compiled['native_source_commit']
            and plan['native_build_sha256'] == bank.signature(probe.build.builder.compact(compiled))
            and plan['boot'] == row['boot'] and plan['driver'] == row['driver']
            and plan['runtime_manifest'] == row['runtime_manifest']
            and plan['model_assets_sha256'] == bank.signature(row['model_assets'])
            and plan['drafter_assets'] == row['dflash2_assets'], 'dense model probes and long run must bind the same engine')
    arm = {k: plan[k] for k in ('binary', 'native_source_commit', 'native_build_sha256')}
    return dict(row, c5_dense_arm=arm, dense_probe_sha256=bank.signature(gate),
                head_comparison=probe.build.HEAD_EVIDENCE)


def environment(row, plan):
    env = width.environment(row, dict(plan, arm='5'))
    require(not {probe.FLAG, probe.head.FLAG}.intersection(env) and plan['arm'] in probe.ORDER,
            'resolve the comparison switches from the frozen arm')
    return dict(env, **{probe.head.FLAG: '1', probe.FLAG: str(int(plan['arm'] == 'dense'))})


def routes(text, samples, arm):
    summary = width.routes(text, samples, '5')
    counts = [dict(reference=int(ref), columns=int(cols)) for ref, cols in probe.head.DECODE_C5.findall(text)]
    expected = admission()['outcomes'][arm]['capture_counts']
    require(counts and all(c == expected for c in counts), 'the real width-five graph differs from the audited dense route')
    return dict(summary, dense_capture_counts=expected, verified_dense_captures=len(counts))


def decision(processes):
    alias = [dict(p, arm='c5' if p['arm'] == 'dense' else p['arm']) for p in processes]
    measured = head.decision(alias)
    for field in ('TG', 'TG_median'):
        measured[field]['dense'] = measured[field].pop('c5')
    measured['dense_gain_qualified'] = measured.pop('head_gain_qualified')
    return measured


def experiment():
    ctx = width.leaves.experiment()
    run = ctx.run
    inherited_pins = head.experiment().pins
    ctx.KEY, ctx.ROOT, ctx.THIS, ctx.ORDER = KEY, ROOT, THIS, ORDER
    ctx.baseline, ctx.configuration, ctx.environment = baseline, head.configuration, environment
    ctx.routes, ctx.protocol = routes, PROTOCOL
    ctx.evidence_reader.read_complete_suite, ctx.evidence_reader.decide = width.read_suite, decision

    def native_context(row, plan):
        require(plan['arm'] == ORDER[plan['index']], 'the counterbalanced dense order changed')
        return dict(row, **row['c5_dense_arm'])

    def pins(revision):
        files = inherited_pins(revision)
        for name in (THIS, probe.THIS, probe.build.THIS):
            path = bank.P / name
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'changed dense measurement input: ' + name)
            files[name] = bank.sha(path)
        return files

    ctx.pins, ctx.native_context = pins, native_context
    ctx.run = lambda index: run(index, native_context=native_context)
    return ctx


if __name__ == '__main__':
    head.snapshot.clock.main(experiment())
