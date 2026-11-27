"""Compare width-three KV through the fixed 96K DFlash2 bench.sh workloads."""

from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_decode_kv_speed as previous
import dflash2_decode_kv_width3_admission as admission
import dflash2_decode_qsa_grid_speed as scorer

bank, basis, require = admission.bank, admission.basis, admission.require
primary, ranked = admission.primary, admission.ranked
kv, build = admission.kv, admission.build
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_decode_kv_width3_speed_v1'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = previous.ORDER
PLAN = dict(previous.PLAN,
            scope='One coherent DFlash2 executable, stock expert placement and scalar QSA. '
                  'Switch only the width-three KV selector before exec. Keep independent prompt processing.',
            coverage='Both 96K workloads must execute fresh long prefill and sustained generation. '
                     'Bind each workload to its archived request and show width three and scalar fallback '
                     'in its own finalized capture interval. Check scalar verifier prompt captures in the '
                     'short warm-up and use the layer-major route checks for the long prompts.')


def baseline():
    base = build.grid.retained.baseline()
    native = bank.previous.read(build.ROOT / 'record.json')
    build.check(native)
    proof = basis.validation.committed(bank.RESULT, (admission.KEY,))
    identity = bank.signature(build.builder.compact(native))
    require(build.grid.retained.closed(proof, 'closed') and proof['build_sha256'] == identity
            and proof['binary'] == native['binaries']['strata']
            and proof['plan'] == admission.PLAN and list(proof['outcomes']) == list(admission.ORDER)
            and all(p['passed'] and p['ownership_closed'] and not p['fresh_faults'] for p in proof['outcomes'].values()),
            'require the same executable and both width-three admission arms')
    profile = proof['stock']
    require(bank.sha(profile['path']) == profile['sha256'], 'the admitted stock expert profile changed')
    original = basis.validation.committed(bank.RESULT, (previous.KEY,))
    require(build.grid.retained.closed(original, 'closed'), 'require the original closed 96K cohort')
    bodies = {}
    for name in primary.WORKLOADS:
        samples = [s for p in original['processes'] for s in p['evidence']['samples'] if s.get('workload') == name]
        (wire, read), = {(s['request_sha256'], s['read_tokens']) for s in samples}
        require(len(ORDER) == len(samples) and all(s['reused_tokens'] == 0 for s in samples),
                'the earlier 96K workload is not an exact fresh control')
        bodies[name] = dict(request_sha256=wire, read_tokens=read)
    base['base_physical_context_sha256'] = base.pop('physical_context_sha256')
    return dict(base, binary=native['binaries']['strata'], native_source_commit=native['native_source_commit'],
                native_build_sha256=identity, native_context_qualified=False,
                width_three_admission_sha256=bank.signature(proof), qualified_profiles={'stock': profile},
                archived_long_requests=bodies)


def environment(row, plan):
    env = scorer.environment(row, dict(plan, arm='reference'))
    require(kv.FLAG not in env and env.get('STRATA_SYCL_KV_COPY_GROUPS') == '0',
            'unexpected inherited KV resolver selector')
    return dict(env, **{kv.FLAG: '3' if plan['arm'] == 'cooperative' else '0'})


def routes(text, samples, arm):
    row = bank.previous.read(ROOT / 'record.json')
    profile = Path(row['qualified_profiles']['stock']['path'])
    report = primary.routes(text, samples, 'dflash2', profile_check=lambda log:
                            ranked.restore.check_loader(log, profile, per_layer=False))
    all_calls = kv.kv_routes(text, cooperative=arm == 'cooperative', widths=(3,))
    require(any(r['phase'] == 'prompt' for r in all_calls), 'require warm-up verifier prompt coverage')
    headers = list(re.finditer(primary.footer.HEADER, text, re.M))
    captures = {}
    for i, (header, sample) in enumerate(zip(headers, samples, strict=True)):
        if 'workload' not in sample:
            continue
        name = sample['workload']
        require(name not in captures
                and {k: sample[k] for k in ('request_sha256', 'read_tokens')} == row['archived_long_requests'][name],
                'the comparison request differs from the archived 96K body')
        start = headers[i - 1].end() if i else 0
        calls = kv.kv_routes(text[start:header.start()], cooperative=arm == 'cooperative', widths=(3,))
        require(any(r['phase'] == 'decode' and r['width'] != 3 for r in calls),
                'the measured long workload lacks a scalar decode fallback capture')
        captures[name] = calls
    require(tuple(captures) == primary.WORKLOADS, 'require resolver coverage in both measured workloads')
    return dict(report, kv_captures=captures)


def experiment():
    ctx = scorer.experiment()
    inherited_pins = ctx.pins
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.protocol, ctx.environment, ctx.routes = baseline, PLAN, environment, routes
    ctx.evidence_reader.decide = previous.decision

    def pins(revision):
        values = inherited_pins(revision)
        for name in (THIS, previous.THIS, admission.THIS, kv.THIS, build.THIS):
            path = bank.P / name
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit the exact width-three comparison input: ' + name)
            values[name] = bank.sha(path)
        return values

    ctx.pins = pins
    return ctx


if __name__ == '__main__':
    primary.clock.main(experiment())
