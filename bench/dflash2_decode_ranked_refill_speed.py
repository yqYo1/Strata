"""Measure final-only DFlash2 expert refill after the native safety gates."""

from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_decode_ranked_generation as ranked
import dflash2_decode_ranked_refill_context as capacity
import dflash2_decode_ranked_refill_restore as refill

primary, bank, basis, require = ranked.primary, ranked.bank, ranked.basis, ranked.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_decode_ranked_refill_speed_v1'
ROOT = bank.B / KEY.replace('_', '-')
ORDER = ('stock', 'refill', 'refill', 'stock', 'stock', 'refill')
PLAN = dict(primary.PLAN, order=list(ORDER),
    scope='Use one qualified binary and stock global startup placement. Enable only final ranked refill in the '
          'candidate. Measure the pinned 96K code and prose requests with thinking in three matched process pairs.',
    acceptance='Keep actual work and dispatch matched. Require a decode gain in at least two pairs and a '
               'positive geometric mean gain for each workload. Require median paired prompt-time and TTFT '
               'ratios at most one on both. A passing screen permits quality and agent validation.')


def baseline():
    previous = primary.retained.baseline()
    native, envelope, _ = refill.qualified_build()
    require(native['base_binary'] == previous['binary'] and native['base_build_sha256'] == previous['native_build_sha256']
            and all(envelope[k] == previous[k] for k in ('boot', 'driver', 'runtime_manifest', 'native_validation_probe')),
            'the retained source and runtime must match the qualified comparison base')
    row = dict(previous, binary=native['binaries']['strata'], native_source_commit=native['native_source_commit'],
               native_build_sha256=envelope['build_sha256'], qualified_profiles=native['profiles'])
    restore_proof = ranked.restore_proof(row, refill.KEY, refill.ORDER)
    require(restore_proof['profiles'] == native['profiles']
            and all(case['passed'] and not case['fresh_faults'] for case in restore_proof['outcomes'].values()),
            'finish both live RESTORE extents and native cancellation on the ranked-refill build')
    physical = basis.validation.committed(bank.RESULT, (capacity.KEY,))
    require(primary.retained.closed(physical, 'closed')
            and all(physical[k] == row[k] for k in ('native_source_commit', 'native_build_sha256', 'boot', 'driver'))
            and physical['binary'] == native['binaries']['strata_full_context_diagnostic']
            and physical['outcomes']['full-copy']['final_slot_consumed'],
            'finish the full physical context on the ranked-refill build')
    row.update(physical_context_sha256=bank.signature(physical), refill_restore_sha256=bank.signature(restore_proof))
    return row


def configuration(row, plan):
    return ranked.configuration(row, dict(plan, arm='stock'))


def environment(row, plan):
    env = primary.curve.environment(row, plan)
    require(refill.SELECTOR not in env and env.get('STRATA_SYCL_CACHE_LOAN_AUDIT', '0') == '0',
            'the timing environment must start without a refill selector or loan observer')
    if plan['arm'] == 'refill':
        env[refill.SELECTOR] = row['qualified_profiles']['blend']['path']
    return env


def check_publication(text, samples, loaded):
    quota = refill.quota
    startup = f'strata DFlash2 decode refill: available slots={quota.EXPECTED_SLOTS} first_changed={quota.PROTECTED}'
    require([line for line in text.splitlines() if line.startswith('strata DFlash2 decode refill: available ')]
            == ([startup] if loaded else []), 'the live alternate profile differs from the arm')
    publish = (f'{refill.PUBLISH}first={quota.PROTECTED} slots={quota.EXPECTED_SLOTS - quota.PROTECTED} '
               f'payload_bytes={quota.EXPECTED_TAIL}')
    work = {}
    start = 0
    for header, sample in zip(re.finditer(primary.footer.HEADER, text, re.M), samples, strict=True):
        frame = text[start:header.start()]
        events = [line for line in frame.splitlines() if line.startswith(refill.PUBLISH)]
        should_refill = loaded and 'workload' in sample
        require(events == ([publish] if should_refill else []), 'the actual final refill or its extent differs')
        if 'workload' in sample:
            work[sample['workload']] = quota.EXPECTED_TAIL if should_refill else 0
        start = header.end()
    require(refill.PUBLISH not in text[start:], 'unexpected refill after the last request')
    return work


def decision(processes):
    result = ranked.decision(processes, ORDER)
    ttft = all(statistics.median(item['ttft_ratio'] for item in pairs) <= 1 for pairs in result['pairs'].values())
    return dict(result, ttft_preserved=ttft, eligible_for_qualification=result['eligible_for_qualification'] and ttft)


def experiment():
    ctx = primary.experiment()
    base_pins = ctx.pins
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.protocol, ctx.configuration, ctx.environment = baseline, PLAN, configuration, environment
    ctx.evidence_reader.decide = decision

    def pins(revision):
        values = base_pins(revision)
        for name in (THIS, ranked.THIS, capacity.THIS, refill.THIS, refill.build.THIS,
                     refill.fixtures.THIS, refill.stock.THIS, refill.quota.THIS, 'tools/make_profile.py'):
            path = bank.P / name
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit the exact final-refill timing input: ' + name)
            values[name] = bank.sha(path)
        return values

    def routes(text, samples, arm):
        row = bank.previous.read(ROOT / 'record.json')
        profile = row['qualified_profiles']['stock']['path']
        report = primary.routes(text, samples, 'dflash2', profile_check=lambda log:
                                refill.stock.check_loader(log, profile, per_layer=False))
        for name, size in check_publication(text, samples, arm == 'refill').items():
            report[name]['expert_work']['ranked_refill_payload_bytes'] = size
        return report

    ctx.routes, ctx.pins = routes, pins
    return ctx


if __name__ == '__main__':
    primary.clock.main(experiment())
