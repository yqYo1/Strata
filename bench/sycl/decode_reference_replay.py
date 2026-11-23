"""Recheck the retained coherent decode reference with the current full suite.

Uses owned Strata measurement controls under the repository LICENSE.
"""

import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True

import decode_profile_generation as profiles
import decode_window_bench as coherent

clock, suite, require = profiles.clock, profiles.suite, profiles.require
old = coherent.previous
reader = suite.evidence_reader
THIS = str(Path(__file__).resolve().relative_to(clock.PROJECT))
ROOT = clock.model.B / 'decode-reference-replay-v1'
KEY = 'decode_reference_replay_v1'
CONTROL = 'decode_mtp_prefix_generation_v1'
ORDER = ('original',) * 3
OFF_CASES = (0, 3, 4)
PROTOCOL = dict(
    order=list(ORDER),
    reference='Run three independent processes of the retained coherent engine. Reuse the '
              'exact MTP-prefix-off configuration, environment and official full-suite request order.',
    comparison='Compare with the three completed MTP-prefix-off processes. The two sets '
               'have different collection periods. Use them to select the next decode development reference.',
    selection='For each sustained workload, match whole preceding request histories and '
              'work counts and confirm the original strong or weak MTP class. Require at least two '
              'replay rates above the control median and a higher replay median for both workloads. '
              'Report PP, client TTFT, TG and full request time separately. No default adoption.',
    stop='Stop on input, route, request, runtime, ownership or GPU fault failure. Preserve every '
         'outcome and keep the experiment within the bounded official suite.',
)


def closed(process):
    return (process['passed'] and process['complete'] and process['ownership_closed']
            and process['native_exit_code'] == 0 and not process['fresh_faults'])


def controls():
    record = suite.validation.committed(clock.model.RESULT, (CONTROL,))
    processes = record['processes']
    require(record['stage'] in ('closed', 'failed')
            and record['ownership_closed'] and not record['active']
            and record['plan']['order'] == list(suite.ORDER) and len(processes) == len(suite.ORDER)
            and all(p['index'] == i and p['arm'] == arm and p['ownership_closed'] and not p['fresh_faults']
                    for i, (p, arm) in enumerate(zip(processes, suite.ORDER, strict=True))),
            'settle the six fixed-prefix processes without an unresolved GPU fault')
    off = [processes[i] for i in OFF_CASES]
    require(all(p['arm'] == 'original' and closed(p) for p in off),
            'three completed prefix control processes required')
    return record, off


def configuration(row, plan):
    config = profiles.configuration(row, plan)
    args = config['args']
    require('--mtp-max-t' not in args and '--no-coupled-draft' in args,
            'expect the stock uncoupled MTP configuration')
    return dict(config, args=[*args, '--mtp-max-t', '4'])


def environment(row, plan):
    return dict(profiles.environment(row, plan), STRATA_SYCL_MTP_DECODE_COMMITTED_PREFIX='0')


def baseline():
    reference, off = controls()
    native, proof, prior, _ = coherent.basis()
    common = {k: prior[k] for k in clock.FIELDS if k not in ('binary', 'native_source_commit')}
    require(all(reference[k] == v for k, v in common.items()
                if k not in ('environment_template', 'model_assets'))
            and all(reference[k + '_sha256'] == coherent.signature(common[k])
                    for k in ('environment_template', 'model_assets'))
            and native['binaries']['strata'] != reference['binary'],
            'model, runtime or common reference settings differ')
    basis = dict(common, ordinary_control=reference['ordinary_control'])
    for i in OFF_CASES:
        plan = reference['plans'][i]
        config = (json.dumps(configuration(basis, plan), indent=2) + '\n').encode()
        require(plan['index'] == i and plan['arm'] == 'original'
                and plan['command'] == suite.command(plan)
                and plan['generated']['config.json'] == hashlib.sha256(config).hexdigest()
                and plan['environment_sha256'] == suite.signature(environment(basis, plan)),
                'benchmark command, configuration or prefix-off environment changed')
    return dict(basis, binary=native['binaries']['strata'],
                native_source_commit=native['native_source_commit'],
                native_build_sha256=proof['original_record_sha256'],
                control_sha256=suite.signature(off))


def routes(log, samples, _arm):
    _, off = controls()
    suite.official.matched(samples, off[0]['evidence']['samples'])
    profiles.check_profile_startup(log, 'original')
    old.clean_observers(log)
    copies = old.copy_records(log, samples, arm='reference')
    heads = list(re.finditer(clock.footer.HEADER, log, re.M))
    require(len(heads) == len(samples), 'native request count differs')
    widths = sorted(set(map(int, re.findall(r'^strata verify: phase=decode tokens=(\d+) ', log, re.M))))
    require(4 in widths and all(1 <= width <= 4 for width in widths),
            'the native verifier must exercise the planned width four')
    prefill_route = old.prefill.context().request_route
    gdn = old.runner.generation.gdn
    measured = {}
    for i, (head, sample) in enumerate(zip(heads, samples, strict=True)):
        if not sample['phase'].startswith('long-generation/'):
            continue
        frame = log[heads[i - 1].end() if i else 0:head.start()]
        read = sample['read_tokens']
        require(32768 <= read <= 65537, 'sustained replay exceeded its admitted input range')
        gdn.native16_route(frame, read)
        gdn.gdn_selection(frame, 'reference')
        check = prefill_route(frame, dict(read=read, heads=1))
        measured[sample['workload']] = dict(
            rows=check['rows'], chunk=check['actual_chunk'],
            gpu_rows=check['residual_gpu_rows'],
            expert_slots=check['fast_window']['prefill_slots'],
            expert_jobs=check['fast_window']['jobs'],
            decode_copy=copies[i],
        )
    require(set(measured) == set(reader.SUSTAINED_WORKLOADS), 'both replay workloads are required')
    return dict(sustained=measured, decode_capture_widths=widths)


def request_work(samples, select):
    return [(s['phase'], s['run'], s['turn'], s['request_sha256'],
             *(s[k] for k in reader.WORK_FIELDS)) for s in samples if select(s)]


def decision(processes):
    require(len(processes) == len(ORDER) and all(closed(p) for p in processes),
            'three completed replay processes required')
    _, off = controls()
    identities = {(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                  for p in (*processes, *off)}
    require(len(identities) == len(processes) + len(off), 'independent processes required')
    control = [p['evidence']['samples'] for p in off]
    replay = [p['evidence']['samples'] for p in processes]
    preceding = lambda s: not s['phase'].startswith('long-generation/')
    prior_work = request_work(control[0], preceding)
    matched_prelude = all(request_work(samples, preceding) == prior_work
                          for samples in (*control, *replay))
    workloads = {}
    for name in reader.SUSTAINED_WORKLOADS:
        target = 'long-generation/' + name
        select = lambda s: s['phase'] == target
        actual = [next(s for s in samples if select(s)) for samples in (*control, *replay)]
        expected = request_work(control[0], select)
        matched_work = all(request_work(samples, select) == expected for samples in (*control, *replay))
        acceptance = [s['drafts_accepted'] / s['drafts_offered'] for s in actual]
        class_confirmed = min(acceptance) >= .60 if name == 'regular-code' else max(acceptance) <= .40
        current, retained = [[p['evidence']['decode_rates'][name] for p in group]
                             for group in (off, processes)]
        workloads[name] = dict(control_TG=current, replay_TG=retained, matched_work=matched_work,
                               class_confirmed=class_confirmed,
                               replay_reference_screen=(matched_work and class_confirmed
                                   and statistics.median(retained) > statistics.median(current)))
    return dict(workloads=workloads, matched_prelude=matched_prelude,
                select_replayed_reference=matched_prelude and all(
                    result['replay_reference_screen'] for result in workloads.values()))


def experiment():
    ctx = clock.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.protocol, ctx.baseline, ctx.routes = PROTOCOL, baseline, routes
    ctx.configuration, ctx.environment = configuration, environment
    ctx.evidence_reader.decide = decision

    def pins(revision):
        values = suite.pins(revision)
        for name in (THIS, clock.THIS, profiles.THIS, coherent.THIS, old.THIS,
                     profiles.profile.PROFILES['stock'], *profiles.profile.SOURCES):
            path = clock.PROJECT / name
            require(path.read_bytes() == suite.gen.git('show', revision + ':' + name),
                    'changed replay controller input: ' + name)
            values[name] = suite.sha(path)
        return values

    ctx.pins = pins
    return ctx


if __name__ == '__main__':
    clock.main(experiment())
