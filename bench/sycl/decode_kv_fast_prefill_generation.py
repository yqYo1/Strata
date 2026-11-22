"""Time the isolated decode KV change on the fast-prefill server."""

from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import decode_kv_fast_prefill_preservation as preservation
import decode_profile_generation as profiles

clock, suite, require = profiles.clock, profiles.suite, profiles.require
bank, basis, handoff = preservation.bank, preservation.basis, preservation.handoff
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'decode_kv_fast_prefill_generation_v1'
ROOT = bank.B / 'decode-kv-fast-prefill-generation-v1'
READ = 65537
ORDER = suite.ORDER
WORKLOADS = suite.evidence_reader.SUSTAINED_WORKLOADS
PROTOCOL = dict(order=list(ORDER), preservation_lengths=list(preservation.SELECTED),
    comparison='Three independent off/on, on/off, off/on pairs on the same new executable. '
               'Stock expert ranking, 1256 decode slots and the KV reuse/vec8 selectors chosen at 65K in every process. '
               'Run the official long, six-turn agent and both v5 thinking workloads. Change only decode KV selection.',
    acceptance='Require both committed boundary and KV prefill preservation results at each length, '
               'matched body and prefill dispatch, '
               'observed strong/weak MTP classes, and greater TG in two of three pairs with a geometric mean ratio '
               'above one for each sustained workload. Additional fixed-body prefill/TTFT ratios are reported '
               'separately, with a median at most one for preservation. Compare agent preservation only with matched '
               'request histories and work. Report all generated and draft counts. Keep full-context qualification separate.',
    stop='Stop on frozen input, dispatch, completion, ownership, budget or runtime failure. '
         'No early speed stop, replacement or automatic adoption.')


def baseline():
    boundary, prior, providers, _ = preservation.admission(READ)
    binding = prior['observer_admission']
    native = binding['native_builds']['off']
    reference = bank.previous.read(basis.reference.ROOT / 'record.json')
    fixed = {k: reference[k] for k in (*clock.model.FIELDS, 'template', 'python_contract', 'python_selected')}
    runtime = fixed['runtime_manifest']
    require(bank.sha(runtime['path']) == runtime['sha256']
            and bank.previous.read(Path(runtime['path'])) == providers
            and bank.sha(native['binary']['path']) == native['binary']['sha256']
            and all(fixed[k] == prior[k] for k in ('boot', 'driver'))
            and fixed['native_validation_probe'] == prior['probe'] and fixed['model_assets'] == prior['models'],
            'the qualification and official suite must use the same runtime and model')
    gates = {}
    model_sha = bank.signature(fixed['model_assets'])
    for read, (selected, revision) in preservation.SELECTED.items():
        plan = preservation.experiment(read)
        result = basis.validation.committed(bank.RESULT, (plan.KEY,))
        boundary_result = preservation.boundary_timing(read, boundary, binding['boundary_control'])
        boundary_sha = bank.signature(boundary_result)
        decision = preservation.summarize(read, plan.RUNS, result['processes'])
        actual = result['observer_admission']
        require(result['stage'] == 'closed' and result['complete'] and result['passed'] and result['ownership_closed']
                and not result['active'] and result['plan'] == preservation.protocol(read, boundary_only=False)
                and result['decision'] == decision and all(decision['preservation'].values())
                and result['binary'] == boundary['binaries']['strata']
                and result['native_source_commit'] == boundary['native_source_commit']
                and actual['boundary_timing_sha256'] == boundary_sha
                and all(actual[k] == binding[k] for k in (
                    'native_builds', 'boundary_control', 'decode_KV_handoff_sha256', 'decode_KV_device_sha256'))
                and (actual['prefill_arm'], actual['prefill_choice_commit']) == (selected, revision)
                and result['runtime_manifest']['sha256'] == runtime['sha256']
                and result['model_assets_sha256'] == model_sha
                and all(result[k] == fixed[k] for k in ('configuration', 'template', 'native_validation_probe',
                                                       'boot', 'driver', 'python_contract', 'python_selected')),
                'complete prefill preservation on both exact lengths and builds')
        for process in result['processes']:
            arm = 'off' if process['kernel'] == 'on' else process['kernel']
            require(all(process[k] == v for k, v in binding['native_builds'][arm].items()),
                    'prefill result ran a different build')
        gates[str(read)] = dict(boundary=boundary_sha, KV=bank.signature(result))
    return dict(fixed, **native, ordinary_control=binding['ordinary_control'], preservation_sha256=gates)


def environment(row, plan):
    env = profiles.environment(row, plan)
    kv, gather = profiles.state.ARMS[preservation.SELECTED[READ][0]]
    return dict(env, **{profiles.state.KV: str(int(kv)), profiles.state.GATHER: str(int(gather)),
                        handoff.SELECTOR: str(int(plan['arm'] == 'candidate')), 'STRATA_SYCL_KV_COPY_GROUPS': '0'})


def routes(log, samples, arm):
    profiles.check_profile_startup(log, 'original')
    result = clock.routes(log, samples, 'clean')
    captures = handoff.previous.kv_captures(log, int(arm == 'candidate'))
    require(any(c['phase'] == 'decode' and c['width'] == 4 for c in captures)
            and profiles.profile.census.PREFIX not in log and 'strata loan audit:' not in log,
            'missing decode T4 capture or unexpected observer')
    result.update(KV_captures=captures, prefill_dispatch={})
    kv, gather = profiles.state.ARMS[preservation.SELECTED[READ][0]]
    headers = list(re.finditer(clock.footer.HEADER, log, re.M))
    for i, (head, sample) in enumerate(zip(headers, samples, strict=True)):
        if sample['phase'].startswith('long-generation/'):
            frame = log[headers[i-1].end() if i else 0:head.start()]
            rows = sample['read_tokens'] - 1
            require(rows >= 32767 and sample['reused_tokens'] == 0 and sample['drafts_offered'] > 0,
                    'sustained workload must use a fresh long prompt and speculative decoding')
            result['prefill_dispatch'][sample['workload']] = dict(
                kv=profiles.state.kv.check_clean_request(frame, 0, rows, 8192, reuse=kv),
                gather=profiles.state.gather_counts(frame, gather))
    require(set(result['prefill_dispatch']) == set(WORKLOADS), 'incomplete actual prefill dispatch')
    return result


def decision(processes):
    result = profiles.decision(processes)
    result['same_suite_prefill_preserved'] = all(
        statistics.median(p[k] for p in pairs) <= 1 for pairs in result['fixed_prompt_pairs'].values()
        for k in ('prompt_time_ratio', 'ttft_ratio'))
    result['agent_preserved_on_matched_work'] = (all(result['matched_agent_request_bodies'])
        and all(result['matching_work_counts_by_metric']['agent_followup'])
        and result['decode']['agent_followup']['preservation_screen_passed'])
    result['sustained_gain_with_matching_counts'] = result['sustained_gain_screen'] and all(
        all(result['matching_work_counts_by_metric'][name]) for name in WORKLOADS)
    return result


def experiment():
    ctx = clock.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.environment, ctx.routes = baseline, environment, routes
    ctx.configuration = lambda row, plan: profiles.configuration(row, dict(plan, arm='original'))
    ctx.protocol, ctx.evidence_reader.decide = PROTOCOL, decision

    def pins(revision):
        values = clock.suite.pins(revision)
        values.update(preservation.experiment(READ).pins(revision))
        for path in (THIS, clock.THIS, profiles.THIS, *profiles.profile.SOURCES, profiles.profile.PROFILES['stock']):
            require((bank.P / path).read_bytes() == suite.gen.git('show', revision + ':' + path),
                    'changed suite input: ' + path)
            values[path] = bank.sha(bank.P / path)
        return values

    ctx.pins = pins
    return ctx


if __name__ == '__main__':
    clock.main(experiment())
