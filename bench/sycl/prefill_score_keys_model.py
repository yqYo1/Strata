"""Compare index-key repacking and retention on the same long-prefill engine."""

import fcntl
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import prefill_score_keys_v2 as score
import prefill_shared_gate_compare as shared_gate

clock = score.model
bank, basis, require = clock.bank, clock.basis, clock.require
upload, previous = clock.upload, shared_gate.previous
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEYS = {'admit': 'prefill_score_keys_model_admission_v1', 'time': 'prefill_score_keys_model_speed_v1'}
COMPONENT = '9d19c029f61ac0e66c9b040a8ca5c54246d56ecd'
REFERENCES = {
    'admit': ('becc132b8', 'prefill_first_gate_issue_admission_v1'),
    'time': ('5e192224a94b431aa2933d5b8bcd70b1a71e2c8f', 'prefill_first_gate_issue_speed_v1'),
}
SELECTOR = 'STRATA_SYCL_PREFILL_RETAIN_SCORE_KEYS'
MODES = ('repack', 'retain')
ORDER = (*MODES, *reversed(MODES), *MODES)
SCORE_LAYERS = tuple(range(3, 48, 4))
SCORE_BATCH, INDEX_BLOCK = 256, 4
MARKER = 'strata prefill score packs: '


def measured_basis(phase):
    parent, prior, providers = clock.retained_v2_model()
    native = bank.previous.read(score.BUILD_ROOT / 'record.json')
    score.build_context(native['source_commit']).check(native)
    proof = prior['observer_admission']
    proof['build_sha256'] = bank.signature(bank.builder.compact(native))
    runtime_sha256 = prior['runtime_manifest']['sha256']
    component = basis.validation.committed(bank.RESULT, (score.COMPONENT_KEY,))
    inputs = component['inputs']
    require(native['stage'] == 'built' and native['passed'] and native['complete']
            and native['ownership_closed'] and not native['active'] and native['host_control']['passed']
            and native['projection']['score_parent_build_sha256'] == bank.signature(bank.builder.compact(parent))
            and native['driver'] == parent['driver'] == prior['driver']
            and component == bank.previous.historic(COMPONENT, score.COMPONENT_KEY)
            and component['stage'] == 'closed' and component['decision']['component_gain']
            and [x['role'] for x in component['stages']] == list(score.ROLES)
            and all(x['passed'] and x['ownership_closed'] and not x['fresh_faults'] for x in component['stages'])
            and component['stages'][2]['stale_extent_control_detected']
            and inputs['native_source_commit'] == native['native_source_commit']
            and inputs['build_sha256'] == proof['build_sha256']
            and inputs['binary'] == native['binaries'][score.keys.TARGET]
            and inputs['boot'] == prior['boot'] and inputs['runtime_manifest']['sha256'] == runtime_sha256,
            'use the component-qualified score engine, fixture and runtime')
    revision, key = REFERENCES[phase]
    reference = bank.previous.historic(revision, key)
    control = reference['processes'][0]
    read = shared_gate.SMALL_READ if phase == 'admit' else clock.READ
    require(upload.gather.closed(reference) and control['issue_mode'] == 'original' and control['read'] == read
            and control['passed'] and control['ownership_closed'] and not control['fresh_faults']
            and len(control['response_sha256']) == clock.RUNS
            and reference['configuration'] == proof['reference_configuration']
            and reference['driver'] == prior['driver'] and reference['boot'] == prior['boot']
            and reference['model_assets_sha256'] == bank.signature(prior['models'])
            and reference['native_validation_probe'] == prior['probe']
            and reference['runtime_manifest']['sha256'] == runtime_sha256, 'recover the exact V2 model workload')
    work = reference['observer_admission']['expected_work']
    require(phase == 'admit' or work == proof['expected_work'], 'long model work differs from the original V2 route')
    proof.update(expected_work=work, expected_response=control['response_sha256'],
                 reference_evidence=revision, component_evidence=COMPONENT, component_sha256=bank.signature(component))
    if phase == 'time':
        admitted = basis.validation.committed(bank.RESULT, (KEYS['admit'],))
        require(upload.gather.closed(admitted) and admitted['decision']['admitted_modes'] == list(MODES)
                and admitted['binary'] == native['binaries']['strata']
                and admitted['native_source_commit'] == native['native_source_commit']
                and admitted['observer_admission']['build_sha256'] == proof['build_sha256']
                and admitted['observer_admission']['component_sha256'] == proof['component_sha256'],
                'qualify the ragged model path with both selectors before timing')
        proof['short_admission_sha256'] = bank.signature(admitted)
    return native, prior, providers, None


def score_packs(text, read, retain):
    # The final input token uses the verifier. The pooled keys are immutable
    # during SELECT, and scratch ownership ends at the next chunk.
    slices = packed = reused = 0
    for chunk in range(0, read - 1, clock.CHUNK):
        end = min(read - 1, chunk + clock.CHUNK)
        previous_rows = 0
        for t in range(chunk, end, SCORE_BATCH):
            active = min(t + SCORE_BATCH, end) // INDEX_BLOCK + 1
            slices += 1
            packed += active - previous_rows
            reused += previous_rows
            previous_rows = active if retain else 0
    expected = [dict(layer=str(layer), slices=str(slices), packed_live_rows=str(packed), reused_live_rows=str(reused))
                for layer in SCORE_LAYERS]
    require(previous.workspace.records(text, MARKER) == expected, 'actual QSA score pack counts differ from the request')
    return dict(layers=len(expected), slices_per_layer=slices, packed_rows_per_layer=packed, reused_rows_per_layer=reused)


def decision(processes, order, small):
    require(tuple(p['score_mode'] for p in processes) == order
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(order)
            and all(p['passed'] and p['complete'] and p['ownership_closed'] and not p['fresh_faults']
                    and p['native_exit_code'] == 0 and p['clean_speed'] == (not small) for p in processes),
            'close every score-key process in the committed order')
    if small:
        return dict(admitted_modes=list(order), matched_work_and_responses=True)
    rates = {mode: [] for mode in MODES}
    ratios = []
    for i in range(0, len(processes), 2):
        times = {p['score_mode']: previous.timing(p) for p in processes[i:i + 2]}
        for mode in MODES:
            rates[mode].append(1000 * clock.READ / times[mode]['warm_prompt_ms'])
        ratios.append({k: times['retain'][k] / v for k, v in times['repack'].items()})
    warm = [r['warm_prompt_ms'] for r in ratios]
    return dict(warm_process_rates=rates, paired_ratios=ratios,
                repeated_warm_gain=sum(r < 1 for r in warm) >= 2 and statistics.median(warm) < 1,
                repeated_1000={mode: min(values) >= 1000 for mode, values in rates.items()}, adopted=False)


def experiment(phase):
    small = phase == 'admit'
    order = MODES if small else ORDER
    read, runs = (shared_gate.SMALL_READ if small else clock.READ), clock.RUNS
    base = upload.experiment('store-time')
    ctx = basis.core.source.device.load_module('prefill_score_keys_' + phase, basis.core.__file__)
    ctx.THIS, ctx.KEY = THIS, KEYS[phase]
    ctx.ROOT = bank.B / ctx.KEY.replace('_', '-')
    ctx.ORDER, ctx.READS, ctx.RUNS, ctx.CLEAN_FAST = [(read, 1)] * len(order), (read,), runs, True
    ctx.basis = lambda: measured_basis(phase)
    ctx.check_sample = base.check_sample
    ctx.decision = lambda processes: decision(processes, order, small)
    ctx.pins = lambda revision: base.pins(revision) | bank.stage.source.pin(
        bank.P, [THIS, shared_gate.THIS, *score.CONTROLS])['source_sha256']
    original_save, original_current = ctx.save, ctx.current
    plan = dict(order=list(order), read=read, requests=runs, trace_progress=small,
        scope='Toggle index-key retention within each QSA SELECT loop. Fix GDN, attention, selector mapping, '
              'expert transfer, KV and decode.',
        acceptance='Require exact work, reference response digests and all twelve score pack records per request. '
                   'For timing, require two faster subsequent prompt times and a paired median below one. '
                   'Record first use, client TTFT and repeated 1000 tok/s separately.',
        stop='Stop on changed identity, unmatched work, unexpected dispatch, response, validation, ownership, '
             'device fault or finite budget failure.')

    def environment(row, arm):
        env = base.environment(row, dict(arm, index=1))
        require(SELECTOR not in env, 'the V2 reference unexpectedly fixes score-key retention')
        env[SELECTOR] = str(int(arm['score_mode'] == 'retain'))
        if small:
            env.update(STRATA_TRACE='1', STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS='384')
        return env

    def route(text, arm):
        value = shared_gate.request_route(text, dict(arm, gate_mode='original', qsa_prefix_page_size=4))
        value['score_packs'] = score_packs(text, arm['read'], arm['score_mode'] == 'retain')
        return value

    def after_execution(outcome, row, arm):
        work = [previous.split.work(r, s) for r, s in zip(outcome['routes'], outcome['samples'], strict=True)]
        require(work == row['observer_admission']['expected_work'], 'actual ragged or split work differs')
        result, = (Path(arm['root']) / 'bench').glob('*/results.json')
        responses = bank.previous.read(result)['results']['prefill_only']['requests']
        require([x['request_sha256'] for x in responses] == [s['request_sha256'] for s in outcome['samples']],
                'response digests refer to different requests')
        digests = [[x['reasoning_sha256'], x['output_sha256']] for x in responses]
        require(digests == row['observer_admission']['expected_response'], 'score-key model response differs')
        outcome.update(score_mode=arm['score_mode'], response_sha256=digests)

    def save(row):
        if row['stage'] == 'prepared':
            require(row['configuration'] == row['observer_admission']['reference_configuration'],
                    'prepared server differs from the measured V2 route')
            row.update(plan=plan, budgets=clock.BUDGET, whole_bytes=len(order) * clock.BUDGET['total_cap'] + (8 << 20))
            row['request_hashes'][str(read)] = [w['request'] for w in row['observer_admission']['expected_work']]
            for arm, mode in zip(row['plans'], order, strict=True):
                arm.update(score_mode=mode, trace_progress=small)
                if small:
                    arm['command'].append('--prefill-short-diagnostic')
                arm['environment_sha256'] = clock.model.fingerprint(environment(row, arm))
        original_save(row)

    def current(row):
        require(row['plan'] == plan and row['configuration'] == row['observer_admission']['reference_configuration']
                and row['budgets'] == clock.BUDGET
                and row['whole_bytes'] == len(order) * clock.BUDGET['total_cap'] + (8 << 20)
                and [(p['index'], p['read'], p['score_mode'], p['trace_progress']) for p in row['plans']]
                    == [(i, read, mode, small) for i, mode in enumerate(order)], 'frozen score-key plan differs')
        return original_current(row)

    ctx.environment, ctx.request_route, ctx.after_execution = environment, route, after_execution
    ctx.save, ctx.current = save, current
    return ctx


if __name__ == '__main__':
    phase, action = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and phase in KEYS,
            'select an admitted score-key phase with the pinned interpreter')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment(phase)
        if action == 'prepare':
            ctx.prepare()
        elif action == 'host':
            clock.model.host(ctx)
        else:
            require(action in map(str, range(len(ctx.ORDER))), 'select a planned process')
            clock.model.run(int(action), ctx)
