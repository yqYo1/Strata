"""Isolate the first shared Gate weight expansion across the MoE status copy."""

import fcntl
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import prefill_moe_gate_clock_build as build
import prefill_moe_gate_clock_measure as clock
import prefill_shared_gate_compare as shared_gate

bank, basis, require = clock.bank, clock.basis, clock.require
upload, previous = clock.upload, shared_gate.previous
BUILD_KEY = 'prefill_first_gate_issue_native_v1'
BUILD_ROOT = bank.B / BUILD_KEY.replace('_', '-')
KEYS = {'admit': 'prefill_first_gate_issue_admission_v1', 'time': 'prefill_first_gate_issue_speed_v1'}
THIS = str(Path(__file__).resolve().relative_to(bank.P))
CLOCK_EVIDENCE = 'fb0fc95d4'
SHORT_EVIDENCE = 'f3707ee5e'
SELECTOR = 'STRATA_SYCL_PREFILL_GATE_FIRST_PREISSUE'
MARKER = 'strata MoE prefill Gate issue: '
MODES = ('original', 'preissue')
ORDER = (*MODES, *reversed(MODES), *MODES)
WINDOWS = 48
POLICY_FILES = (THIS, build.THIS, clock.THIS, shared_gate.THIS)


def schedule_projection(source):
    replace = build.replace
    unit, window = source[build.UNIT], source[build.WINDOW]
    layer = upload.build.upload.LAYER
    unit = replace(unit, '''    uint32_t status_observed = 0;''',
                   '''    uint32_t status_observed = 0;
    uint64_t first_gate_preissues = 0;''')
    signature = '''bool native_proj16(Gemm& gm, const core::WeightRef* w, const uint16_t* x, uint16_t* y,
                   int64_t rows, const std::string& name, std::string& err) {'''
    unit = replace(unit, signature,
        signature.replace('std::string& err)', 'std::string& err, int64_t preissued_columns = 0)'))
    unit = replace(unit, '''    for (int64_t r = 0; r < n; r += block) {
        const int64_t count = std::min(block, n - r);
        strata::kernels::iq_dequant_f16(w->native_type,
            static_cast<const uint8_t*>(w->native_data) + size_t(r) * row_bytes,
            count * k, gm.scratch(), gm.stream());''', '''    if (preissued_columns && preissued_columns != std::min(block, n))
        throw std::logic_error("prefill Gate: preissued block geometry changed");
    for (int64_t r = 0; r < n; r += block) {
        const int64_t count = std::min(block, n - r);
        if (r || !preissued_columns)
            strata::kernels::iq_dequant_f16(w->native_type,
                static_cast<const uint8_t*>(w->native_data) + size_t(r) * row_bytes,
                count * k, gm.scratch(), gm.stream());''')
    window = replace(window, '''    // The host status destination belongs to the outer checked lease, even if DMA fails.
    owner.status_observed = ~uint32_t(0);''', '''    static const bool preissue_gate = [] {
        const char* value = std::getenv("STRATA_SYCL_PREFILL_GATE_FIRST_PREISSUE");
        if (!value || std::strcmp(value, "0") == 0) return false;
        if (std::strcmp(value, "1") == 0) return true;
        throw std::invalid_argument("prefill Gate preissue requires 0 or 1");
    }();
    int64_t preissued_columns = 0;
    if (preissue_gate && queue.is_in_order() && m.gemm.stream() == m.cs &&
        !use_prefix_upload && !owner.command_trace && wsg->native_data && wsg->ne0 > 0 &&
        wsg->ne1 > 0 && m.gemm.scratch_elems() >= wsg->ne0 &&
        strata::kernels::iq_row_bytes(wsg->native_type, wsg->ne0)) {
        const int64_t columns = std::min(wsg->ne1, m.gemm.scratch_elems() / wsg->ne0);
        const int64_t elements = columns * wsg->ne0;
        if (uint64_t(elements) > SIZE_MAX / sizeof(uint16_t))
            throw std::invalid_argument("prefill Gate preissue scratch extent overflow");
        const size_t scratch_bytes = size_t(elements) * sizeof(uint16_t);
        sycl_backend::validate_device_span(m.gemm.scratch(), scratch_bytes, queue);
        sycl_backend::validate_spans({{m.gemm.scratch(), scratch_bytes}},
            {{status, sizeof(uint32_t)}, {&owner.status_observed, sizeof(uint32_t)}}, alignof(uint16_t));
        // Routed readers are ordered before this scratch write. The copy and
        // checked status gate retire it before the first projection consumes it.
        strata::kernels::iq_dequant_f16(wsg->native_type, wsg->native_data, elements, m.gemm.scratch(), m.cs);
        preissued_columns = columns;
    }
    // The host status destination belongs to the outer checked lease, even if DMA fails.
    owner.status_observed = ~uint32_t(0);''')
    window = replace(window, '''        auto* input = mixed_piece(at, count);
        if (!native_proj16(m.gemm, wsg, input, sgate, count, v.name("ffn_gate_shexp.weight"), err) ||''',
        '''        auto* input = mixed_piece(at, count);
        if (!native_proj16(m.gemm, wsg, input, sgate, count, v.name("ffn_gate_shexp.weight"), err,
                           at == 0 ? preissued_columns : 0) ||''')
    window = replace(window, '''    if (batch_upload) ++owner.batched_upload_windows;
    ++owner.windows;''', '''    if (batch_upload) ++owner.batched_upload_windows;
    if (preissued_columns) ++owner.first_gate_preissues;
    ++owner.windows;''')
    report = replace(source[layer],
        '''        std::fprintf(stderr, "strata MoE accumulator window: windows=%llu''',
        '''        std::fprintf(stderr, "strata MoE prefill Gate issue: windows=%llu preissued=%llu final_drain=checked\\n",
            (unsigned long long)window_context->windows, (unsigned long long)window_context->first_gate_preissues);
        std::fprintf(stderr, "strata MoE accumulator window: windows=%llu''')
    return {build.UNIT: unit, build.WINDOW: window, layer: report}


def build_context(revision):
    ctx = upload.build.context()
    ctx.ROOT, ctx.KEY = BUILD_ROOT, BUILD_KEY
    ctx.scope = 'Move only the first shared Gate dequant before the status read on the same checked queue.'
    ctx.retention = ('Keep the measured V2 prefill and coherent decode generations through the bounded '
                     'admission and speed comparison. Review this third generation at closure.')

    def project(expected=None):
        native, foundation, source, recipe, proof, parent = build.retained_v2_projection(expected)
        generator = dict(bank.stage.source.blobs(bank.P, revision, [THIS]))[THIS]
        require(generator == (bank.P / THIS).read_bytes(), 'use the committed Gate issue source generator')
        overlay = schedule_projection(source)
        source.update(overlay)
        proof = dict(proof, issue_source_commit=revision,
                     issue_parent_build_sha256=bank.signature(bank.builder.compact(parent)),
                     overlays=proof['overlays'] | bank.stage.hashed(overlay))
        return native, foundation, source, build.recipe_at(recipe, BUILD_ROOT), proof

    ctx.project = project
    return ctx


def measured_basis(small):
    parent, prior, providers = clock.retained_v2_model()
    native = bank.previous.read(BUILD_ROOT / 'record.json')
    build_context(native['source_commit']).check(native)
    require(native['stage'] == 'built' and native['passed'] and native['complete']
            and native['ownership_closed'] and not native['active'] and native['host_control']['passed']
            and native['projection']['issue_parent_build_sha256'] == bank.signature(bank.builder.compact(parent))
            and native['driver'] == parent['driver'] == prior['driver'], 'exact Gate issue engine required')
    measured = bank.previous.historic(CLOCK_EVIDENCE, clock.KEY)
    require(upload.gather.closed(measured) and measured['decision']['clock_cost'][1]['overhead_qualified']
            and measured['observer_admission']['timeline_sha256']
                == prior['observer_admission']['timeline_sha256'], 'closed same-route V2 clock motivation required')
    proof = prior['observer_admission']
    proof.update(build_sha256=bank.signature(bank.builder.compact(native)), clock_evidence=CLOCK_EVIDENCE)
    if small:
        original = bank.previous.historic(SHORT_EVIDENCE, shared_gate.KEYS['admit'])
        reference = original['processes'][0]
        require(upload.gather.closed(original) and reference['gate_mode'] == 'original'
                and reference['passed'] and reference['ownership_closed'] and not reference['fresh_faults']
                and original['configuration'] == proof['reference_configuration']
                and original['boot'] == prior['boot'] and original['driver'] == prior['driver']
                and original['native_validation_probe'] == prior['probe']
                and original['model_assets_sha256'] == bank.signature(prior['models']),
                'recover the same ragged GPU route and request bodies')
        proof.update(expected_work=reference['exact_work'], expected_response=reference['response_sha256'],
                     short_evidence=SHORT_EVIDENCE)
    else:
        admitted = basis.validation.committed(bank.RESULT, (KEYS['admit'],))
        require(upload.gather.closed(admitted) and admitted['decision']['admitted_modes'] == list(MODES)
                and admitted['binary'] == native['binaries']['strata']
                and admitted['native_source_commit'] == native['native_source_commit']
                and admitted['observer_admission']['build_sha256'] == proof['build_sha256'],
                'complete the same-build checked ragged admission before timing')
        proof['short_admission_sha256'] = bank.signature(admitted)
    return native, prior, providers, None


def decision(processes, order, small):
    require(tuple(p['issue_mode'] for p in processes) == order
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(order)
            and all(p['passed'] and p['complete'] and p['ownership_closed'] and not p['fresh_faults']
                    and p['native_exit_code'] == 0 and p['clean_speed'] == (not small) for p in processes),
            'complete every process in the frozen Gate issue order')
    if small:
        return dict(admitted_modes=list(order), matched_responses=True)
    rates = {mode: [] for mode in MODES}
    ratios = []
    for index in range(0, len(processes), 2):
        values = {p['issue_mode']: previous.timing(p) for p in processes[index:index + 2]}
        for mode in MODES:
            rates[mode].append(1000 * clock.READ / values[mode]['warm_prompt_ms'])
        ratios.append({k: values['preissue'][k] / v for k, v in values['original'].items()})
    warm = [p['warm_prompt_ms'] for p in ratios]
    return dict(warm_process_rates=rates, paired_ratios=ratios,
                repeated_warm_gain=sum(r < 1 for r in warm) >= 2 and statistics.median(warm) < 1,
                repeated_1000={mode: min(values) >= 1000 for mode, values in rates.items()}, adopted=False)


def experiment(phase):
    small = phase == 'admit'
    order = MODES if small else ORDER
    read, runs = (shared_gate.SMALL_READ if small else clock.READ), clock.RUNS
    ctx = basis.core.source.device.load_module('prefill_first_gate_' + phase, basis.core.__file__)
    base = upload.experiment('store-time')
    ctx.THIS, ctx.KEY = THIS, KEYS[phase]
    ctx.ROOT = bank.B / ctx.KEY.replace('_', '-')
    ctx.ORDER, ctx.READS, ctx.RUNS, ctx.CLEAN_FAST = [(read, 1)] * len(order), (read,) * len(order), runs, True
    ctx.basis = lambda: measured_basis(small)
    ctx.check_sample = base.check_sample
    ctx.decision = lambda processes: decision(processes, order, small)
    ctx.pins = lambda revision: base.pins(revision) | bank.stage.source.pin(bank.P, POLICY_FILES)['source_sha256']
    original_save, original_current = ctx.save, ctx.current
    plan = dict(order=list(order), read=read, requests=runs, trace_progress=small,
        scope='Only first shared Gate dequant placement across status copy. Same in-order queue, existing '
              'temporary weights and checked device status before any shared residual update.',
        acceptance='Match the original 16385-row ragged path, exact work and response digests. Require one '
                   'preissue per finished layer in the candidate and zero in the original.' if small else
                   'Three independent balanced pairs with matched actual work, responses and Gate issue counts. '
                   'At least two faster subsequent prompt times and a paired median ratio below one flag a gain. '
                   'Record first use, client TTFT and repeated 1000 tok/s separately.',
        stop='Stop on identity, route, response, validation, ownership, device fault or budget failure. '
             'Finish every eligible arm in order and retain all samples.')

    def environment(row, arm):
        env = base.environment(row, dict(arm, index=1))
        env[SELECTOR] = str(int(order[arm['index']] == 'preissue'))
        if small:
            env.update(STRATA_TRACE='1', STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS='384')
        return env

    def route(text, arm):
        value = shared_gate.request_route(text, dict(arm, gate_mode='original', qsa_prefix_page_size=4))
        issued = WINDOWS if arm['issue_mode'] == 'preissue' else 0
        require(previous.workspace.records(text, MARKER) ==
                [dict(windows=str(WINDOWS), preissued=str(issued), final_drain='checked')],
                'first Gate issue or retirement count differs')
        value['first_gate_preissues'] = issued
        return value

    def after_execution(outcome, row, arm):
        require([previous.split.work(r, s) for r, s in zip(outcome['routes'], outcome['samples'], strict=True)]
                == row['observer_admission']['expected_work'], 'actual split or ragged routing work differs')
        result, = (Path(arm['root']) / 'bench').glob('*/results.json')
        responses = bank.previous.read(result)['results']['prefill_only']['requests']
        require([x['request_sha256'] for x in responses] == [s['request_sha256'] for s in outcome['samples']],
                'response digests refer to different requests')
        digests = [[x['reasoning_sha256'], x['output_sha256']] for x in responses]
        expected = row['observer_admission']['expected_response'] if small else (
            row['processes'][0]['response_sha256'] if row['processes'] else digests)
        require(digests == expected, 'Gate issue changed the short diagnostic response')
        outcome.update(issue_mode=arm['issue_mode'], response_sha256=digests)

    def save(row):
        if row['stage'] == 'prepared':
            require(row['configuration'] == row['observer_admission']['reference_configuration'],
                    'the prepared server differs from the measured route')
            row.update(plan=plan, budgets=clock.BUDGET,
                       whole_bytes=len(order) * clock.BUDGET['total_cap'] + (8 << 20))
            row['request_hashes'][str(read)] = [w['request'] for w in row['observer_admission']['expected_work']]
            for arm, mode in zip(row['plans'], order, strict=True):
                arm.update(issue_mode=mode, trace_progress=small)
                if small:
                    arm['command'].append('--prefill-short-diagnostic')
                arm['environment_sha256'] = clock.model.fingerprint(environment(row, arm))
        original_save(row)

    def current(row):
        require(row['plan'] == plan and row['configuration'] == row['observer_admission']['reference_configuration']
                and row['budgets'] == clock.BUDGET
                and row['whole_bytes'] == len(order) * clock.BUDGET['total_cap'] + (8 << 20)
                and [(p['index'], p['read'], p['issue_mode'], p['trace_progress']) for p in row['plans']]
                == [(i, read, m, small) for i, m in enumerate(order)], 'frozen Gate issue plan differs')
        return original_current(row)

    ctx.environment, ctx.request_route, ctx.after_execution = environment, route, after_execution
    ctx.save, ctx.current = save, current
    return ctx


if __name__ == '__main__':
    phase, action = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and phase in ('build', *KEYS),
            'select a Gate issue phase with the pinned interpreter')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if phase == 'build':
            require(action in ('projection', 'prepare', 'configure', 'compile', 'host'), 'select a build action')
            require(not (build.ROOT / 'build').exists(), 'retire the completed clock comparison engine')
            row = bank.previous.read(BUILD_ROOT / 'record.json') if action in ('configure', 'compile', 'host') else None
            revision = row['source_commit'] if row else basis.validation.git('rev-parse', 'HEAD').decode().strip()
            build.run_stage(build_context(revision), revision, action, POLICY_FILES)
        else:
            ctx = experiment(phase)
            if action == 'prepare':
                ctx.prepare()
            elif action == 'host':
                clock.model.host(ctx)
            else:
                require(action in map(str, range(len(ctx.ORDER))), 'select a planned process')
                clock.model.run(int(action), ctx)
