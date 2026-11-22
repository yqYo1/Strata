"""Compare BF16 attention stores on the measured KV/gather routes."""

import fcntl
from pathlib import Path
from statistics import median
import sys

sys.dont_write_bytecode = True
import prefill_kv_gather_speed as speed
import prefill_split_length_curve as curve
import prefill_split_store as fixture
from engine_once import native_environment
from prefill_prefix_expert_measure import timing
from run_q6_columns import validation_log

bank, basis, require = speed.bank, speed.basis, speed.require
model, state, dispatch = speed.model, speed.state, speed.dispatch
THIS = str(Path(__file__).resolve().relative_to(bank.P))
FIXTURE_KEY = 'prefill_gather_direct_store_component_v1'
FIXTURE_ROOT = bank.B / FIXTURE_KEY.replace('_', '-')
SELECTION = {65537: 'both', 117489: 'reuse'}
ADMISSION_ORDER = (True,)
TIMING_ORDER = (False, True, True, False, False, True)


def closed(row):
    return (row['stage'] == 'closed' and row['complete'] and row['passed']
            and row['ownership_closed'] and not row['active'])


def measured_route(read):
    native, prior, providers, extra = speed.admission(read)
    key = f'prefill_kv_gather_speed_{read}_v1'
    source = basis.validation.committed(bank.RESULT, (key,))
    require(closed(source) and source['binary'] == native['binaries']['strata']
            and source['native_source_commit'] == native['native_source_commit']
            and source['observer_admission'] == prior['observer_admission']
            and source['decision'] == speed.decision(read, source['processes'])
            and curve.select_arm(source['decision']) == SELECTION[read],
            'recover the measured same-build KV/gather winner')
    return native, prior, providers, extra, source


def fixture_inputs(revision):
    native, prior, providers, _, _ = measured_route(65537)
    env = model.ordinary_environment(prior, str(FIXTURE_ROOT), prior['observer_admission']['ordinary_control'])
    env.update(TMPDIR=str(FIXTURE_ROOT / 'tmp'), XDG_CACHE_HOME=str(FIXTURE_ROOT / 'cache'))
    probe = prior['probe']
    env = native_environment(dict(native_validation_probe=probe), env)
    env.update(UR_ENABLE_LAYERS='UR_LAYER_TRACING,UR_LAYER_PARAMETER_VALIDATION',
               UR_LOG_TRACING='level:info;flush:info;output:stderr')
    sources = basis.comparison.build.controls(revision, controller=bank.P / THIS) | controller_sources(revision)
    binary = native['binaries'][fixture.FIXTURE]
    plan = dict(source_commit=revision, sources_sha256=bank.signature(sources), sources_count=len(sources),
                native_source_commit=native['native_source_commit'],
                native_build_sha256=bank.signature(bank.builder.compact(native)),
                engine=native['binaries']['strata'], fixture=binary, command=[binary['path'], 'correctness'],
                budgets=fixture.attention.BUDGET, environment_sha256=bank.signature(env),
                providers_sha256=bank.signature(providers), boot=prior['boot'], driver=prior['driver'],
                scope='Exact BF16 store bits, tails, guards and checked cancellation with Level Zero tracing.')
    return plan, env, probe


def fixture_context():
    ctx = basis.core.source.device.load_module('gather_store_fixture', fixture.__file__)
    ctx.THIS, ctx.KEY, ctx.ROOT, ctx.inputs = THIS, FIXTURE_KEY, FIXTURE_ROOT, fixture_inputs
    ctx.validate_runtime = validate_traced_fixture
    return ctx


def validate_traced_fixture(log, pid, probe):
    receipt = validation_log(log, pid, probe['loader']['path'], traced=True)
    require('urEnqueueKernelLaunch' in log and 'retirement failed' not in log,
            'missing checked attention store launch or failed retirement')
    return receipt


def controller_sources(revision):
    paths = {THIS, fixture.THIS, fixture.upload.THIS, fixture.upload.build.THIS,
             speed.THIS, speed.capacity.THIS, curve.THIS,
             dispatch.THIS, dispatch.split.THIS, dispatch.split.bank.THIS, *state.SCRIPTS}
    sources = {}
    for path in paths:
        original = bank.stage.source.git(bank.P, 'show', revision + ':' + path)
        require((bank.P / path).read_bytes() == original, 'changed store comparison source: ' + path)
        sources[path] = bank.sha(bank.P / path)
    return sources


def key_for(read, traced):
    phase = 'admission' if traced else 'speed'
    return f'prefill_gather_direct_store_{read}_{phase}_v1'


def basis_for(read, traced, runs):
    native, prior, providers, extra, source = measured_route(read)
    component = basis.validation.committed(bank.RESULT, (FIXTURE_KEY,))
    plan = component['plan']
    require(closed(component) and plan['engine'] == native['binaries']['strata']
            and plan['fixture'] == native['binaries'][fixture.FIXTURE]
            and plan['native_source_commit'] == native['native_source_commit']
            and plan['native_build_sha256'] == bank.signature(bank.builder.compact(native))
            and plan['boot'] == prior['boot'] and plan['driver'] == prior['driver']
            and plan['providers_sha256'] == bank.signature(providers)
            and component['result']['cases'] == 4 and not component['result']['fresh_faults'],
            'exact same-build BF16 store and cancellation proof required')

    index = speed.ORDER.index(SELECTION[read])
    control = source['processes'][index]
    work = [dispatch.signature(r, s) for r, s in
            zip(control['routes'][:runs], control['samples'][:runs], strict=True)]
    require(len(work) == runs
            and [w['request'] for w in work] == source['request_hashes'][str(read)][:runs]
            and all(w['store']['pieces'] > 0 and w['store']['direct'] == 0 and w['store']['copy_bytes'] > 0
                    for w in work), 'require the exact warmed comparison bodies and copy geometry')

    copy_trace = basis.validation.committed(bank.RESULT, (dispatch.KEY,))
    copy_index = dispatch.ORDER.index((read, SELECTION[read]))
    previous = copy_trace['processes'][copy_index]
    require([dispatch.signature(r, s) for r, s in zip(previous['routes'], previous['samples'], strict=True)]
            == work[:2], 'recover the two same-build traced copy requests for the selected route')
    proofs = dict(component_sha256=bank.signature(component), timing_sha256=bank.signature(source['decision']),
                  copy_trace_sha256=bank.signature(previous))
    if not traced:
        admitted = basis.validation.committed(bank.RESULT, (key_for(read, True),))
        observer = admitted['observer_admission']
        require(closed(admitted) and admitted['binary'] == native['binaries']['strata']
                and admitted['native_source_commit'] == native['native_source_commit']
                and observer['store_proofs'] == proofs and observer['expected_work'] == work[:2]
                and admitted['decision'] == dict(direct_dispatch_verified=True, requests=2, clean_speed=False),
                'the traced direct store must match the current workload and fixture')
        proofs['admission_sha256'] = bank.signature(admitted)
    prior['observer_admission'].update(expected_work=work, store_proofs=proofs)
    return native, prior, providers, extra


def decision(processes, order, read, traced):
    require(len(processes) == len(order)
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                     for p in processes}) == len(order)
            and tuple(p['direct_output'] for p in processes) == order
            and all(p['passed'] and p['complete'] and p['ownership_closed'] and p['native_exit_code'] == 0
                    and p['clean_speed'] == (not traced) and not p['fresh_faults'] for p in processes),
            'complete all planned independent attention-store processes')
    if traced:
        return dict(direct_dispatch_verified=True, requests=2, clean_speed=False)
    paired, rates = [], dict(copy=[], direct=[])
    for a, b in zip(processes[::2], processes[1::2], strict=True):
        copy, direct = sorted((a, b), key=lambda p: p['direct_output'])
        c, d = timing(copy), timing(direct)
        rates['copy'].append(read * 1000 / c['warm_prompt_ms'])
        rates['direct'].append(read * 1000 / d['warm_prompt_ms'])
        paired.append({name: value / c[name] for name, value in d.items()})
    warm = [p['warm_prompt_ms'] for p in paired]
    return dict(warm_process_PP=rates, paired_direct_over_copy=paired,
                repeated_warm_gain=sum(t < 1 for t in warm) >= 2 and median(warm) < 1,
                repeated_1000={arm: all(t >= 1000 for t in rates[arm]) for arm in rates},
                model_qualified=False, adopted=False)


def experiment(read, traced):
    order = ADMISSION_ORDER if traced else TIMING_ORDER
    runs = 2 if traced else (3 if read == 65537 else 4)
    ctx = basis.core.source.device.load_module('gather_store_comparison', basis.core.__file__)
    ctx.THIS, ctx.KEY = THIS, key_for(read, traced)
    ctx.ROOT = bank.B / ctx.KEY.replace('_', '-')
    ctx.READS, ctx.ORDER, ctx.RUNS, ctx.CLEAN_FAST = (read,) * len(order), [(read, 1)] * len(order), runs, True
    reference = basis.reference.context()
    ctx.basis, ctx.check_sample = lambda: basis_for(read, traced, runs), reference.check_sample
    ctx.decision = lambda processes: decision(processes, order, read, traced)
    base_pins, base_save, base_current = ctx.pins, ctx.save, ctx.current
    reuse, vec8 = state.ARMS[SELECTION[read]]
    plan = dict(length=read, kv_reuse=reuse, vector_gather=vec8, trace=traced, requests=runs,
                direct_output_order=list(order), unit='Change only the BF16 attention output store.',
                gate='Match request, KV, gather, expert placement and draft work. Check executed bank pieces '
                     'and all output copies. Speed needs two faster warmed pairs and paired median below one.',
                scope='Official exact-count bench.sh. Keep first-use and warm PP and TTFT separately. '
                      'The traced admission has no speed claim. Short output qualifies neither TG nor adoption.',
                stop='Stop on first mismatch, ownership refusal, validation fault or budget failure.')

    def pins(revision):
        return base_pins(revision) | controller_sources(revision)

    def environment(row, case):
        env = model.ordinary_environment(row, case['root'], row['observer_admission']['ordinary_control'])
        env.update({model.INPUT: '0', model.OUTPUT: str(int(order[case['index']])),
                    state.KV: str(int(reuse)), state.GATHER: str(int(vec8))})
        if traced:
            env['STRATA_TRACE'] = '1'
        return env

    def route(text, case):
        direct = order[case['index']]
        result = reference.request_route(text, case)
        store = model.restore.require_store(text, dict(direct=direct), case)
        require(store['gpu_rows'] == read - 1 and store['host_rows'] == 0 and store['window_mode'] == 10
                and 'strata MoE prefix expert upload layer:' not in text, 'prefill placement changed')
        result['attention_store'] = store
        result['attention_input'] = model.input_receipt(text, read - 1, store, False, trace_layers=traced)
        result['kv'] = (state.kv.check_request if traced else state.kv.check_clean_request)(
            text, 0, read - 1, model.restore.capacity.CHUNK, reuse=reuse)
        result['gather'] = state.gather_counts(text, vec8)
        if read == dispatch.split.READ:
            result['admission'] = dispatch.split.check_split_budget(text)
        return result

    def check_work(outcome, row, case):
        direct = order[case['index']]
        expected = row['observer_admission']['expected_work']
        if direct:
            expected = [dict(w, store=dict(w['store'], direct=w['store']['pieces'], copy_bytes=0)) for w in expected]
        require(outcome['clean_speed'] == (not traced)
                and [dispatch.signature(r, s) for r, s in zip(outcome['routes'], outcome['samples'], strict=True)]
                    == expected, 'store arm did not match admitted request and placement')
        outcome['direct_output'] = direct

    def save(row):
        if row['stage'] == 'prepared':
            shared = None
            for case, direct in zip(row['plans'], order, strict=True):
                case['direct_output'] = direct
                env = environment(row, case)
                require(basis.comparison.fingerprint(env) == case['environment_sha256'], 'frozen store environment differs')
                normalized = {name: value.replace(case['root'], '<case>') for name, value in env.items()
                              if name != model.OUTPUT}
                require(shared is None or shared == normalized, 'expected identical non-store settings')
                shared = normalized
            row.update(plan=plan, whole_bytes=len(order) * row['budgets']['total_cap'] + (8 << 20))
        base_save(row)

    def current(row):
        require(row['plan'] == plan and [(p['index'], p['read'], p['direct_output']) for p in row['plans']]
                == [(i, read, direct) for i, direct in enumerate(order)], 'frozen store comparison differs')
        return base_current(row)

    ctx.pins, ctx.environment, ctx.request_route, ctx.after_execution = pins, environment, route, check_work
    ctx.save, ctx.current = save, current
    return ctx


def main():
    require(len(sys.argv) in (3, 4),
            'use fixture prepare|run or admit|time prepare|host|INDEX LENGTH')
    phase, action = sys.argv[1:3]
    require(sys.executable == basis.reference.model.PYTHON, 'use the pinned benchmark interpreter')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if phase == 'fixture':
            require(len(sys.argv) == 3 and action in ('prepare', 'run'), 'choose fixture prepare or run')
            ctx = fixture_context()
            (ctx.prepare if action == 'prepare' else ctx.run)()
            return
        length, = sys.argv[3:]
        read = int(length)
        require(phase in ('admit', 'time') and read in SELECTION, 'choose admit or time and an admitted length')
        ctx = experiment(read, phase == 'admit')
        if action == 'prepare':
            ctx.prepare()
        elif action == 'host':
            basis.comparison.host(ctx)
        else:
            require(action in map(str, range(len(ctx.ORDER))), 'choose a planned process')
            basis.comparison.run(int(action), ctx)


if __name__ == '__main__':
    main()
