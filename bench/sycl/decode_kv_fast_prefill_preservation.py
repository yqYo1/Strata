"""Compare prefill before and after the isolated decode KV change."""

import fcntl
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
import prefill_kv_gather_speed as speed
import decode_kv_fast_prefill_build as build
import decode_kv_fast_prefill_restore as handoff
import prefill_split_length_curve as curve
from prefill_prefix_expert_measure import timing

bank, basis, model, require = speed.bank, speed.basis, speed.model, speed.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
SELECTED = {
    65537: ('both', '84c8833935644f989692ebe3a8a8075210b86c15'),
    117489: ('reuse', '568d015bae99e9aa0455ffedabe9a3f61a70f1f3'),
}
BLOCKS = (('retained', 'off', 'on'), ('on', 'off', 'retained'), ('retained', 'off', 'on'))
ORDER = tuple(arm for block in BLOCKS for arm in block)
CONTRASTS = (('off', 'retained'), ('on', 'off'))
METRICS = ('first_prompt_ms', 'warm_prompt_ms', 'first_ttft_s', 'warm_ttft_s')


def admission(read):
    parent, prior, providers, _ = speed.admission(read)
    selected, commit = SELECTED[read]
    evidence = bank.previous.historic(commit, f'prefill_kv_gather_speed_{read}_v1')
    require(evidence['stage'] == 'closed' and evidence['complete'] and evidence['passed']
            and evidence['ownership_closed'] and not evidence['active']
            and evidence['binary'] == parent['binaries']['strata']
            and curve.select_arm(evidence['decision']) == selected,
            'the exact repeated prefill selection is required')
    candidate = bank.previous.read(build.ROOT / 'record.json')
    build.check(candidate)
    gate = basis.validation.committed(bank.RESULT, (handoff.KEY,))
    device = basis.validation.committed(bank.RESULT, (build.DEVICE_KEY,))
    runtime = gate['runtime_manifest']
    require(candidate['stage'] == 'built' and candidate['complete'] and candidate['passed']
            and candidate['ownership_closed'] and not candidate['active']
            and gate['stage'] == 'closed' and gate['complete'] and gate['passed']
            and gate['ownership_closed'] and not gate['active'] and gate['plan'] == handoff.PLAN
            and gate['native_source_commit'] == candidate['native_source_commit']
            and gate['binary'] == candidate['binaries']['strata']
            and gate['build_sha256'] == bank.signature(build.builder.compact(candidate))
            and device['stage'] == 'closed' and device['passed']
            and gate['device_sha256'] == bank.signature(device)
            and list(gate['outcomes']) == handoff.ORDER
            and all(out['passed'] and out['complete'] and out['native_exit_code'] == 0
                    and out['ownership_closed'] and not out['fresh_faults'] for out in gate['outcomes'].values())
            and gate['boot'] == prior['boot'] and gate['driver'] == prior['driver']
            and gate['native_validation_probe'] == prior['probe']
            and bank.sha(runtime['path']) == runtime['sha256']
            and bank.previous.read(Path(runtime['path'])) == providers
            and parent['binaries']['strata']['sha256'] != candidate['binaries']['strata']['sha256'],
            'close the matched model/RESTORE route qualification for the new ELF first')
    builds = {name: dict(binary=row['binaries']['strata'],
                        native_source_commit=row['native_source_commit'])
              for name, row in (('retained', parent), ('off', candidate))}
    prior['observer_admission'].update(prefill_arm=selected, prefill_choice_commit=commit, native_builds=builds,
        decode_KV_handoff_sha256=bank.signature(gate), decode_KV_device_sha256=bank.signature(device))
    return parent, prior, providers, None


def summarize(read, runs, processes):
    require(len(processes) == len(ORDER) and len({(p['native_identity']['pid'], p['native_identity']['start_ticks'])
                for p in processes}) == len(ORDER), 'nine independent closed processes required')
    times, pp = {arm: [] for arm in BLOCKS[0]}, {arm: [] for arm in BLOCKS[0]}
    by_contrast = {a + '/' + b: {metric: [] for metric in METRICS} for a, b in CONTRASTS}
    for index, (process, arm) in enumerate(zip(processes, ORDER, strict=True)):
        samples = process['samples']
        require(process['index'] == index and process['read'] == read and process['kernel'] == arm
                and process['clean_speed'] and process['passed'] and process['complete']
                and process['ownership_closed'] and process['native_exit_code'] == 0
                and not process['fresh_faults'] and len(samples) == runs, 'ineligible prefill process')
        row = timing(process)
        times[arm].append(row)
        pp[arm].append(1000 * read / row['warm_prompt_ms'])
        if (index + 1) % 3 == 0:
            for a, b in CONTRASTS:
                for k, v in by_contrast[a + '/' + b].items():
                    v.append(times[a][-1][k] / times[b][-1][k])
    medians = {name: {k: statistics.median(v) for k, v in ratios.items()} for name, ratios in by_contrast.items()}
    return dict(process_times=times, warmed_process_PP=pp, paired_time_ratios=by_contrast,
                paired_medians=medians,
                preservation={name: all(v <= 1 for v in ratio.values()) for name, ratio in medians.items()},
                adopted=False, sustained_decode_qualified=False)


def experiment(read):
    ctx = basis.core.source.device.load_module('decode_kv_prefill_preservation', basis.core.__file__)
    reference = basis.reference.context()
    ctx.THIS = THIS
    ctx.ROOT = bank.B / f'decode-kv-fast-prefill-preservation-{read}-v1'
    ctx.KEY = f'decode_kv_fast_prefill_preservation_{read}_v1'
    ctx.READS, ctx.ORDER = (read,) * len(ORDER), [(read, 1)] * len(ORDER)
    ctx.RUNS, ctx.CLEAN_FAST = (3 if read == 65537 else 4), True
    ctx.basis, ctx.check_sample = lambda: admission(read), reference.check_sample
    ctx.decision = lambda processes: summarize(read, ctx.RUNS, processes)
    base_save, base_current, base_pins = ctx.save, ctx.current, ctx.pins
    protocol = dict(blocks=[list(block) for block in BLOCKS], requests=ctx.RUNS,
        contrasts=[a + '/' + b for a, b in CONTRASTS], prefill_arm=SELECTED[read][0],
        acceptance='All four paired median prompt/TTFT ratios must be <= 1 in each contrast. '
                   'Retain first and warmed clocks and every individual process. No speed-based stopping.',
        scope='Retained/off compares the executable projection. Off/on compares only decode KV selection. '
              'Both contrasts share the off process independently in each block.')

    def pins(revision):
        values = base_pins(revision)
        require({THIS, speed.THIS, curve.THIS, build.THIS, handoff.THIS, speed.capacity.THIS,
                 speed.dispatch.THIS, *speed.state.SCRIPTS} <= values.keys(), 'unfrozen controller or qualifier')
        return values

    def native_context(row, plan):
        arm = ORDER[plan['index']]
        require(plan['kernel'] == arm, 'unplanned native case')
        native = row['observer_admission']['native_builds']['retained' if arm == 'retained' else 'off']
        return dict(native, native_validation_probe=row['native_validation_probe'], boot=row['boot'])

    def environment(row, plan):
        env = model.ordinary_environment(row, plan['root'], row['observer_admission']['ordinary_control'])
        reuse, vec = speed.state.ARMS[SELECTED[read][0]]
        env.update(**{model.INPUT: '0', model.OUTPUT: '0', speed.state.KV: str(int(reuse)),
                      speed.state.GATHER: str(int(vec)), handoff.SELECTOR: str(int(ORDER[plan['index']] == 'on')),
                      'STRATA_SYCL_KV_COPY_GROUPS': '0'})
        require('STRATA_TRACE' not in env and 'STRATA_DECODE_TIMING' not in env, 'unplanned observer')
        return env

    def route(text, plan):
        reuse, vec = speed.state.ARMS[SELECTED[read][0]]
        value = reference.request_route(text, plan)
        store = model.restore.require_store(text, dict(direct=False), plan)
        rows = read - 1
        require(store['gpu_rows'] == rows and store['host_rows'] == 0 and store['window_mode'] == 10
                and 'strata MoE prefix expert upload layer:' not in text, 'prefill placement changed')
        value['attention_store'] = store
        value['attention_input'] = model.input_receipt(text, rows, store, False, trace_layers=False)
        value['kv'] = speed.state.kv.check_clean_request(text, 0, rows, 8192, reuse=reuse)
        value['gather'] = speed.state.gather_counts(text, vec)
        if read == speed.dispatch.split.READ:
            value['admission'] = speed.dispatch.split.check_split_budget(text)
        if plan['kernel'] == 'retained':
            require('strata verify KV_PLAN:' not in text, 'retained scalar executable differs')
            value['decode_KV'] = []
        else:
            value['decode_KV'] = handoff.previous.kv_captures(text, int(plan['kernel'] == 'on'))
        return value

    def after_execution(outcome, row, plan):
        work = [speed.dispatch.signature(r, s) for r, s in
                zip(outcome['routes'], outcome['samples'], strict=True)]
        require(outcome['clean_speed'] and outcome['index'] == plan['index']
                and work == row['observer_admission']['expected_work'], 'fixed input or measured work changed')
        if plan['kernel'] != 'retained':
            captures = [c for route in outcome['routes'] for c in route['decode_KV']]
            require(any(c['phase'] == 'decode' and c['width'] == 4 for c in captures),
                    'the selected decode shape was absent from the timed requests')

    def save(row):
        if row['stage'] == 'prepared':
            for plan, arm in zip(row['plans'], ORDER, strict=True):
                plan['kernel'] = arm
                root = Path(plan['root'])
                native = native_context(row, plan)
                ctx.write(root / 'plan.json', {k: native[k] for k in ('binary', 'native_validation_probe')})
                plan['generated']['plan.json'] = bank.sha(root / 'plan.json')
            row.update(plan=protocol, whole_bytes=len(ORDER) * row['budgets']['total_cap'] + (8 << 20))
        base_save(row)

    def current(row):
        providers = base_current(row)
        require(row['plan'] == protocol and row['configuration']['exe'] == str(bank.P / 'tools/sycl/engine_once.py'),
                'frozen comparison or supervised launcher differs')
        for index, (plan, arm) in enumerate(zip(row['plans'], ORDER, strict=True)):
            root = ctx.ROOT / str(index)
            native = native_context(row, plan)
            fields = ['config.json', 'plan.json']
            port = int(plan['command'][5])
            command = ['/bin/sh', str(bank.P / 'bench.sh'), '--config', str(root / 'config.json'),
                '--port', str(port), '--prefill-exact-tokens', str(read), '--think', '--runs', str(ctx.RUNS),
                '--profile-request-gates', '--workload-id', 'native_gdn_20261018',
                '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root / 'bench')]
            require((plan['index'], plan['read'], plan['heads'], plan['root'], plan['kernel']) ==
                    (index, read, 1, str(root), arm) and set(plan['generated']) == set(fields)
                    and 1 <= port <= 65535 and plan['command'] == command
                    and all(bank.sha(root / n) == plan['generated'][n] for n in fields)
                    and bank.previous.read(root / 'plan.json') == {
                        k: native[k] for k in ('binary', 'native_validation_probe')}
                    and bank.previous.read(root / 'config.json') == dict(row['configuration'], log=str(root/'engine.stderr'))
                    and basis.comparison.fingerprint(environment(row, plan)) == plan['environment_sha256'],
                    'changed immutable native, configuration or environment')
        return providers

    ctx.pins, ctx.native_context, ctx.environment, ctx.request_route = pins, native_context, environment, route
    ctx.save, ctx.current, ctx.after_execution = save, current, after_execution
    return ctx


if __name__ == '__main__':
    length, action = sys.argv[1:]
    read = int(length)
    require(sys.executable == basis.reference.model.PYTHON and read in SELECTED
            and action in ('prepare', 'host', *map(str, range(len(ORDER)))), 'select one planned comparison stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment(read)
        if action == 'prepare':
            ctx.prepare()
        elif action == 'host':
            basis.comparison.host(ctx)
        else:
            basis.comparison.run(int(action), ctx)
