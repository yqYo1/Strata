"""Compare request-level decode phase clocks with clean timing on the ordinary B570 route."""
import fcntl
import math
from pathlib import Path
import re
import socket
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'tools/sycl'))
import decode_footer as footer
import prefill_attention_bank_generation as suite
import prefill_resident_input_timing as bank

model, build, require = bank.model, bank.build, bank.require
ORDER = ('clean', 'clocks', 'clocks', 'clean')
ROOT = model.B / 'ordinary-decode-round-cost-v1'
KEY = 'ordinary_decode_round_cost_v1'
THIS = str(Path(__file__).resolve().relative_to(PROJECT))
FIELDS = (*model.FIELDS, 'binary', 'native_source_commit', 'template', 'python_contract', 'python_selected')


def round_cost(segment, tail, sample):
    times = re.findall(footer.TIMING, segment, re.M)
    copies = re.findall(footer.COPY, tail, re.M)
    require(len(times) == len(copies) == 1, 'one clock and subsequent expert footer required')
    windows = int(times[0][0])
    ms = dict(zip(footer.LABELS, map(float, times[0][1:]), strict=True))
    counts = dict(zip(footer.COPY_KEYS, map(int, copies[0]), strict=True))
    offered, accepted = sample['drafts_offered'], sample['drafts_accepted']
    output, duration = sample['generated_tokens'], sample['decode_ms']
    rows = offered + windows
    require(windows > 0 and output > 0 and math.isfinite(duration) and duration > 0
            and offered == 3 * (windows - 1) and 0 <= accepted <= offered
            and 0 <= windows + accepted - output <= 4
            and all(math.isfinite(v) and v >= 0 for v in ms.values()),
            'fixed spec4 work or clock differs')
    jobs, entries = counts['cpu_unique_misses'], counts['cpu_entries']
    require(all(counts[k] == 0 for k in footer.COPY_KEYS[:6]) and ms['PCIe'] == 0
            and 0 < jobs <= entries <= min(4 * jobs, rows * 48 * 10), 'expert dispatch differs')
    for key, value in (('average_T', rows/windows), ('tokens_per_window', output/windows),
                       ('CPU_experts', jobs/(48*windows)), ('CPU_entries', entries/(48*windows)),
                       ('VRAM_hits', (rows*480-entries)/(48*windows))):
        require(abs(ms[key] - value) <= .00501, 'rounded work differs: ' + key)
    require(abs(windows*ms['total'] - duration) <= windows*.005 + .050001, 'engine/API decode clock differs')
    outside = ms['total'] - sum(ms[k] for k in ('verify', 'commit_emit', 'draft'))
    require(outside >= -.021 and ms['verify'] + .021 >= sum(ms[k] for k in
            ('GPU_reach_wait','per_layer_host','stage')) and ms['per_layer_host'] + .026 >=
            sum(ms[k] for k in ('plan','activation_quant','jobs','CPU_pool_run')), 'nested wall clocks exceed parent')
    return dict(windows=windows, verified_rows=rows, emitted=output, unemitted_final_positions=windows+accepted-output,
                cpu_unique_jobs=jobs, cpu_entries=entries, ms_per_window=ms, outer_residual_ms=outside,
                target_ms_per_window_at_70=1000*output/(70*windows))


def routes(text, samples, arm):
    placement = suite.routes(text, samples, 'original')
    observed = arm == 'clocks'
    heads = list(re.finditer(footer.HEADER, text, re.M))
    require(len(heads) == len(samples) and arm in ORDER and 'strata trace:' not in text
            and text.count('strata serve: SYCL_EXPERT_COPY') == len(re.findall(footer.COPY, text, re.M)) == len(heads)
            and text.count('strata decode timing:') == len(re.findall(footer.TIMING, text, re.M))
            == (len(samples) if observed else 0), 'unexpected request count, observer or timing footer')
    summary = {}
    for i, (head, sample) in enumerate(zip(heads, samples, strict=True)):
        require(tuple(map(int, head.groups())) == tuple(sample[k] for k in
                ('prompt_tokens','reused_tokens','read_tokens','generated_tokens','drafts_accepted','drafts_offered')),
                'native/client prompt, reuse, generation or MTP counts differ')
        segment = text[heads[i-1].end() if i else 0:head.start()]
        if observed:
            require(segment.count('strata decode timing:') == 1, 'clock escaped request')
        if sample['phase'].startswith('long-generation/'):
            store = next(p for p in placement if p['phase'] == sample['phase'])
            model.input_receipt(segment, sample['read_tokens']-1, store, False, trace_layers=False)
            if observed:
                tail = text[head.end():heads[i+1].start() if i+1 < len(heads) else len(text)]
                summary[sample['workload']] = round_cost(segment, tail, sample)
    return dict(placement=placement, sustained=summary)


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER and len({
        (p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ORDER)
        and all(p['passed'] and p['complete'] and p['ownership_closed'] and p['native_exit_code'] == 0
                and not p['fresh_faults'] for p in processes), 'four independent completed processes required')
    result = {}
    for name in suite.evidence_reader.SUSTAINED_WORKLOADS:
        ratios, matching = [], []
        acceptance = []
        for p in processes:
            sample, = (s for s in p['evidence']['samples'] if s['phase'] == 'long-generation/' + name)
            acceptance.append(sample['drafts_accepted'] / sample['drafts_offered'])
            require((name in p['routes']['sustained']) == (p['arm'] == 'clocks'), 'wrong observation arm')
        classification = 'strong' if min(acceptance) >= .60 else 'weak' if max(acceptance) <= .40 else 'unconfirmed'
        for clean, clocks in ((processes[0], processes[1]), (processes[3], processes[2])):
            a, b = clean['evidence'], clocks['evidence']
            ratios.append(a['decode_rates'][name] / b['decode_rates'][name])
            matching.append(suite.evidence_reader.work_signature(a['samples'], name) ==
                            suite.evidence_reader.work_signature(b['samples'], name))
        result[name] = dict(clocks_over_clean_time=ratios, matched_work=matching,
                            draft_acceptance=acceptance, observed_MTP_class=classification,
                            attribution_qualified=all(matching) and all(1/1.10 <= r <= 1.10 for r in ratios))
    return result


def baseline():
    native, speed, identity, _ = model.inputs()
    key = 'prefill_resident_input_timing_v1'
    row = build.previous.historic('73d9394a9cb482a2dd091405ce42ca7343d708e3', key)
    retained = model.basis.validation.committed(model.RESULT, (key,))
    require({k: v for k, v in retained.items() if k != 'raw_retirement'} == row
            and row['stage'] == 'closed' and row['passed'] and row['complete'] and row['ownership_closed']
            and not row['active'] and len(row['processes']) == len(bank.ORDER)
            and all(p['passed'] and p['complete'] and p['ownership_closed']
                    and p['native_exit_code'] == 0 and not p['fresh_faults'] for p in row['processes'])
            and row['binary'] == native['binaries']['strata'] and row['native_source_commit'] == native['native_source_commit']
            and build.sha(row['binary']['path']) == row['binary']['sha256'],
            'completed 64K comparison and verified same native image required')
    # The 64K capture directory, including its copy of the provider manifest,
    # has been retired. Check the retained reference against the original bytes.
    runtime = speed['runtime_manifest']
    original = row['runtime_manifest']
    require(runtime['files'] == original['files'] and
            build.sha(runtime['path']) == runtime['sha256'] == original['sha256'],
            'retained runtime providers differ from the measured 64K run')
    source = speed['environment_template']
    case_root = str(Path(row['plans'][0]['root']).parent)
    env = {k: v.replace(source['STRATA_MEASUREMENT_ROOT'], case_root) for k, v in source.items()}
    require(build.signature(env) == row['environment_template_sha256']
            and build.signature(speed['model_assets']) == row['model_assets_sha256'],
            'original 64K environment or model files do not match')
    row.update(runtime_manifest=runtime, environment_template=env, model_assets=speed['model_assets'])
    return {**{k: row[k] for k in FIELDS}, 'native_build_sha256': identity,
            'ordinary_control': row['observer_admission']['ordinary_control']}


def environment(row, plan):
    env = suite.environment(dict(row, ordinary_controls={'original': row['ordinary_control']}),
                            dict(root=plan['root'], arm='original'))
    env[model.INPUT] = '0'
    require(all(k not in env for k in ('STRATA_TRACE', 'STRATA_DECODE_TIMING', 'STRATA_VERIFY_PROFILE',
                                       'STRATA_DRAFT_TIMING', 'STRATA_SNAPSHOT_VERIFY', 'LD_PRELOAD')),
            'unexpected native observer')
    if plan['arm'] == 'clocks':
        env['STRATA_DECODE_TIMING'] = '1'
    return env


CLOCK_COMPARISON = (
    'Current ordinary copy/input and copy/output in all four processes. '
    'Compare clean/clocks then clocks/clean on both official v5 thinking workloads. '
    'Require exact paired input, output and MTP work, complete provider/owner/fault checks, '
    'and a paired latency ratio within a factor of 1.10 in both directions before attributing. '
    'Keep both pairs and the clean rates even if the clock overhead screen fails.')


def inputs(ctx, row, plan):
    native = ctx.native_context(row, plan) if hasattr(ctx, 'native_context') else row
    return {'config.json': ctx.configuration(row, plan), 'plan.json': {
        'binary': native['binary'], 'native_validation_probe': native['native_validation_probe']}}


def prepare(ctx):
    require(not ctx.ROOT.exists() and ctx.KEY not in model.basis.validation.committed(model.RESULT),
            'fresh comparison required')
    revision = suite.validation.git('rev-parse', 'HEAD').decode().strip()
    row = dict(ctx.baseline(), stage='prepared', source_commit=revision, sources=ctx.pins(revision),
               host_passed=False, ownership_closed=True, active=False, passed=False, complete=False,
               plans=[], processes=[], budgets=suite.LIMITS, plan=ctx.protocol)
    ctx.ROOT.mkdir(mode=0o700)
    for index, arm in enumerate(ctx.ORDER):
        root = ctx.ROOT / str(index)
        for name in ('tmp', 'cache'):
            (root / name).mkdir(parents=True, mode=0o700)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        plan = dict(index=index, arm=arm, root=str(root), port=port, binary=row['binary'])
        files = inputs(ctx, row, plan)
        plan['binary'] = files['plan.json']['binary']
        for name, data in files.items():
            build.stage.write(root / name, data)
        plan.update(command=ctx.command(plan), generated={n: build.sha(root / n) for n in files},
                    environment_sha256=suite.signature(ctx.environment(row, plan)))
        row['plans'].append(plan)
    ctx.save(row)


def experiment():
    ctx = model.basis.core.source.device.load_module('ordinary_decode_cost_suite', suite.__file__)
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.environment, ctx.routes, ctx.baseline = environment, routes, baseline
    ctx.protocol = dict(order=list(ORDER), compare=CLOCK_COMPARISON)
    ctx.evidence_reader = SimpleNamespace(read_complete_suite=suite.evidence_reader.read_complete_suite, decide=decision)

    def current(row):
        base = ctx.baseline()
        require(ctx.compact(row) == model.basis.validation.committed(model.RESULT, (ctx.KEY,))
                and ctx.pins(row['source_commit']) == row['sources'] and ctx.THIS in row['sources']
                and all(row[k] == v for k, v in base.items())
                and build.sha(row['template']['path']) == row['template']['sha256']
                and row['plan'] == ctx.protocol and row['budgets'] == suite.LIMITS,
                'frozen suite inputs or campaign state changed')
        for i, (p, arm) in enumerate(zip(row['plans'], ctx.ORDER, strict=True)):
            root = ctx.ROOT / str(i)
            data = inputs(ctx, row, p)
            require(p['index'] == i and p['root'] == str(root) and p['arm'] == arm
                    and p['binary'] == data['plan.json']['binary']
                    and p['command'] == ctx.command(p) and suite.signature(ctx.environment(row, p)) == p['environment_sha256']
                    and set(p['generated']) == set(data) and all(
                    build.previous.read(root/n) == value and build.sha(root/n) == p['generated'][n] for n,value in data.items()),
                    'actual command, generated files or environment changed')
        contract, manifest = model.basis.core.frontend.python_manifest()
        require(contract == row['python_contract'] and manifest['selected'] == row['python_selected'],
                'benchmark providers changed')
        return build.previous.read(Path(row['runtime_manifest']['path'])), manifest

    ctx.compact, ctx.current = model.basis.compact, current
    return ctx


def main(ctx):
    action, = sys.argv[1:]
    require(sys.executable == model.basis.reference.model.PYTHON
            and action in ('prepare', 'host', *map(str, range(len(ctx.ORDER)))),
            'use the pinned interpreter and one planned stage')
    with (model.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.builder.guards.no_model_owners([])
        if action == 'prepare':
            prepare(ctx)
        elif action == 'host':
            model.basis.comparison.host(ctx)
        else:
            ctx.run(int(action))


if __name__ == '__main__':
    main(experiment())
