"""Collect separate long DFlash2 routing workloads on the measured stock layout."""

import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_verifier_routing_check as probe

confidence, model, bank, basis = probe.confidence, probe.model, probe.bank, probe.basis
leaves, clock, require = confidence.leaves, confidence.clock, probe.require
census, footer, routing = probe.census, probe.footer, probe.routing
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ROLES = {'train': 'regular-code', 'holdout': 'varied-prose'}
CONFIDENCE = 'ce3ef1a3a341d15b1a88bc83802018c960c93be7'
ROUTING_GATE = 'c13a80bd4cfb2117258c6eb79874a98a2af56d98'
PROSE_BODY = 'e0a799263acdceb9d773a6745313b4d8a66004f4'
WORKLOAD_VERSION = 5
WORKLOAD_ID = leaves.generation.WORKLOAD_ID
MAX_OUTPUT = 16384
ROUTING_FILE = 'expert-routing.dat'
USAGE_FILE = 'verifier-usage.json'
FORECAST = 'dflash2_expert_rank_forecast_v1'


def request_reference(role):
    if role == 'train':
        row = bank.previous.historic(CONFIDENCE, confidence.KEY)
        sample = row['processes'][0]['evidence']['samples'][-1]
        revision = CONFIDENCE
        require(row['stage'] == 'closed' and row['ownership_closed']
                and row['processes'][0]['arm'] == 'threshold', 'missing measured code body')
    else:
        row = bank.previous.historic(PROSE_BODY, 'dflash2_v2_generation_sync_v1')
        sample = row['processes'][0]['evidence']['samples'][-1]
        revision = PROSE_BODY
        # The old route checker failed. Its observed request only pins the new input.
        require(row['stage'] == 'failed' and row['ownership_closed']
                and sample['read_tokens'] == 33287, 'missing original prose body')
    require(sample['phase'] == 'long-generation/' + ROLES[role]
            and sample['read_tokens'] >= 32768 and sample['reused_tokens'] == 0,
            'reference must identify a fresh official long request')
    return dict(commit=revision, request_sha256=sample['request_sha256'], read_tokens=sample['read_tokens'])


def baseline(role):
    base = confidence.baseline()
    gate = bank.previous.historic(ROUTING_GATE, probe.KEY)
    plan = gate['plan']
    require(gate['stage'] == 'closed' and gate['passed'] and gate['complete']
            and gate['ownership_closed'] and not gate['active']
            and gate['outcome']['native_exit_code'] == 0 and not gate['outcome']['fresh_faults']
            and all(plan[k] == base[k] for k in ('binary', 'native_build_sha256', 'native_source_commit',
                                                 'runtime_manifest', 'boot', 'driver')),
            'require the retained engine and closed independent census check')
    build = leaves.speed.build.DFLASH2_KV_ROOT
    require(all(bank.sha(build / 'source' / p) == h for p, h in plan['producer_pins'].items())
            and all(bank.sha(plan['routes'][k]) == plan['routes'][k + '_sha256'] for k in ('profile', 'manifest')),
            'routing producers or expert metadata changed')
    metadata = {k: plan['routes'][k] for k in ('profile', 'profile_sha256', 'manifest', 'manifest_sha256')}
    row = dict(base, routing_gate=ROUTING_GATE, routing_gate_sha256=bank.signature(gate),
               routing_metadata=metadata, routing_workload=request_reference(role))
    if role == 'holdout':
        forecast = basis.validation.committed(bank.RESULT, (FORECAST,))
        geometry = dict(metadata, binary=base['binary']['sha256'], native_source_commit=base['native_source_commit'],
                        boot=base['boot'], driver=base['driver'])
        require(forecast['stage'] == 'frozen' and forecast['geometry'] == geometry
                and bank.sha(forecast['profile']['path']) == forecast['profile']['sha256']
                and forecast['train']['request_sha256'] != row['routing_workload']['request_sha256'],
                'require the committed training-only profile before the held-out capture')
        row['rank_candidate_freeze'] = bank.signature(forecast)
    return row


def configuration(row, plan):
    cfg = confidence.configuration(row, dict(plan, arm='threshold'))
    option = ['--dump-routing', str(Path(plan['root']) / ROUTING_FILE)]
    out = dict(cfg, user_args=[*cfg['user_args'], *option])
    resolved = model.server.engine_args(out, exe=out['exe'], announce=False)
    require('--dump-routing' not in cfg['args'] and resolved.count('--dump-routing') == 1,
            'expected one new routing argument')
    offset = resolved.index('--dump-routing')
    require(resolved[offset:offset + 2] == option
            and resolved[:offset] + resolved[offset + 2:] == cfg['args']
            and resolved[resolved.index('--draft-conf') + 1] == '0.6',
            'routing must be the only native argument added to the measured default')
    return dict(out, args=resolved)


def environment(row, plan):
    env = confidence.environment(row, dict(plan, arm='threshold'))
    require(env['STRATA_COMMIT_SYNC'] == '1' and env['STRATA_WATCHDOG_S'] == '240'
            and env.get('STRATA_SYCL_VERIFY_OVERLAP', '1') != '0',
            'retain the qualified verifier order')
    return dict(env, STRATA_TRACE='1', STRATA_DECODE_TIMING='1', STRATA_CPU_WORK_CENSUS='1')


def command(plan, role):
    args = leaves.command(plan)
    args[args.index('--long-generation-workload') + 1] = ROLES[role]
    return args


def evidence(root, workload_id, role):
    name = ROLES[role]
    path, = (root / 'bench').glob('*/results.json')
    saved = bank.previous.read(path)
    params, identity, results = (saved[k] for k in ('params', 'comparison_identity', 'results'))
    client = leaves.suite.official.d.client
    journal = client.verify_request_journal(path.parent)
    workload = dict(workload_id=workload_id, nonce=workload_id, nonce_policy='fixed-id')
    generation = results['long_generation']
    require(saved['complete'] and not results['failures'] and not results['skipped']
            and params['runs'] == 1 and params['think'] and params['long_generation_only']
            and generation['complete'] and generation['diagnostic_subset']
            and generation['selected_workloads'] == [name]
            and generation['thinking'] and not generation['speed_only']
            and generation['reasoning_budget_tokens'] == 0 and generation['max_tokens'] == MAX_OUTPUT
            and generation['workload_version'] == identity['long_generation_version'] == WORKLOAD_VERSION
            and params['workload'] == identity['workload'] == workload
            and saved['machine']['context'] == identity['context'] == 262144
            and identity['client'] == client.client_fingerprint()
            and journal['valid'] and journal['complete'] and not journal['pending'],
            'incomplete or unbound official long-generation trial')
    observations = [json.loads(line) for line in (path.parent / 'requests.jsonl').read_text().splitlines()]
    leaves.suite.gen.warmup_reconciliation(observations, (name,))
    samples = [dict(e['sample'], phase=e['phase'], run=e['run'], turn=e['turn'])
               for e in observations if e['kind'] == 'result']
    sample, = leaves.suite.gen.collect_samples(generation)
    last = samples[-1]
    reference = request_reference(role)
    require(sample['controller_eligible'] and (sample['workload'], sample['run'], sample['order']) == (name, 0, 0)
            and last['phase'] == 'long-generation/' + name
            and last['request_sha256'] == reference['request_sha256']
            and last['read_tokens'] == reference['read_tokens'] and last['reused_tokens'] == 0
            and all(sample[k] == last[k] for k in (*leaves.WORK, 'decode_ms', 'prompt_ms', 'ttft_s')),
            'actual native request must match the predeclared body, extent and journal')
    basis.comparison.require_valid_client_ttft(last)
    fields = (*leaves.WORK, 'phase', 'run', 'turn', 'prompt_ms', 'decode_ms', 'ttft_s')
    return dict(samples=[{k: s[k] for k in fields} for s in samples], journal=journal,
                decode_rates={name: 1000 * last['generated_tokens'] / last['decode_ms']})


def census_inputs(log, samples, native):
    headers = list(re.finditer(footer.HEADER, log, re.M))
    bindings = []
    widths = []
    for i, (header, sample, counts) in enumerate(zip(headers, samples, native, strict=True)):
        frame = log[headers[i - 1].end() if i else 0:header.start()]
        windows = [(int(p), int(t)) for p, t in re.findall(r'^strata trace: window (\d+) (\d+)$', frame, re.M)]
        elapsed, = re.findall(footer.TIMING, frame, re.M)
        clocks = dict(zip(footer.LABELS, map(float, elapsed[1:]), strict=True))
        widths = [t for _, t in windows]
        require(len(windows) == int(elapsed[0]) > 0 and windows[0] == (sample['prompt_tokens'] - 1, 1)
                and all(1 <= t <= 4 and 0 <= p <= 262144 - t for p, t in windows)
                and all(1 <= b[0] - a[0] <= a[1] for a, b in zip(windows, windows[1:]))
                and sum(widths) - len(widths) == sample['drafts_offered'] and clocks['PCIe'] == 0,
                'wrong decode extent, observed draft count or expert tier')
        bindings.append(dict(prompt_tokens=sample['prompt_tokens'], windows=len(widths), rows=sum(widths),
                             cpu_jobs=counts['cpu_unique_misses'], cpu_entries=counts['cpu_entries'],
                             pool_ms=clocks['CPU_pool_run'] * len(widths)))
    return bindings, widths


def routes(log, samples, root):
    leaves.speed.validate_model_text(log, {'arm': 'retain'})
    native, _ = leaves.leaf_reader.evidence(log, samples, 4, False)
    bindings, widths = census_inputs(log, samples, native)
    row = bank.previous.read(root / 'record.json')
    metadata = row['routing_metadata']
    require(all(bank.sha(metadata[k]) == metadata[k + '_sha256'] for k in ('profile', 'manifest')),
            'stock profile or expert formats changed during capture')
    dirs = Path(row['plans'][0]['root'])
    target = samples[-1]
    require(target['generated_tokens'] >= 1024 and target['drafts_offered'] >= 256
            and len(samples) == 5, 'require four warm-ups then one sustained thinking request')
    headers = list(re.finditer(footer.HEADER, log, re.M))
    frame = log[headers[-2].end():headers[-1].start()]
    read = target['read_tokens']
    store = leaves.generation.model.restore.require_store(frame, dict(direct=False), dict(read=read))
    require(store['gpu_rows'] == read - 1 and store['window_mode'] == 10
            and len(re.findall(r'^strata DFlash2: eager, prompt_tail_rows=2046 cold_prefix=0 '
                               r'vram_bytes=[1-9][0-9]*$', frame, re.M)) == 1
            and len(re.findall(r'^strata cache loan: phase=ready generation=\d+ chunk=8192 first=132 '
                               r'lent=1124 decode_ready=1$', frame, re.M)) == 1,
            'long residual, DFlash2 tap or restored expert cache route differs')
    kv = leaves.speed.KV.check_request(frame, 0, read - 1, 8192, reuse=True,
                                       tap_layers=leaves.speed.gate.TAPPED_QSA_LAYERS)
    formats = census.manifest_formats(metadata['manifest'])
    membership = census.expected_membership(metadata['profile'])
    observed = census.extract(log, bindings, formats, membership)[-1]
    trace = dirs / ROUTING_FILE
    usage = routing.verifier_routing_usage(trace, widths, observed, formats, membership,
                                          max_bytes=row['budgets']['file_cap'])
    file = dirs / USAGE_FILE
    require(not file.exists(), 'unexpected existing routing evidence')
    payload = dict(request_sha256=target['request_sha256'], format_sha256=metadata['manifest_sha256'],
                   profile_sha256=metadata['profile_sha256'], native_source_commit=row['native_source_commit'],
                   usage=usage)
    bank.stage.write(file, payload)
    return dict(attention_store=store, kv=kv, windows=observed['windows'], target_rows=observed['rows'],
                cpu_jobs=observed['cpu_jobs'], cpu_entries=observed['cpu_entries'],
                logical_job_blob_bytes=observed['logical_job_blob_bytes'], jobs_NT=observed['jobs_NT'],
                membership_sha256=observed['membership_sha256'], distinct_layer_expert_pairs=len(usage),
                routing_bytes=trace.stat().st_size, routing_sha256=bank.sha(trace),
                usage_file=str(file), usage_sha256=bank.sha(file))


def decision(processes, role):
    process, = processes
    sample = process['evidence']['samples'][-1]
    fraction = sample['drafts_accepted'] / sample['drafts_offered']
    return dict(role=role, observed_acceptance=fraction, admitted_routing=bool(process['passed']),
                references=process['routes']['usage_sha256'])


def experiment(role):
    ctx = leaves.experiment()
    ctx.KEY = 'dflash2_long_routing_' + role + '_v1'
    ctx.ROOT = bank.B / ctx.KEY.replace('_', '-')
    ctx.THIS, ctx.ORDER = THIS, (role,)
    ctx.baseline = lambda: baseline(role)
    ctx.configuration, ctx.environment = configuration, environment
    ctx.command = lambda plan: command(plan, role)
    ctx.evidence_reader.read_complete_suite = lambda root, id_: evidence(root, id_, role)
    ctx.evidence_reader.decide = lambda processes: decision(processes, role)
    ctx.routes = lambda log, samples, arm: routes(log, samples, ctx.ROOT)
    original_pins = ctx.pins

    def pins(revision):
        result = original_pins(revision)
        for name in (THIS, *probe.READERS):
            require((bank.P / name).read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'changed routing or trial source: ' + name)
            result[name] = bank.sha(bank.P / name)
        return result

    ctx.pins = pins
    ctx.protocol = dict(case=role, workload=ROLES[role], order=list(ctx.ORDER),
        diagnostic='Collect one fresh stock-layout official long request after the four standard warm-ups. '
                   'Keep thinking, confidence 0.6, per-layer slots, all providers and finite suite budgets.',
        acceptance='Conserve reconstructed per-window CPU jobs by format and multiplicity after cache refill. '
                   'Keep the exact body, coverage and logical workload. Use the separate held-out request '
                   'only to forecast a rank selected from training data. Obtain clean performance separately.',
        stop='Stop on changed sources, frame, owner, device, work, runtime or finite budget. Close original '
             'evidence before the next GPU job. Review private dumps after the profile is scored.')
    return ctx


if __name__ == '__main__':
    role, = sys.argv[1:2]
    require(role in ROLES, 'use train or holdout, then the planned stage')
    del sys.argv[1]
    clock.main(experiment(role))
