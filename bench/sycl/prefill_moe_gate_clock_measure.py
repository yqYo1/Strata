"""Bracket the exact prefill MoE status-to-Gate clocks with clean requests."""

import fcntl
import math
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
import prefill_kv_upload_compare as upload
import prefill_moe_gate_clock_build as build

bank, basis, require = upload.bank, upload.basis, upload.require
model = basis.comparison
READ, RUNS, LAYERS, CHUNK = upload.READ, 2, 48, 8192
ROLES = ('clean-before', 'observed', 'clean-after')
KEY = 'prefill_moe_gate_clock_model_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
TIMELINE = '4bc26aaa9cbdd8d4084336826361aae7bb886449'
TIMELINE_KEY = 'prefill_kv_upload_model_pti_v1'
SELECTOR = 'STRATA_PREFILL_MOE_GATE_CLOCK'
PREFIX = 'strata prefill MoE gate clock: '
REGIONS = ('copy', 'wait', 'handoff', 'projection', 'dequant')
FIELDS = tuple(f'{region}_{clock}' for region in REGIONS for clock in ('wall', 'cpu'))
RECORD = re.compile(
    r'layer=(\d+) first=(\d+) rows=(\d+) chunk=(\d+) valid=1'
    + ''.join(rf' {field}=([\d]+\.[\d]{{9}})' for field in FIELDS))
MAX_OBSERVER_SECONDS = (3.0, 1.5)
BUDGET = dict(wall=1800, cpu=21600, rss_cap=104 << 30, text_cap=16 << 20,
              file_cap=16 << 20, total_cap=64 << 20)
PLAN = dict(roles=list(ROLES), read=READ, requests=RUNS,
    scope='Same V2 split128, batched H2D, direct attention output, KV reuse, scalar gather and decode. '
          'Collect disjoint calling-thread wall and CPU intervals around the exact D2H, checked wait, '
          'post-wait handoff and first Gate dequant wrapper per successful layer.',
    acceptance='All arms match the original two fresh thinking requests, 48 MoE windows, actual dispatch, '
               '24 output tokens, draft counts, response digests and unchanged runtime. '
               'Require ordered complete boundary records only in the observed arm. '
               'Check first-use and second-request prompt/TTFT differences against both clean controls. '
               'A clocked request may qualify independently of the other request position.',
    max_observer_seconds=list(MAX_OBSERVER_SECONDS),
    stopping='Stop on the first identity, work, boundary, validation, fault or finite budget failure. '
             'Keep every eligible sample. Finish the bracket before assessing clock cost.',
    limits='Host wrapper intervals. Copy enqueue can wait internally. Checked wait includes pending GPU work. '
           'The dequant interval includes argument checks and submission. No device service, PCIe '
           'saturation, removable-time or speedup claim follows from host intervals alone.')


def clock_summary(text, observed):
    lines = [line.removeprefix(PREFIX) for line in text.splitlines() if line.startswith(PREFIX)]
    require(len(lines) == (LAYERS if observed else 0), 'missing or extra prefill MoE Gate clocks')
    if not observed:
        return None
    points = []
    for layer, line in enumerate(lines):
        match = RECORD.fullmatch(line)
        require(match and tuple(map(int, match.groups()[:4])) == (layer, 0, READ - 1, CHUNK),
                'MoE clock order, row range or validity changed')
        values = dict(zip(FIELDS, map(float, match.groups()[4:]), strict=True))
        require(all(math.isfinite(v) and v >= 0 for v in values.values())
                and all(values[r + '_cpu'] <= values[r + '_wall'] + 0.0002 for r in REGIONS),
                'invalid wall or calling-thread CPU interval')
        points.append(values)
    return dict(layers=LAYERS, seconds={k: sum(p[k] for p in points) for k in FIELDS},
                handoff_wall_median_s=statistics.median(p['handoff_wall'] for p in points),
                handoff_wall_max_s=max(p['handoff_wall'] for p in points))


def admitted_basis():
    parent, prior, providers, extra = upload.store_basis(RUNS, False)
    timeline = basis.validation.committed(bank.RESULT, (TIMELINE_KEY,))
    require(timeline == bank.previous.historic(TIMELINE, TIMELINE_KEY) and upload.gather.closed(timeline)
            and timeline['decision']['normal_route_attribution_qualified']
            and timeline['binary'] == parent['binaries']['strata']
            and timeline['native_source_commit'] == parent['native_source_commit'],
            'recover the matched V2 PTI route and original work')
    native = bank.previous.read(build.ROOT / 'record.json')
    build.context(native['source_commit']).check(native)
    require(native['stage'] == 'built' and native['passed'] and native['complete']
            and native['ownership_closed'] and not native['active'] and native['host_control']['passed']
            and native['projection']['clock_parent_build_sha256'] == bank.signature(bank.builder.compact(parent))
            and native['driver'] == parent['driver'] == prior['driver'],
            'complete the checked clock build and preserve the V2 baseline')
    proof = prior['observer_admission']
    work = upload.previous.direct_work(proof['expected_work'])
    require(len(work) == RUNS and [w['request'] for w in work] == timeline['request_hashes'][str(READ)]
            and all([upload.previous.split.work(r, s) for r, s in zip(p['routes'], p['samples'], strict=True)] == work
                    for p in timeline['processes']), 'unchanged V2 work across all PTI arms required')
    proof.update(expected_work=work, timeline=TIMELINE, timeline_sha256=bank.signature(timeline),
                 clock_build_sha256=bank.signature(bank.builder.compact(native)))
    return native, prior, providers, extra


def decision(processes):
    require([p['role'] for p in processes] == list(ROLES)
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == 3
            and all(p['passed'] and p['ownership_closed'] and not p['fresh_faults']
                    and p['native_exit_code'] == 0 for p in processes)
            and [p['clean_speed'] for p in processes] == [True, False, True],
            'complete matched clock bracket with three independent owners')
    points = []
    for i, bound in enumerate(MAX_OBSERVER_SECONDS):
        sample = processes[1]['samples'][i]
        deltas = [dict(prompt_s=(sample['prompt_ms'] - p['samples'][i]['prompt_ms']) / 1000,
                       ttft_s=sample['ttft_s'] - p['samples'][i]['ttft_s'])
                  for p in (processes[0], processes[2])]
        points.append(dict(scope='first-use' if i == 0 else 'second-request',
            observed_minus_clean_seconds=deltas, max_absolute_seconds=bound,
            overhead_qualified=all(abs(v) <= bound for d in deltas for v in d.values())))
    return dict(clock_cost=points, measured_host_intervals=True, speedup_qualified=False)


def experiment():
    ctx = basis.core.source.device.load_module('v2_model_moe_gate_clock', basis.core.__file__)
    base = upload.experiment('store-time')
    ctx.THIS, ctx.ROOT, ctx.KEY = THIS, ROOT, KEY
    ctx.ORDER, ctx.READS, ctx.RUNS, ctx.CLEAN_FAST = [(READ, 1)] * len(ROLES), (READ,) * len(ROLES), RUNS, True
    ctx.basis, ctx.decision, ctx.check_sample = admitted_basis, decision, base.check_sample
    original_current, original_save = ctx.current, ctx.save

    def pins(revision):
        return base.pins(revision) | bank.stage.source.pin(bank.P, [THIS, build.THIS, build.CLOCK])['source_sha256']

    def environment(row, plan):
        env = base.environment(row, dict(plan, index=1))
        require(SELECTOR not in env, 'the measured V2 environment unexpectedly selects the clock')
        env[SELECTOR] = str(int(plan['index'] == 1))
        return env

    def request_route(text, plan):
        route = upload.request_route(text, dict(plan, direct_output=True, mode='batched'))
        summary = clock_summary(text, plan['index'] == 1)
        if summary:
            route['gate_clock'] = summary
        return route

    def current(row):
        require(row['plan'] == PLAN and row['budgets'] == BUDGET
                and row['whole_bytes'] == len(ROLES) * BUDGET['total_cap'] + (8 << 20)
                and [(p['index'], p['role'], p['mode'], p['direct_output'], p['trace_progress'])
                     for p in row['plans']] == [(i, role, 'batched', True, False) for i, role in enumerate(ROLES)],
                'the frozen clock bracket differs')
        return original_current(row)

    def after_execution(outcome, row, plan):
        require([upload.previous.split.work(r, s) for r, s in
                 zip(outcome['routes'], outcome['samples'], strict=True)] == row['observer_admission']['expected_work'],
                'clocked route changed the actual request or GPU work')
        samples_path, = (Path(plan['root']) / 'bench').glob('*/results.json')
        samples = bank.previous.read(samples_path)['results']['prefill_only']['requests']
        require([s['request_sha256'] for s in samples] == [s['request_sha256'] for s in outcome['samples']],
                'response digests refer to different requests')
        digests = [[s['reasoning_sha256'], s['output_sha256']] for s in samples]
        require(not row['processes'] or digests == row['processes'][0]['response_sha256'],
                'observer changed the fixed short diagnostic response')
        outcome.update(role=plan['role'], clean_speed=plan['index'] != 1, response_sha256=digests)

    def save(row):
        if row['stage'] == 'prepared':
            row.update(plan=PLAN, budgets=BUDGET, whole_bytes=len(ROLES) * BUDGET['total_cap'] + (8 << 20))
            row['request_hashes'][str(READ)] = [w['request'] for w in row['observer_admission']['expected_work']]
            for plan, role in zip(row['plans'], ROLES, strict=True):
                plan.update(role=role, mode='batched', direct_output=True, trace_progress=False)
                plan['environment_sha256'] = model.fingerprint(environment(row, plan))
        original_save(row)

    ctx.pins, ctx.environment, ctx.request_route = pins, environment, request_route
    ctx.current, ctx.after_execution, ctx.save = current, after_execution, save
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON
            and action in ('prepare', 'host', '0', '1', '2'), 'select one frozen clock measurement stage')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment()
        if action == 'prepare':
            ctx.prepare()
        elif action == 'host':
            model.host(ctx)
        else:
            model.run(int(action), ctx)
