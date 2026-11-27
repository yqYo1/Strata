"""Measure the stock DFlash2 memory boundary with a coherent observer build."""

import re
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import dflash2_decode_ranked_generation as ranked
import dflash2_primary_prefill_curve as curve
import dflash2_primary_retained as retained

bank, basis, require, speed = retained.bank, retained.basis, retained.require, retained.speed
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_decode_handoff_memory_v1'
ROOT = bank.B / KEY.replace('_', '-')
BUILD_KEY = 'dflash2_decode_handoff_native_v1'
BUILD_ROOT = bank.B / BUILD_KEY.replace('_', '-')
NATIVE = 'b72ed4fb09c10f11fafa5ce37dd24a7ef61f83eb'
GENERATOR = 'src/program/generate.cpp'
CURVE = '71ddd9046'
REPLAY = 'dflash2_decode_extra_capacity_replay_v1'
ORDER = (94209, 126977)
RUNS = 2
PHASES = ('before-prefill', 'after-refill', 'after-draft-activation', 'after-decode')
MEMORY = 'strata DFlash2 handoff memory: '
PLAN = dict(order=list(ORDER), requests_per_process=RUNS,
    scope='Fresh coherent copy of the primary build with just the handoff observer. Run the same exact '
          'long-prefill requests and aligned-KV retention on the qualified stock global profile.',
    gates='Require actual all-GPU mode16 prefill, matched request bodies, unchanged expert arena and '
          'all four ordered memory queries. Provider, validation, checked teardown and fault checks remain.',
    decision='Record the two post-activation queries and actual allocation limit separately from '
             'startup/refill. Compare with the replay slice only to select a bounded admission test. '
             'No allocation, throughput, full-context or model-route qualification.',
    retention='Keep the two existing phase references. This is the one temporary diagnostic generation '
              'through memory capture; retire it after committing the measurements and verifying recovery.')


def build_controller():
    path = bank.P / 'bench/dflash2_cold_host_build.py'
    ctx = basis.core.source.device.load_module('decode_handoff_native_build', path)
    ctx.ROOT, ctx.KEY, ctx.TREE = BUILD_ROOT, BUILD_KEY, BUILD_ROOT / 'source'
    ctx.OVERLAY, ctx.OVERLAYS = NATIVE, (GENERATOR,)
    ctx.CONTROLLERS = (*ctx.CONTROLLERS, THIS, curve.THIS, ranked.THIS, ranked.restore.THIS,
                       ranked.restore.quota.THIS, 'tools/make_profile.py')

    def prepare():
        require(not BUILD_ROOT.exists() and BUILD_KEY not in basis.validation.committed(bank.RESULT),
                'fresh handoff diagnostic build required')
        require(not (bank.B / 'dflash2-cold-layer-native-v1/build/strata').exists(),
                'the cold-layer observer already owns the temporary comparison generation')
        revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
        fields, payload, _, _ = ctx.inputs(revision)
        require(payload[GENERATOR] == (bank.P / GENERATOR).read_bytes(), 'handoff observer source changed')
        BUILD_ROOT.mkdir(mode=0o700)
        for name in ('owned', 'tmp', 'cache'):
            (BUILD_ROOT / name).mkdir(mode=0o700)
        for name, raw in payload.items():
            target = ctx.TREE / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        row = dict(fields, stage='prepared', commands=[], active=False, passed=False, complete=False,
                   ownership_closed=True, scope=PLAN['scope'], retention=PLAN['retention'])
        ctx.check(row, committed=False)
        ctx.save(row)

    ctx.prepare = prepare
    return ctx


def baseline(build):
    base = retained.baseline()
    native = bank.previous.read(BUILD_ROOT / 'record.json')
    require(retained.closed(native, 'built') and native['base_binary'] == base['binary']
            and native['base_build_sha256'] == base['native_build_sha256']
            and native['native_base_commit'] == base['native_source_commit']
            and native['overlay_commit'] == NATIVE, 'exact coherent handoff observer required')
    build.check(native)
    stock = ranked.restore_proof(base, ranked.restore.KEY, ranked.restore.ORDER)
    original = bank.previous.historic(CURVE, curve.KEY)
    require(retained.closed(original, 'closed') and original['binary'] == base['binary'],
            'recover the same-build length reference')
    hashes = {}
    for read in ORDER:
        first, second = [p for p in original['processes'] if p['read'] == read]
        keys = [s['request_sha256'] for s in first['samples'][:RUNS]]
        require(len(keys) == RUNS and keys == [s['request_sha256'] for s in second['samples'][:RUNS]],
                'original repeated request bodies differ')
        hashes[str(read)] = keys
    replay = basis.validation.committed(bank.RESULT, (REPLAY,))
    require((replay['fixed_slots'], replay['fixed_bytes']) ==
            (ranked.restore.quota.EXPECTED_SLOTS, ranked.restore.quota.EXPECTED_BYTES), 'stock arena differs from replay')
    first_cap = replay['cases'][1]
    require(first_cap['cap_bytes'] == 256 << 20 and 0 < first_cap['extra_bytes'] <= first_cap['cap_bytes'],
            'expected the first qualified replay budget')
    base['base_physical_context_sha256'] = base.pop('physical_context_sha256')
    return dict(base, binary=native['binaries']['strata'], native_source_commit=NATIVE,
                native_build_sha256=bank.signature(build.builder.compact(native)),
                base_native_source_commit=base['native_source_commit'],
                native_overlay_sha256=bank.signature(native['overlays']), native_context_qualified=False,
                qualified_profiles={'stock': ranked.qualified_profile('stock', stock, None)},
                stock_restore_sha256=bank.signature(stock), fixed_bodies=hashes, extra_bytes=first_cap['extra_bytes'])


def memory_phases(text, read):
    lines = [line.removeprefix(MEMORY) for line in text.splitlines() if line.startswith(MEMORY)]
    require(len(lines) == len(PHASES), 'missing or duplicate DFlash2 handoff phases')
    phases = {}
    for phase, line in zip(PHASES, lines, strict=True):
        match = re.fullmatch(r'phase=([a-z-]+) prompt=([0-9]+) read_from=([0-9]+) free_bytes=([0-9]+) '
                             r'total_bytes=([0-9]+) expert_slots=([0-9]+) expert_bytes=([0-9]+)', line)
        require(match is not None and match[1] == phase, 'handoff phase order or fields changed')
        n, resume, free, total, slots, size = map(int, match.groups()[1:])
        require((n, resume, slots, size) == (read, 0, ranked.restore.quota.EXPECTED_SLOTS,
                ranked.restore.quota.EXPECTED_BYTES) and 0 <= free <= total and total > size,
                'handoff output disagrees with the request or the stock arena')
        phases[phase] = dict(free_bytes=free, total_bytes=total)
    require(len({p['total_bytes'] for p in phases.values()}) == 1
            and text.index(MEMORY + 'phase=before-prefill ') < text.index('strata MoE split admission: ')
            < text.index('strata MoE split complete: ') < text.index(MEMORY + 'phase=after-refill ')
            < text.index('strata DFlash2: eager, ') < text.index(MEMORY + 'phase=after-draft-activation ')
            < text.index(MEMORY + 'phase=after-decode '), 'handoff telemetry escaped the request phases')
    return phases


def request_route(text, plan):
    frame = curve.route(text, plan)
    split = speed.workspace.one(text, 'strata MoE split admission: ')
    maximum = int(split['max_single_allocation_bytes'])
    phases = memory_phases(text, plan['read'])
    require(0 < maximum <= phases['before-prefill']['total_bytes'], 'invalid single-allocation limit')
    return dict(frame, memory=phases, max_single_allocation_bytes=maximum)


def environment(row, plan):
    return dict(curve.environment(row, plan), STRATA_SYCL_DECODE_HANDOFF_MEMORY='1')


def experiment():
    build = build_controller()
    ctx = speed.clock.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.protocol, ctx.environment = lambda: baseline(build), PLAN, environment
    ctx.configuration = lambda row, plan: ranked.configuration(row, dict(plan, arm='stock'))
    curve_pins, prior_save, prior_current = curve.experiment().pins, ctx.save, ctx.current

    def command(plan):
        cmd = curve.command(plan)
        cmd[cmd.index('--runs') + 1] = str(RUNS)
        return cmd

    def pins(revision):
        result = curve_pins(revision)
        for name in (THIS, retained.THIS, ranked.THIS, ranked.restore.THIS,
                     ranked.restore.quota.THIS, 'tools/make_profile.py', 'bench/dflash2_cold_host_build.py'):
            path = bank.P / name
            require(path.read_bytes() == basis.validation.git('show', revision + ':' + name), 'changed handoff input: ' + name)
            result[name] = bank.sha(path)
        return result

    def save(row):
        if row['stage'] == 'prepared':
            row['request_hashes'] = dict(row['fixed_bodies'])
            for plan in row['plans']:
                plan.update(read=plan['arm'], heads=speed.HEADS, kv_arm='retain')
        prior_save(row)

    def current(row):
        require(row['request_hashes'] == row['fixed_bodies'] and all((p['read'], p['heads'], p['kv_arm'])
                == (p['arm'], speed.HEADS, 'retain') for p in row['plans']), 'frozen observer requests changed')
        return prior_current(row)

    def after_execution(outcome, row, plan):
        for frame, sample in zip(outcome['routes'], outcome['samples'], strict=True):
            require(frame['native_request'] == [plan['read'], 0, plan['read'], 24, sample['drafts_accepted'],
                    sample['drafts_offered']] and sample['drafts_offered'] > 0, 'native drafter work differs')
        outcome.update(clean_speed=False, extra_bytes=row['extra_bytes'])

    def decision(processes):
        require(tuple(p['read'] for p in processes) == ORDER and all(p['passed'] and p['ownership_closed']
                and not p['fresh_faults'] for p in processes)
                and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ORDER),
                'need two independently closed diagnostic processes')
        frames = [f for p in processes for f in p['routes']]
        free = min(f['memory'][phase]['free_bytes'] for f in frames for phase in PHASES[2:])
        limit = min(f['max_single_allocation_bytes'] for f in frames)
        needed = processes[0]['extra_bytes']
        return dict(min_reported_decode_free_bytes=free, min_single_allocation_limit_bytes=limit,
                    replay_extra_bytes=needed, observed_arithmetic_margin_bytes=min(free, limit) - needed,
                    allocation_verified=False, clean_speed_qualified=False, adopted=False)

    def run(index):
        row = bank.previous.read(ROOT / 'record.json')
        runner = basis.core.source.device.load_module('decode_handoff_observer', basis.comparison.__file__)
        original = runner.shared.models
        runner.shared = SimpleNamespace(**vars(runner.shared))

        def model_pins(cfg, assets, full=False):
            original.model_pins(row['configuration'], assets, full=full)
            for flag in ('--draft-model', '--mtp-draft-vocab'):
                path = cfg['args'][cfg['args'].index(flag) + 1]
                require(bank.sha(path) == row['dflash2_assets'][path], 'DFlash2 asset changed')

        def validate_model(text, _):
            ranked.restore.check_loader(text, row['qualified_profiles']['stock']['path'], per_layer=False)
            require(text.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1
                    and not re.search(r'(?m)^strata mtp: .*loaded|strata loan audit:', text), 'unexpected native route')

        runner.shared.models = SimpleNamespace(model_pins=model_pins)
        runner.fingerprint = bank.signature
        policy = SimpleNamespace(ROOT=ROOT, ORDER=[(n, speed.HEADS) for n in ORDER], RUNS=RUNS,
            current=current, native_context=lambda r, _: r, environment=environment, save=save,
            request_route=request_route, after_execution=after_execution, decision=decision, validate_model_text=validate_model)
        runner.run(index, policy)

    ctx.command, ctx.pins, ctx.save, ctx.current, ctx.run = command, pins, save, current, run
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    if action.startswith('build-'):
        build_controller().main(action.removeprefix('build-'))
    else:
        speed.clock.main(experiment())
