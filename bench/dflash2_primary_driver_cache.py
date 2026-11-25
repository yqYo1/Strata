"""Isolate the primary DFlash2 route's private NEO disk-cache contribution."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import statistics
import sys
import threading
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
import dflash2_primary_prefill_curve as curve
import prefill_driver_cache as driver_cache

speed, bank, basis, require = curve.speed, curve.bank, curve.basis, curve.require
clock = speed.clock
KEY = 'dflash2_primary_driver_cache_v1'
ROOT = bank.B / KEY.replace('_', '-')
THIS = str(Path(__file__).resolve().relative_to(bank.P))
READ, RUNS, DONOR_INDEX = 102401, 2, 3
DONOR = '0577a8bbca081ba5ec5d03d32d3ed4376d7d553f'
DONOR_CACHE = (54, 12434952, 'c93c2b41a2fc8642b3abc00424449f3466c584634b971c36b28258f5206b3ddb')
TIMED = ('empty', 'seed', 'seed', 'empty', 'empty', 'seed')
ORDER = ('empty', 'seed', *TIMED)
PLAN = dict(order=list(ORDER), diagnostics=[0, 1], read=READ, requests_per_process=RUNS,
    scope='Use a complete private NEO cache from the closed same-build 102401-token run. '
          'Compare it with a fresh empty private cache directory. Preserve the current engine, '
          'providers, pre-exec safety controls, selected KV retention and exact requests.',
    diagnostics_policy='Enable NEO statistics only in the first two processes. Require more '
                       'hits and fewer misses in the seeded first-long request before clean timing.',
    criteria='Three opposite-order pairs, statistics disabled in both arms. Require a first-use gain '
             'when at least two pairs improve and their median time ratio is below one. Report '
             'server readiness, prefill, client TTFT, cache copy cost and subsequent requests separately. '
             'Record first-use 1000 tok/s independently.',
    limits='Measures the effect of a NEO disk seed. In-memory SYCL program reuse, module creation and '
           'expert page-in are not directly attributed. No automatic adoption or decode qualification.',
    stop='Stop on failed diagnostics, changed work/route, ownership, validation, runtime, budget or driver fault.')


def baseline():
    row = curve.baseline()
    old = bank.previous.historic(DONOR, curve.KEY)
    donor = old['processes'][DONOR_INDEX]
    plan = old['plans'][DONOR_INDEX]
    require(row['selected_arm'] == 'retain' == old['selected_arm'] == plan['kv_arm']
            and all(row[k] == old[k] for k in ('boot', 'driver', 'binary', 'native_source_commit',
                                              'native_build_sha256', 'runtime_manifest', 'native_validation_probe',
                                              'dflash2_assets', 'template'))
            and bank.signature(row['model_assets']) == old['model_assets_sha256']
            and bank.signature(row['environment_template']) == old['environment_template_sha256']
            and donor['read'] == plan['read'] == READ and donor['binary'] == row['binary']
            and donor['native_exit_code'] == 0 and donor['complete'] and donor['passed']
            and donor['clean_speed'] and donor['ownership_closed'] and not donor['fresh_faults'],
            'the seed must come from the exact closed primary route')
    source = Path(plan['root']) / 'cache' / 'neo_compiler_cache'
    require(source == curve.ROOT / str(DONOR_INDEX) / 'cache' / 'neo_compiler_cache',
            'the donor cache must belong to its owned launch')
    receipt = bank.previous.read(Path(plan['root']) / 'engine-exit.json')
    require(receipt['native'] == donor['native_identity'] and receipt['exit_code'] == 0
            and receipt['binary'] == donor['binary'], 'the cache donor native exit differs')
    require(bank.signature(curve.environment(row, plan)) == plan['environment_sha256'],
            'the donor cache environment differs from the original launch')
    require(driver_cache.inventory(source) == DONOR_CACHE, 'the closed donor NEO inventory changed')
    return dict(row, seed=dict(path=str(source), record_commit=DONOR, inventory=list(DONOR_CACHE)),
                donor_work=work(donor['routes'][:RUNS]),
                donor_request_sha256s=[s['request_sha256'] for s in donor['samples'][:RUNS]])


def command(plan):
    args = speed.command(plan)
    args[args.index('--prefill-exact-tokens') + 1] = str(READ)
    args[args.index('--runs') + 1] = str(RUNS)
    return args


def environment(row, plan):
    env = curve.environment(row, plan)
    require(env['SYCL_CACHE_PERSISTENT'] == '0' and not any(k.startswith('NEO_CACHE_') for k in env),
            'the NEO disk-cache contrast must be independent of SYCL disk cache')
    return dict(env, NEO_CACHE_DIR=str(Path(plan['root']) / 'neo'), NEO_CACHE_PERSISTENT='1',
                NEO_CACHE_STATS=str(int(plan['index'] < 2)))


def work(routes):
    return [{key: value for key, value in route.items() if key != 'free_bytes'} for route in routes]


def hit_gate(empty, seed):
    def delta(process, key):
        all_counts = process['driver_stats']
        return all_counts['end'][key] - all_counts['begin'][key]

    miss = [delta(p, 'misses') for p in (empty, seed)]
    hits = [delta(p, 'hits') for p in (empty, seed)]
    require(empty['seed_arm'] == 'empty' and seed['seed_arm'] == 'seed'
            and hits[1] > hits[0] >= 0 and miss[0] > miss[1] >= 0,
            'statistics do not establish the first-long disk-cache contrast')
    return dict(empty=dict(hits=hits[0], misses=miss[0]), seeded=dict(hits=hits[1], misses=miss[1]))


def decision(processes):
    require(tuple(p['seed_arm'] for p in processes) == ORDER
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ORDER)
            and all(p['passed'] and p['complete'] and p['ownership_closed'] and p['native_exit_code'] == 0
                    and not p['fresh_faults'] and p['clean_speed'] == (i >= 2)
                    for i, p in enumerate(processes)), 'complete independent diagnostic and clean pairs required')
    ratios = []
    for x, y in zip(processes[2::2], processes[3::2], strict=True):
        empty, seed = (x, y) if x['seed_arm'] == 'empty' else (y, x)
        first, warm = seed['samples']
        cold, later = empty['samples']
        ratios.append(dict(readiness_upper=seed['readiness']['upper_s'] / empty['readiness']['lower_s'],
                           short_warmup=seed['short_warmup_s'] / empty['short_warmup_s'],
                           first_prefill=first['prompt_ms'] / cold['prompt_ms'], first_TTFT=first['ttft_s'] / cold['ttft_s'],
                           subsequent_prefill=warm['prompt_ms'] / later['prompt_ms'], subsequent_TTFT=warm['ttft_s'] / later['ttft_s']))
    repeat_gain = {k: sum(p[k] < 1 for p in ratios) >= 2 and statistics.median(p[k] for p in ratios) < 1 for k in ratios[0]}
    return dict(driver_contrast=hit_gate(*processes[:2]), seeded_over_empty_ratios=ratios, repeated_gain=repeat_gain,
                first_1000={arm: all(p['samples'][0]['prompt_ms'] <= READ for p in processes[2:] if p['seed_arm'] == arm)
                            for arm in ('empty', 'seed')}, adopted=False)


def experiment():
    ctx = clock.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.protocol, ctx.baseline = PLAN, baseline
    ctx.configuration, ctx.command, ctx.environment = speed.generation.configuration, command, environment
    base_pins, base_current, base_save = speed.experiment().pins, ctx.current, ctx.save

    def pins(revision):
        entries = base_pins(revision)
        for name in (THIS, curve.THIS, driver_cache.THIS):
            file = bank.P / name
            require(file.read_bytes() == basis.validation.git('show', revision + ':' + name),
                    'commit exact cache comparison input: ' + name)
            entries[name] = bank.sha(file)
        return entries

    def save(row):
        if row['stage'] == 'prepared':
            row['request_hashes'] = {str(READ): row['donor_request_sha256s']}
            for plan in row['plans']:
                plan.update(read=READ, heads=1, kv_arm='retain')
        base_save(row)

    def current(row):
        require(row['request_hashes'] == {str(READ): row['donor_request_sha256s']}
                and all((p['read'], p['heads'], p['kv_arm']) == (READ, 1, 'retain') for p in row['plans']),
                'cache comparison input/route changed')
        if len(row['processes']) >= 2 and row['stage'] != 'failed':
            hit_gate(*row['processes'][:2])
        return base_current(row)

    def run(index):
        runner = basis.core.source.device.load_module('primary_neo_cache_timing', basis.comparison.__file__)
        row = bank.previous.read(ROOT / 'record.json')
        plan = row['plans'][index]
        folder = Path(plan['root']) / 'neo'
        old_models = runner.shared.models
        runner.shared = SimpleNamespace(**vars(runner.shared))
        stopped = threading.Event()
        with ThreadPoolExecutor(max_workers=1) as thread:
            ready = None

            def model_pins(config, assets, full=False):
                old_models.model_pins(row['configuration'], assets, full=full)
                for flag in ('--draft-model', '--mtp-draft-vocab'):
                    path = config['args'][config['args'].index(flag) + 1]
                    require(bank.sha(path) == row['dflash2_assets'][path], 'actual DFlash2 asset changed')

            def lifetime(outcome, edge, _):
                nonlocal ready
                if edge != 'begin':
                    return
                require(not folder.exists() and not folder.is_symlink(), 'fresh private NEO directory required')
                start = time.monotonic()
                if plan['arm'] == 'seed':
                    driver_cache.private_copy(Path(row['seed']['path']), folder, tuple(row['seed']['inventory']))
                else:
                    folder.mkdir(mode=0o700)
                outcome.update(seed_arm=plan['arm'], clean_speed=index >= 2,
                               prepare_directory_s=time.monotonic() - start)
                ready = thread.submit(driver_cache.read_ready, plan, stopped)

            def gate(outcome, request, edge, _):
                if request != 0:
                    return
                if edge == 'begin':
                    outcome['readiness'] = ready.result(timeout=1)
                    require(outcome['readiness']['native'] == outcome['native_identity'], 'wrong ready owner')
                if index < 2:
                    outcome.setdefault('driver_stats', {})[edge] = driver_cache.counters(folder)

            def after_execution(outcome, row, _plan):
                require(work(outcome['routes']) == row['donor_work']
                        and outcome['request_routes_checked_before_continue'] == RUNS
                        and driver_cache.inventory(folder, stats=index < 2) == tuple(row['seed']['inventory']),
                        'timed work, route or final cache payload differs from the donor')
                journal, = (Path(plan['root']) / 'bench').glob('*/requests.jsonl')
                events = [json.loads(line) for line in journal.read_text().splitlines()]
                warmup, = (e['sample'] for e in events if e['kind'] == 'result' and e['phase'] == 'warm-up/chat')
                interval = warmup['client_interval']
                require(warmup['ok'] and warmup['stream_complete'] and interval['request_begin_monotonic_ns']
                        < interval['first_output_monotonic_ns'] <= interval['request_end_monotonic_ns'], 'invalid short warm-up')
                outcome['short_warmup_s'] = (interval['request_end_monotonic_ns'] - interval['request_begin_monotonic_ns']) / 1e9
                outcome['warmup_work'] = {key: warmup[key] for key in ('request_sha256', 'read_tokens', 'reused_tokens',
                    'generated_tokens', 'drafts_accepted', 'drafts_offered')}
                require(not index or outcome['warmup_work'] == row['processes'][0]['warmup_work'], 'short warm-up changed')
                if index == 1:
                    hit_gate(row['processes'][0], outcome)

            def validate_model(text, _):
                require(text.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1
                        and not re.search(r'(?m)^strata mtp: .*loaded|strata loan audit:|strata DFlash2 tap payload:', text),
                        'unexpected head, MTP layer or observer')

            runner.shared.models = SimpleNamespace(model_pins=model_pins)
            runner.fingerprint = bank.signature
            policy = SimpleNamespace(ROOT=ROOT, ORDER=[(READ, 1)] * len(ORDER), RUNS=RUNS,
                current=current, native_context=lambda row, _: row, environment=environment, save=save,
                request_route=curve.route, observe_lifetime=lifetime, observe_gate=gate,
                after_execution=after_execution, validate_model_text=validate_model, decision=decision)
            try:
                runner.run(index, policy)
            finally:
                stopped.set()

    ctx.pins, ctx.current, ctx.save, ctx.run = pins, current, save, run
    return ctx


if __name__ == '__main__':
    clock.main(experiment())
