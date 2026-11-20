"""Separate driver-cache effects on server readiness, short warm-up and first long prefill."""

from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import struct
import sys
import threading
import time

sys.dont_write_bytecode = True
import prefill_split_length_curve as curve

speed, bank, basis, require = curve.speed, curve.bank, curve.basis, curve.require
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY, ROOT = 'prefill_driver_cache_v1', bank.B / 'prefill-driver-cache-v1'
READ, RUNS = 117489, 2
BLOCKS = (('empty', 'short', 'full'), ('short', 'full', 'empty'), ('full', 'empty', 'short'))
SEEDS = ('empty', 'empty', 'short', 'full', *(seed for block in BLOCKS for seed in block))
DIAGNOSTIC = (1, 2, 3)
SHORT = (41, 9522856, 'a26c48d6b0065425deb9f01e47f483d3aee1e5489d1bf5919caa21fb88d5a3cf')
FULL = (53, 10461608, '12438bf9e0c669cb1a93c09e27fc82b83e51f10677ac7673e1c5e68d094e3414')
PLAN = dict(order=list(SEEDS), diagnostic=list(DIAGNOSTIC), blocks=list(map(list, BLOCKS)),
    producer='Process zero creates two complete driver directories, before the first long request and after exit. '
             'Require the actual inventory to match the short/full expectations before making copies.',
    controls='Fresh private directory for every process, deliberate NEO cache enable in every arm, SYCL disk cache off. '
             'Distinguish readiness at the server log, short warm-up, and first and subsequent fresh long requests. '
             'Verify the direction of actual driver hit and miss changes separately with statistics on.',
    acceptance='Compare full/empty, short/empty and full/short within three balanced blocks. For each clock, '
               'report all paired ratios and require at least two gains plus a median time ratio below one. '
               'Resolve the startup clock only for disjoint observation bounds. Report seed copy time separately. '
               'No speed exclusions. A first-use benefit qualifies only this restart setup; decode adoption remains open.')


def inventory(directory, *, stats=False):
    """Hash only the device images. Validate the driver's other files independently."""
    require(directory.is_dir() and not directory.is_symlink(), 'missing private driver cache')
    image_hash = hashlib.sha256()
    count = size = 0
    paths = sorted(directory.rglob('*'))
    require(len(paths) <= 320 and not any(p.is_symlink() for p in paths), 'unexpected cache tree')
    for path in paths:
        name = path.relative_to(directory).as_posix()
        if path.is_dir():
            require(re.fullmatch(r'[0-9a-f](/[0-9a-f])?', name), 'unrecognized NEO cache directory')
        elif path.name == 'stats':
            require(stats and path.is_file() and path.stat().st_size == 24, 'unexpected statistics output')
        elif name != 'config.file':
            require(path.is_file() and re.fullmatch(r'[0-9a-f]/[0-9a-f]/[0-9a-f]{16}\.l0_cache', name)
                    and name[0] + name[2] == path.stem[:2], 'unrecognized NEO device image')
            count += 1
            size += path.stat().st_size
            require(size <= 16 << 20, 'image budget exceeded')
            image_hash.update(name.encode() + b'\0' + hashlib.sha256(path.read_bytes()).digest())
    config = directory / 'config.file'
    require(config.is_file() and config.stat().st_size == 8 and int.from_bytes(config.read_bytes(), 'little') == size,
            'NEO config.file does not account for the complete cache')
    return count, size, image_hash.hexdigest()


def counters(directory):
    data = (directory / 'stats').read_bytes()
    require(len(data) == 24, 'missing release-format NEO statistics')
    hits, misses, version = struct.unpack('<QQQ', data)
    require(version == 1, 'unrecognized NEO statistics version')
    return dict(hits=hits, misses=misses)


def private_copy(source, target, expected):
    require(not target.exists() and not target.is_symlink() and inventory(source) == expected,
            'fresh destination and exact complete seed required')
    shutil.copytree(source, target)
    require(inventory(target) == expected and inventory(source) == expected, 'seed changed during private copy')


def read_ready(plan, stopped):
    """Bracket the first flushed ready line, without sending another HTTP request."""
    root = Path(plan['root'])
    # bench.sh selects its private port on the CLI independently of the template.
    command = plan['command']
    require(command.count('--port') == 1, 'one official server port required')
    port = int(command[command.index('--port') + 1])
    marker = f'ready: http://127.0.0.1:{port}/v1  '.encode()
    def boot_ns():
        return time.clock_gettime_ns(time.CLOCK_BOOTTIME)
    deadline = time.monotonic() + 640
    lower = boot_ns()
    while time.monotonic() < deadline and not stopped.is_set():
        before = boot_ns()
        logs = list((root / 'tmp').glob('strata-bench-*.log'))
        require(len(logs) <= 1, 'ambiguous owned server log')
        if logs:
            with logs[0].open('rb') as f:
                data = f.read(65537)
            require(len(data) <= 65536, 'server startup log too large')
            if any(line.startswith(marker) for line in data.splitlines(keepends=True) if line.endswith(b'\n')):
                upper = boot_ns()
                claim = json.loads((root / 'engine-start.json').read_text())
                leader = int(logs[0].stem.split('-')[-1])
                def proc(pid):
                    values = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
                    return int(values[1]), int(values[3]), int(values[19])
                parent, sid, start = proc(claim['native']['pid'])
                require((parent, sid, start) == (claim['wrapper']['pid'], leader, claim['native']['start_ticks']),
                        'native owner changed during startup')
                server, sid, start = proc(parent)
                require((sid, start) == (leader, claim['wrapper']['start_ticks']), 'server wrapper identity differs')
                parent, sid, ticks = proc(server)
                argv = Path(f'/proc/{server}/cmdline').read_bytes().split(b'\0')
                require(parent == sid == leader and argv[1] == os.fsencode(bank.P / 'serve/server.py')
                        and argv.count(b'--port') == 1 and int(argv[argv.index(b'--port') + 1]) == port,
                        'ready message belongs to a different server')
                hz = os.sysconf('SC_CLK_TCK')
                lo = lower / 1e9 - (ticks+1) / hz
                hi = upper / 1e9 - ticks / hz
                require(0 < lo <= hi and hi-lo <= 0.5, 'ready publication clock unresolved')
                return dict(server_pid=server, server_start_ticks=ticks, owner=leader,
                            native=claim['native'], lower_s=lo, upper_s=hi)
        lower = before
        stopped.wait(.075)
    raise RuntimeError('server readiness was not observed')


def stats_gate(processes):
    require(len(processes) >= 4 and all(p['passed'] for p in processes[:4]), 'missing diagnostic processes')
    empty, short, full = [p['driver_stats'] for p in processes[1:4]]
    require(all(counts['end'][key] >= counts['begin'][key]
                for counts in (empty, short, full) for key in ('hits', 'misses')), 'driver counters decreased')
    def first(counts, key):
        return counts['end'][key] - counts['begin'][key]
    return (short['begin']['hits'] > empty['begin']['hits'] and short['begin']['misses'] < empty['begin']['misses']
            and first(full, 'hits') > first(short, 'hits') and first(full, 'misses') < first(short, 'misses'))


def decide(processes):
    require(len(processes) == len(SEEDS) and stats_gate(processes) and
            len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(SEEDS),
            'complete independent driver-cache campaign required')
    for i, (seed, process) in enumerate(zip(SEEDS, processes, strict=True)):
        require(process['index'] == i and process['seed'] == seed and process['read'] == READ
                and process['clean_speed'] == (i > DIAGNOSTIC[-1]) and len(process['samples']) == RUNS
                and process['complete'] and process['passed'] and process['ownership_closed']
                and not process['fresh_faults'] and process['native_exit_code'] == 0,
                'missing, misordered or ineligible cache comparison')
    results = {}
    for candidate, reference in (('short', 'empty'), ('full', 'empty'), ('full', 'short')):
        pairs = []
        for block in range(len(BLOCKS)):
            start = 4 + block*3
            by_seed = dict(zip(BLOCKS[block], processes[start:start+3], strict=True))
            a, b = by_seed[candidate], by_seed[reference]
            pairs.append(dict(ready_upper_ratio=a['readiness']['upper_s']/b['readiness']['lower_s'],
                warmup_ratio=a['short_warmup_s']/b['short_warmup_s'],
                first_prompt_ratio=a['samples'][0]['prompt_ms']/b['samples'][0]['prompt_ms'],
                first_ttft_ratio=a['samples'][0]['ttft_s']/b['samples'][0]['ttft_s'],
                warm_prompt_ratio=a['samples'][1]['prompt_ms']/b['samples'][1]['prompt_ms'],
                warm_ttft_ratio=a['samples'][1]['ttft_s']/b['samples'][1]['ttft_s']))
        results[candidate+'/'+reference] = dict(pairs=pairs, gains={
            k: sum(pair[k] < 1 for pair in pairs) >= 2 and statistics.median(pair[k] for pair in pairs) < 1
            for k in pairs[0]})
    return dict(contrasts=results, driver_hits_verified=True, adopted=False)


def experiment(executor, stopped):
    ctx = basis.core.source.device.load_module('neo_cache_prefill', basis.core.__file__)
    ordinary = curve.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS = ROOT, KEY, THIS
    ctx.ORDER, ctx.READS, ctx.RUNS, ctx.CLEAN_FAST = [(READ, 1)]*len(SEEDS), (READ,)*len(SEEDS), RUNS, True
    ctx.basis, ctx.check_sample, ctx.request_route = curve.admission, ordinary.check_sample, ordinary.request_route
    ctx.decision = decide
    original_save, original_current, original_pins = ctx.save, ctx.current, ctx.pins
    readiness = None

    def pins(revision):
        values = original_pins(revision)
        require({THIS, curve.THIS, speed.THIS, *speed.state.SCRIPTS} <= values.keys(), 'missing committed cache controller')
        return values

    def environment(row, plan):
        env = ordinary.environment(row, plan)
        require(env['SYCL_CACHE_PERSISTENT'] == '0' and not any(k.startswith('NEO_CACHE_') for k in env),
                'unexpected inherited driver or SYCL cache policy')
        return dict(env, NEO_CACHE_DIR=str(Path(plan['root'])/'neo'), NEO_CACHE_PERSISTENT='1',
                    NEO_CACHE_STATS=str(int(plan['index'] in DIAGNOSTIC)))

    def save(row):
        if row['stage'] == 'prepared':
            for p in row['plans']:
                p['arm'] = row['observer_admission']['curve_arm']
            row.update(whole_bytes=len(SEEDS)*row['budgets']['total_cap']+(24<<20), plan=PLAN)
        original_save(row)

    def current(row):
        require(row['plan'] == PLAN and len(row['plans']) == len(SEEDS) and all(
            p['arm'] == row['observer_admission']['curve_arm'] and p['read'] == READ and p['index'] == i
            and p['root'] == str(ROOT/str(i)) for i,p in enumerate(row['plans'])), 'cache comparison plan changed')
        if len(row['processes']) >= 4 and row['stage'] != 'failed':
            require(stats_gate(row['processes']), 'stats do not establish both seed effects')
        return original_current(row)

    def lifetime(out, stage, clock_ns):
        nonlocal readiness
        if stage != 'begin':
            return
        i = out['index']
        plan = json.loads((ROOT/'record.json').read_text())['plans'][i]
        case = Path(plan['root'])
        require(not (case/'engine-start.json').exists() and not any((case/'tmp').glob('strata-bench-*.log')),
                'readiness observation must start before server launch')
        d = case/'neo'
        out['clean_speed'] = i > DIAGNOSTIC[-1]
        start = time.monotonic()
        seed = SEEDS[i]
        if seed == 'empty':
            d.mkdir(mode=0o700)
            require(not any(d.iterdir()), 'fresh empty private driver directory required')
        else:
            source = ROOT/'short-seed' if seed == 'short' else ROOT/'0/neo'
            private_copy(source, d, SHORT if seed == 'short' else FULL)
        out['seed'] = seed
        out['prepare_directory_s'] = time.monotonic() - start
        readiness = executor.submit(read_ready, plan, stopped)

    def at_gate(out, request, edge, raw_ns):
        if request != 0:
            return
        i = out['index']
        directory = ROOT/str(i)/'neo'
        if edge == 'begin':
            out['readiness'] = readiness.result(timeout=1)
            require(out['readiness']['native'] == out['native_identity'], 'ready observer and benchmark saw different engines')
            if i == 0:
                private_copy(directory, ROOT/'short-seed', SHORT)
        if i in DIAGNOSTIC:
            out.setdefault('driver_stats', {})[edge] = counters(directory)

    def check_work(out, row, plan):
        work = [speed.dispatch.signature(r,s) for r,s in zip(out['routes'], out['samples'], strict=True)]
        require(work == row['observer_admission']['expected_work'][:RUNS] and len(work) == RUNS,
                'fresh fixed input, expert work, placement, output or draft work changed')
        i = out['index']
        directory = ROOT/str(i)/'neo'
        require(inventory(directory, stats=i in DIAGNOSTIC) == FULL and out['clean_speed'] == (i > DIAGNOSTIC[-1]),
                'final driver images or observation class differ')
        journal, = (Path(plan['root'])/'bench').glob('*/requests.jsonl')
        entries = [json.loads(line) for line in journal.read_text().splitlines()]
        warmup, = (e['sample'] for e in entries if e['kind'] == 'result' and e['phase'] == 'warm-up/chat')
        interval = warmup['client_interval']
        require(warmup['ok'] and warmup['stream_complete'] and interval['request_begin_monotonic_ns']
                < interval['first_output_monotonic_ns'] <= interval['request_end_monotonic_ns'], 'invalid short warm-up')
        out['short_warmup_s'] = (interval['request_end_monotonic_ns'] - interval['request_begin_monotonic_ns'])/1e9
        out['warmup_work'] = {k:warmup[k] for k in ('request_sha256','prompt_tokens','read_tokens','reused_tokens',
                                                      'generated_tokens','drafts_accepted','drafts_offered')}
        if i:
            require(out['warmup_work'] == row['processes'][0]['warmup_work'], 'short warm-up work differs')
        if i == DIAGNOSTIC[-1]:
            require(stats_gate([*row['processes'], out]), 'seeded caches did not shift startup and first-long hits/misses')

    ctx.pins, ctx.environment, ctx.save, ctx.current = pins, environment, save, current
    ctx.observe_lifetime, ctx.observe_gate, ctx.after_execution = lifetime, at_gate, check_work
    return ctx


if __name__ == '__main__':
    action, = sys.argv[1:]
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'host', *map(str,range(len(SEEDS)))),
            'use the pinned Python and a planned stage')
    stopped = threading.Event()
    with (bank.B/'owned-v0141-measurement.lock').open('a+') as lock, ThreadPoolExecutor(max_workers=1) as thread:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        ctx = experiment(thread, stopped)
        try:
            if action == 'prepare':
                ctx.prepare()
            elif action == 'host':
                basis.comparison.host(ctx)
            else:
                basis.comparison.run(int(action), ctx)
        finally:
            stopped.set()
