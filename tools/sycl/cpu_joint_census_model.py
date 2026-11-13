#!/usr/bin/env python3
"""Observe physically valid target CPU jobs on the unchanged row path.

Main-authored from this fork's coherent model admission and census contracts.
Repository LICENSE. Charged diagnostics are not clean throughput evidence.
"""
from collections import Counter
import fcntl
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import check_cpu_joint_census as device
import coherent_prefill_model as previous
import cpu_work_census as census

P, B, RESULT = device.P, device.B, device.RESULT
ROOT, KEY = B / 'cpu-joint-census-model-v1', 'cpu_joint_census_model_v1'
THIS = 'tools/sycl/cpu_joint_census_model.py'
require, sha = device.require, device.sha
original_route = previous.route
JOINT = re.compile(r'^strata cpu schedule: layer=(\d+) gu=(\d+) down=(\d+) jobs=(\d+) '
    r'nt1=(\d+) nt2=(\d+) nt3=(\d+) nt4=(\d+) other=(\d+) whole_calls=(\d+)$')


def observe(text, formats, membership, prompts):
    """Conserve callback shapes against trace widths and the existing census.

    Existing census admission also checks restored membership and old pool clocks.
    No counters here measure physical memory traffic or arithmetic saturation.
    """
    require(len(text) <= 64 << 20 and len(formats) == len(membership) == 48,
        'bounded complete model census required')
    startup = re.findall(r'^CPU_DECODE_SCHEDULE whole_mixed=(\d+) down=42 jobs=24,36 '
        r'nt=mixed_1_to_4 workers=5 host=1$', text, re.M)
    require(startup == ['0'], 'baseline selector must actually initialize exactly once')
    require(not re.search(r'^CPU_COMPACT_GU_(BEGIN|PREPARED|READY)\b', text, re.M),
        'baseline must not prepare optional CPUcache')
    current, frames, all_cases = None, [], Counter()
    for line in text.splitlines():
        if line.startswith(census.PREFIX):
            kind, values = census.rows(line + '\n')[0]
            if kind == 'begin':
                require(current is None and len(frames) < len(prompts)
                    and values['phase'] == 'decode'
                    and census.integer(values['request']) == len(frames) + 1
                    and census.integer(values['prompt_tokens']) == prompts[len(frames)],
                    'joint request framing differs')
                current = dict(prompt_tokens=prompts[len(frames)], windows=[], calls=0,
                    cpu_jobs=0, cpu_entries=0, cases=Counter())
            elif kind == 'end':
                require(current is not None and current['windows']
                    and current['calls'] == 48 * len(current['windows'])
                    and census.integer(values['callbacks']) == current['calls'],
                    'missing/partial callback windows')
                frames.append(current); all_cases.update(current['cases']); current = None
            continue
        trace = re.fullmatch(r'strata trace: window (\d+) (\d+)', line)
        if trace and current is not None:
            position, width = map(int, trace.groups())
            require(1 <= width <= 4 and current['calls'] == 48 * len(current['windows']),
                'trace window width/order differs')
            if current['windows']:
                before, extent = current['windows'][-1]
                require(1 <= position - before <= extent, 'target positions do not advance')
            current['windows'].append((position, width))
        if not line.startswith('strata cpu schedule:'):
            continue
        match = JOINT.fullmatch(line)
        require(current is not None and match and current['windows'],
            'unbound/malformed CPUjoint callback')
        layer, gu, down, jobs, *tail = map(census.integer, match.groups())
        nt, other, whole = tail[:4], tail[4], tail[5]
        width = current['windows'][-1][1]
        require(layer == current['calls'] % 48
            and current['calls'] // 48 == len(current['windows']) - 1
            and (gu, down) == formats[layer][:2]
            and sum(nt) == jobs and jobs <= 10 * width
            and sum((i + 1) * n for i, n in enumerate(nt)) <= 10 * width
            and not any(nt[width:]) and other == whole == 0,
            'CPUjoint type/NT/physical volume/actual baseline route differs')
        current['calls'] += 1; current['cpu_jobs'] += jobs
        current['cpu_entries'] += sum((i + 1) * n for i, n in enumerate(nt))
        current['cases'][(gu, down, jobs, *nt)] += 1
    require(current is None and len(frames) == len(prompts), 'incomplete CPUjoint requests')
    timing = re.findall(r'^strata decode timing: (\d+) windows,[^\n]* CPU ([0-9.]+)\]', text, re.M)
    require(len(timing) == len(frames), 'old independent pool clocks missing')
    bindings = []
    for frame, (windows, cpu_ms) in zip(frames, timing):
        require(int(windows) == len(frame['windows']), 'timing/trace window counts differ')
        bindings.append(dict(prompt_tokens=frame['prompt_tokens'], windows=int(windows),
            rows=sum(t for _, t in frame['windows']), cpu_jobs=frame['cpu_jobs'],
            cpu_entries=frame['cpu_entries'], pool_ms=float(cpu_ms) * int(windows)))
    work = census.extract(text, bindings, formats, membership)
    kept, coverage = [], []
    for pair in sorted({k[:2] for k in all_cases}):
        cells = sorted((k for k in all_cases if k[:2] == pair), key=lambda k: (-all_cases[k], k))
        selected = set(cells[:12]) | {min(cells, key=lambda k: k[2]), max(cells, key=lambda k: k[2])}
        kept.extend([*k, all_cases[k]] for k in sorted(selected))
        coverage.append([*pair, sum(all_cases[k] for k in cells),
            sum(all_cases[k] for k in selected), len(cells)])
    return dict(requests=[dict(request=w['request'], prompt_tokens=b['prompt_tokens'],
            windows=w['windows'], rows=w['rows'], cpu_jobs=w['cpu_jobs'],
            cpu_entries=w['cpu_entries'], vram_entries=w['vram_entries'],
            jobs_NT=w['jobs_NT'], pool_ms=w['pool_ms'],
            membership_sha256=w['membership_sha256'], logical_job_blob_bytes=w['logical_job_blob_bytes'])
            for w, b in zip(work, bindings)],
        case_columns=['gu', 'down', 'jobs', 'nt1', 'nt2', 'nt3', 'nt4', 'calls'],
        representative_cases=kept,
        coverage_columns=['gu', 'down', 'all_callbacks', 'retained_callbacks', 'unique_cases'],
        coverage=coverage, actual_whole_calls=0,
        conservation='Each callback obeys the actual traced width*topk10 and layer format;'
            'independent native census checks NT/jobs/entries,cache membership and old pool clocks.',
        limits='Target-verifier decode only,after refill;separate draft excluded. Retain at most12most '
            'frequent plus minimum/maximum job cases per format for the next CPUworkload plan. '
            'No full temporal trace,clean speed,physical RAM/PCIe traffic or saturation claim.')


def controls():
    bits = '0000000003ffffff' + '0' * 112
    header = 'CPU_DECODE_SCHEDULE whole_mixed=0 down=42 jobs=24,36 nt=mixed_1_to_4 workers=5 host=1\n'
    begin = 'strata cpu work: begin request=1 phase=decode prompt_tokens=32769 layers=48 experts=512 cache=restored_decode_snapshot_immutable\n'
    cache = ''.join(f'strata cpu work: cache request=1 layer={l} bin=0 resident_bits={bits}\n' for l in range(48))
    joint = ''.join(f'strata cpu schedule: layer={l} gu=18 down=42 jobs={24 if l == 0 else 0} '
        f'nt1={20 if l == 0 else 0} nt2={4 if l == 0 else 0} nt3=0 nt4=0 other=0 whole_calls=0\n' for l in range(48))
    bin_ = ('strata cpu work: bin request=1 bin=0 gu=18 down=42 H=2560 FF=640 blob_bytes=1715200 '
        'gu_act=0 down_act=0 gu_row=980 down_row=180 act_bytes=0 h_bytes=0 calls=48 pool_ns=1000000 '
        'CPU=28 VRAM=1892 PCIe=0 peer=0 jobs_NT=20,4,0,0,0,0,0,0\n')
    end = 'strata cpu work: end request=1 phase=decode valid=1 windows=1 rows=4 callbacks=48 k=10 cpu_jobs=24 CPU=28 VRAM=1892 PCIe=0 peer=0 pcie_jobs=0 error=none\n'
    text = header + begin + cache + 'strata trace: window 32768 4\n' + joint + bin_ + end + 'strata decode timing: 1 windows, host [jobs 0.00 CPU 1.00]\n'
    args = ([(18, 42, 1715200)] * 48, [(1 << 26) - 1] * 48, [32769])
    good = observe(text, *args); require(good['requests'][0]['cpu_entries'] == 28, 'literal positive control')
    wrong = [text.replace('jobs=24 nt1=20 nt2=4', 'jobs=36 nt1=24 nt2=12'),
        text.replace('whole_calls=0', 'whole_calls=1', 1),
        text.replace('strata cpu schedule: layer=0 ', 'strata cpu schedule: layer=1 ', 1),
        text.replace(' callbacks=48 ', ' callbacks=47 '), text.replace(' k=10 ', ' k=9 ')]
    for candidate in wrong:
        try:
            observe(candidate, *args)
        except RuntimeError:
            continue
        raise RuntimeError('CPUjoint negative control escaped')
    return dict(literal_positive=1, rejected_controls=len(wrong), actual_model_or_GPU=False)


def route(text):
    value = original_route(text)
    row = json.loads((ROOT / 'record.json').read_text()); argv = row['argv']
    require('--expert-cache-per-layer' in argv and argv[argv.index('--expert-cache') + 1] == '1256',
        'preserve per-layer static1256cache')
    membership = census.expected_membership(argv[argv.index('--expert-profile') + 1], 1256)
    formats = census.manifest_formats(device.candidate.cpu.inputs.METADATA)
    value['CPU_joint_shape_census'] = observe(text, formats, membership, [32769, 82, 32769])
    return value


def memory_admission():
    available = next(int(s.split()[1]) * 1024 for s in Path('/proc/meminfo').read_text().splitlines()
        if s.startswith('MemAvailable:'))
    require(available >= 60 << 30, 'baseline model requires60GiB currently available')


device.context = device.inherited.context
previous.device = device
previous.ROOT, previous.KEY, previous.THIS = ROOT, KEY, THIS
previous.route = route


def prepare():
    memory_admission(); checked = controls(); previous.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['environment'].update(STRATA_CPU_COMPACT_GU_MIB='0', STRATA_CPU_DECODE_WHOLE='0',
        STRATA_CPU_WORK_CENSUS='1', STRATA_CPU_SERVICE='0', STRATA_DECODE_TIMING='1', STRATA_TRACE='1')
    row['budgets']['rss_cap'] = 80 << 30
    row.pop('previous_preparation', None); row.pop('previous_control_failure', None)
    row['joint_parser_controls'] = checked
    row['plan'] = ('Baseline selector0/cache0joint-shape observation,not an optimized-model trial. '
        'Preserve32769/82/32769thinking inputs,32outputs each,spec4,5workers+host,static1256slots, '
        '8192prefill,logical262144/resident20480INT8KV and qualified runtime. Require all11fresh '
        'native gates,actual handoff/restore/recapture,quiet validation,15full assets before/after, '
        'normal0/closed ownership/no fresh fault. Conserve every CPUcallback against the actual '
        'traced width,format,NT,census,resident membership and old pool clocks. Stop first failure;'
        'no retry,hash refresh or optimized policy. Use observed legal shapes for a new CPUplan.')
    row['limits'] = ('Charged functional/shape diagnostic;32generated tokens per request do not '
        'qualify sustained TG/task quality,clean PP/TTFT,physical262144/tails/RESTORE or adoption. '
        'A separate thinking-enabled1K+strong/weak workload census is needed for broader coverage.')
    row['retention'] = ('Keep compact counts,representative cases/coverage,state/fault and source '
        'proof for the next CPUcandidate. Retire successful raw logs/cache after closure. Keep14508 '
        'validated reference and this temporary f59observer;legacy holds remain separate.')
    previous.context().context(row['native_observer_admission']).save(row)


if __name__ == '__main__':
    require(sys.executable == previous.PYTHON and len(sys.argv) == 2
        and sys.argv[1] in (*previous.ACTIONS, 'controls'), 'qualified Python;prepare|host|run|controls')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous.original.guards.no_model_owners([])
        if sys.argv[1] == 'controls':
            print(json.dumps(controls()))
        elif sys.argv[1] == 'prepare':
            prepare()
        else:
            if sys.argv[1] == 'run':
                memory_admission()
            previous.context().execute(sys.argv[1])
