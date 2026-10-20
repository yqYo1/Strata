#!/usr/bin/env python3
"""Analyse one closed request; never load a runtime or execute native code."""
import collections
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys

P = Path(__file__).resolve().parents[2]
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
ROOT = B / 'unitrace-production-32k-v1'
RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/cpu-profiler-current.json'
COLLECTION = '487645cbf84fb93c58e4cee6e1eb5e56a3195626'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(P), *args], timeout=20)


def original():
    return json.loads(git('show', COLLECTION + ':' + str(RESULT.relative_to(P))))['unitrace_production_request']


def closed(record):
    require({k: v for k, v in record.items() if k not in ('analysis_plan', 'analysis')} == original(),
            'original collection outcome changed')
    require(record['passed'] and not record['active'] and record['ownership_closed']
            and record['model_observation']['session_removed'], 'closed successful collection required')


def prepare(container, record):
    closed(record)
    require('analysis_plan' not in record, 'inspect existing analysis; never blind retry')
    source = git('rev-parse', 'HEAD').decode().strip()
    relative = str(Path(__file__).resolve().relative_to(P))
    require(git('show', source + ':' + relative) == Path(__file__).read_bytes(), 'commit analyser first')
    files = [ROOT / 'record.json', ROOT / 'engine.stderr', *sorted((ROOT / 'trace').glob('*.json')),
             *sorted((ROOT / 'bench').glob('*/results.json')),
             *sorted((ROOT / 'bench').glob('*/requests.jsonl')),
             *sorted((ROOT / 'bench').glob('*/journal-state.json'))]
    require(len(files) == 6, 'one complete trace and official result/journal/state required')
    pinned = {str(path): dict(sha256=sha(path), bytes=path.stat().st_size) for path in files}
    timeline = next(path for path in files if path.parent.name == 'trace')
    require(pinned[str(timeline)]['sha256'] == record['capture']['timeline_sha256'], 'timeline differs')
    record['analysis_plan'] = dict(source_commit=source, source_path=relative, source_sha256=sha(__file__),
        interpreter=dict(path=str(Path(sys.executable).resolve()), sha256=sha(sys.executable), version=sys.version),
        collection_commit=COLLECTION, inputs=pinned,
        gates='Preserve collection outcome. Verify closed input hashes and journal chain/order/request binding, actual48-layer BF16 GPU-residual path and complete restoration, finite host/GPU intervals, native-PID/request filtering, per-record host-append links and direct-append inventory conservation. Retain edge/excluded counts. Categorize whole request and first-stream-output prefix separately; linear paired RAW/monotonic clock conversion. Source-supported collector loss limits remain an independent review gate.',
        limits='No native execution, timing comparison, hardware saturation, exact prefill-only boundary or full dependency/critical-path claim. Family names are source lookup groups. Requested copy bytes are not measured bus traffic. Host API wall time is not CPU time. Unions overlap across rows.',
        budgets=dict(wall_seconds=180, cpu_seconds=120, address_space_bytes=6 << 30, result_bytes=24 << 10),
        consumer='Select matched hardware/operation references and next prefill optimization; retire raw trace after remaining source/loss/observer questions close.')
    RESULT.write_text(json.dumps(container, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(stage='analysis-prepared', source_commit=source, input_files=len(files))))


def merged(intervals):
    result = []
    for begin, end in sorted(intervals):
        if result and begin <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((begin, end))
    return result


def length(intervals):
    return sum(end - begin for begin, end in merged(intervals))


def intersection(left, right):
    a, b = merged(left), merged(right)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def interval(event, begin, end):
    return max(begin, event['ts']), min(end, event['ts'] + event['dur'])


def family(name):
    if name.startswith('zeCommandListAppendMemoryCopy'):
        return 'copy_' + name.split('(', 1)[1].split(')', 1)[0]
    if name.startswith('zeCommandListAppendMemoryFill'):
        return 'fill'
    if name.startswith('zeCommandListAppendBarrier'):
        return 'barrier'
    for token, group in [('prefix_batch_impl', 'QSA_prefix_attention'), ('recurrence_columns', 'GDN_recurrence'),
                         ('DownScatterKernel', 'MoE_down_scatter'), ('GuSwiGLUKernel', 'MoE_GU_SwiGLU'),
                         ('vllm_xe2_native16::', 'native16_other_dense'), ('iq_dequant_gu_f16', 'MoE_GU_dequant'),
                         ('parallel_iq_launch', 'parallel_IQ_dequant'), ('dequant', 'other_dequant'),
                         ('prompt_product', 'quantized_prompt_product'), ('gemm_', 'library_GEMM'),
                         ('prefill::native16::', 'native16_vector_metadata'), ('qsa', 'QSA_other'),
                         ('gdn', 'GDN_other'), ('moe', 'MoE_other')]:
        if token in name:
            return group
    return 'unclassified_kernel'


def summary(gpu, host, begin, end):
    chosen = [e for e in gpu if e['ts'] < end and e['ts'] + e['dur'] > begin]
    cpus = [e for e in host if e['ts'] < end and e['ts'] + e['dur'] > begin]
    intervals = [interval(e, begin, end) for e in chosen]
    host_intervals = [interval(e, begin, end) for e in cpus]
    kernels = [interval(e, begin, end) for e in chosen if not e['name'].startswith('zeCommandList')]
    copies = [interval(e, begin, end) for e in chosen if family(e['name']).startswith('copy_')]
    rows = {}
    for event in chosen:
        key = family(event['name'])
        row = rows.setdefault(key, dict(count=0, service_sum_us=0.0, intervals=[], requested_bytes=0))
        lo, hi = interval(event, begin, end)
        row['count'] += 1
        row['service_sum_us'] += hi - lo
        row['intervals'].append((lo, hi))
        if key.startswith('copy_') or key == 'fill':
            size = re.search(r'\[(\d+)\]$', event['name'])
            require(size, 'copy/fill requested byte count missing')
            row['requested_bytes'] += int(size[1])
    for row in rows.values():
        row['service_union_us'] = length(row.pop('intervals'))
    gpu_us = length(intervals)
    return dict(wall_us=end - begin, gpu_events=len(chosen), host_events=len(cpus), gpu_union_us=gpu_us,
        kernel_union_us=length(kernels), copy_union_us=length(copies),
        kernel_copy_overlap_us=intersection(kernels, copies), host_api_union_us=length(host_intervals),
        host_api_during_gpu_empty_us=length(host_intervals) - intersection(host_intervals, intervals),
        gpu_empty_us=end - begin - gpu_us,
        families=dict(sorted(rows.items(), key=lambda item: item[1]['service_sum_us'], reverse=True)))


def journal_binding(record):
    path = next((ROOT / 'bench').glob('*/requests.jsonl'))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    previous = None
    for index, row in enumerate(rows):
        body = {k: v for k, v in row.items() if k != 'sha256'}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        require(row['index'] == index and row['previous_sha256'] == previous and row['sha256'] == digest, 'journal chain differs')
        previous = digest
    require(len(rows) == 5 and rows[0]['kind'] == 'identity', 'unexpected journal extent')
    for offset, phase in [(1, 'warm-up/chat'), (3, 'prefill-only')]:
        start, result = rows[offset:offset + 2]
        require(start['kind'] == 'start' and result['kind'] == 'result' and result['request_index'] == offset
            and all(r['phase'] == phase for r in (start, result)) and start['thinking'] is True
            and result['sample']['ok'] and start['request_sha256'] == result['sample']['request_sha256'], 'journal order/request differs')
    sample = dict(record['request']); sample.pop('run')
    require(rows[4]['sample'] == sample and rows[3]['run'] == 0, 'captured sample not bound to journal')
    state = json.loads((path.parent / 'journal-state.json').read_text())
    require(state['complete'] and state['last_sha256'] == previous and state['pending'] == [], 'journal state incomplete')
    return dict(request_sha256=sample['request_sha256'], last_sha256=previous, warmup_then_one_long_request=True)


def native_route(record, stderr):
    route = re.findall(r'layer-major candidate, tokens=(\d+) chunk=(\d+) residual_dtype=(\w+).*?residual_gpu_rows=(\d+)', stderr)
    require(len(route) == 1, 'ambiguous native long-prefill route')
    tokens, chunk, dtype, gpu_rows = route[0]
    require(int(tokens) == record['request']['read_tokens'] - 1 and dtype == 'bf16' and tokens == gpu_rows, 'actual residual route differs')
    sequence = re.findall(r'layer-major layer (\d+) of (\d+): (loading (\d+) experts|completed (\d+) chunks)', stderr)
    expected_chunks = (int(tokens) + int(chunk) - 1) // int(chunk)
    require(len(sequence) == 96 and all(int(row[0]) == index // 2 and row[1] == '48'
        and (row[3] == '512' if index % 2 == 0 else int(row[4]) == expected_chunks)
        for index, row in enumerate(sequence)), 'native layer/chunk sequence incomplete')
    candidate = stderr.index('layer-major candidate,')
    done = stderr.index('layer-major layer 47 of 48: completed', candidate)
    restored = stderr.index('phase=restored residency=Full activations=48', done)
    read = stderr.index('strata trace: read ' + tokens + ' tokens (batched)', restored)
    refill = stderr.index('strata trace: refilled ', read)
    require('strata prefill: native16 BF16 HC residual/projection IO' in stderr[candidate:done]
        and 'entry_done=1 run_returned=1 restored=1' in stderr[restored:read], 'native16 or restoration completion missing')
    warnings = [line.strip()[:500] for line in stderr.splitlines() if re.search(r'(?i)warning|overflow|dropped|lost|symbol lookup error', line)]
    return dict(batched_tokens=int(tokens), chunk=int(chunk), residual_dtype=dtype, residual_gpu_rows=int(gpu_rows),
        layers=48, chunks_per_layer=expected_chunks, ordered_route_restore_refill=True,
        diagnostic_lines=warnings[:8], source_loss_and_loader_review_pending=True)


def analyse(container, record):
    closed(record)
    require('analysis' not in record, 'inspect existing closed analysis')
    plan = record['analysis_plan']
    require(RESULT.read_bytes() == git('show', 'HEAD:' + str(RESULT.relative_to(P))), 'commit analysis plan first')
    require(sha(__file__) == plan['source_sha256'] and git('show', plan['source_commit'] + ':' + plan['source_path']) == Path(__file__).read_bytes(), 'analysis source changed')
    require(str(Path(sys.executable).resolve()) == plan['interpreter']['path'] and sha(sys.executable) == plan['interpreter']['sha256']
        and sys.version == plan['interpreter']['version'], 'analysis interpreter changed')
    for path, pin in plan['inputs'].items():
        require(Path(path).stat().st_size == pin['bytes'] and sha(path) == pin['sha256'], 'closed input changed: ' + path)
    result = dict(native_executed=False, analysis_passed=False, complete_production_attribution_qualified=False,
                  source_commit=plan['source_commit'], limits=plan['limits'])
    try:
        result['journal_binding'] = journal_binding(record)
        result['native_route'] = native_route(record, (ROOT / 'engine.stderr').read_text())
        timeline = next((ROOT / 'trace').glob('*.json'))
        events = json.loads(timeline.read_text())['traceEvents']
        gpu = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'gpu_op']
        all_host = [e for e in events if e.get('ph') == 'X' and e.get('cat') == 'cpu_op']
        native = record['native_identity']['pid']
        host = [e for e in all_host if e['pid'] == native]
        begin, end = record['request_raw_begin_ns'] / 1000, record['request_raw_end_ns'] / 1000
        require(len(gpu) == record['capture']['gpu_events'] and len(all_host) == record['capture']['host_events'], 'collection counts differ')
        require(all(isinstance(e.get(k), (int, float)) and math.isfinite(e[k]) for e in gpu + all_host for k in ('ts', 'dur'))
            and all(e['dur'] >= 0 for e in gpu + all_host), 'nonfinite/negative interval')
        flows = collections.defaultdict(list)
        host_starts = collections.defaultdict(list)
        for e in host:
            host_starts[e['tid'], e['ts']].append(e)
        for e in events:
            if e.get('pid') == native and e.get('ph') == 's' and e.get('cat', '').startswith('Flow_H2D_'):
                flows[str(e['id'])].append(e)
        links = collections.Counter(); unmatched = []
        for e in gpu:
            identifier = str(e['args']['id']); sources = flows[identifier]
            require(len(sources) == 1, 'missing/ambiguous host source')
            source = sources[0]; parents = host_starts[source['tid'], source['ts']]
            if len(parents) == 1:
                links[parents[0]['name']] += 1
            elif len(unmatched) < 5:
                unmatched.append(dict(id=identifier, parent_count=len(parents), gpu_name=e['name'][:160]))
        appends = collections.Counter(e['name'] for e in host if e['name'].startswith('zeCommandListAppend'))
        result['host_gpu_inventory'] = dict(host_append_counts=dict(appends), linked_gpu_counts=dict(links),
            unmatched_examples=unmatched, all_gpu_parent_apis_matched=sum(links.values()) == len(gpu),
            direct_append_counts_equal=appends == links)
        edges = [e for e in gpu if e['ts'] < begin or e['ts'] + e['dur'] > end]
        result['filtering'] = dict(excluded_other_host_pids=len(all_host) - len(host),
            excluded_native_host_outside=sum(e['ts'] + e['dur'] <= begin or e['ts'] >= end for e in host),
            gpu_outside_exact_window=len(edges),
            largest_gpu_edge_us=max([0.0] + [max(begin - e['ts'], e['ts'] + e['dur'] - end) for e in edges]),
            declared_allowance_us=1000)
        require(result['filtering']['largest_gpu_edge_us'] <= 1000, 'GPU exceeds declared edge allowance')
        left, right = record['clock_begin'], record['clock_end']
        mono = record['request']['client_interval']['first_output_monotonic_ns']
        ratio = (mono - left['monotonic_ns']) / (right['monotonic_ns'] - left['monotonic_ns'])
        require(0 <= ratio <= 1, 'first output outside paired clocks')
        raw_mid = lambda pair: (pair['raw_before_ns'] + pair['raw_after_ns']) / 2
        prefix_end = (raw_mid(left) + ratio * (raw_mid(right) - raw_mid(left))) / 1000
        require(begin < prefix_end < end, 'first output outside controlled request')
        result['first_output_conversion'] = dict(raw_us=prefix_end, method='linear paired-clock interpolation',
            maximum_pair_width_ns=max(p['raw_after_ns'] - p['raw_before_ns'] for p in (left, right)),
            limit='Clock interpolation error uncalibrated; first stream output is a prefill-dominated bound, not native prefill end.')
        result['whole_request'] = summary(gpu, host, begin, end)
        result['before_first_output'] = summary(gpu, host, begin, prefix_end)
        result['after_first_output'] = summary(gpu, host, prefix_end, end)
        unclassified = collections.Counter()
        for e in gpu:
            if family(e['name']) == 'unclassified_kernel': unclassified[e['name']] += e['dur']
        result['largest_unclassified'] = [dict(name=name, service_sum_us=duration)
            for name, duration in unclassified.most_common(8)]
        result['analysis_passed'] = True
    except Exception as error:
        result['error'] = repr(error)
    require(len(json.dumps(result).encode()) <= plan['budgets']['result_bytes'], 'compact analysis result budget exceeded')
    record['analysis'] = result
    RESULT.write_text(json.dumps(container, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(analysis_passed=result['analysis_passed'], native_executed=False, error=result.get('error'))))
    return 0 if result['analysis_passed'] else 1


def main():
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'analyse'), 'use profile_analysis.py prepare|analyse')
    resource.setrlimit(resource.RLIMIT_AS, (6 << 30, 6 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    signal.alarm(180)
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        container = json.loads(RESULT.read_text()); record = container['unitrace_production_request']
        if sys.argv[1] == 'prepare':
            prepare(container, record)
            return 0
        return analyse(container, record)


if __name__ == '__main__':
    raise SystemExit(main())
