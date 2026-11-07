"""CPU leaf accounting without a static residency census; repository LICENSE.

Main-authored from this fork's version2 CPU_SERVICE and expert-copy footers.
This checks completed request/job/entry and thread/bin conservation, not dynamic
map identity, physical memory traffic, hardware saturation or observer cost.
The owner must separately qualify source, routes, clocks, exit and fault state.
"""
import re

import cpu_service_evidence as service
import cpu_work_census as census
import decode_footer as footer
from measurement_guards import require

MAX = (1 << 64) - 1
FIELDS = ('prompt_tokens', 'reused_tokens', 'read_tokens', 'generated_tokens',
          'drafts_accepted', 'drafts_offered')


def add(total, value):
    for key in total:
        require(value[key] <= MAX - total[key], 'CPU leaf aggregate overflow')
    for key in total:
        total[key] += value[key]


def extract(text, samples, width, workers=5):
    """Bind every leaf record to its completed native request and CPU footer.

    HQ has exactly one call per final expert job. GU/DOWN rows must conserve
    that job's fixed model geometry. The separate native footer conserves all
    jobs and token entries; it does not independently identify each format bin.
    """
    require(type(text) is str and 0 < len(text) <= 64 << 20 and
            type(samples) is list and 0 < len(samples) <= 64 and
            all(type(sample) is dict for sample in samples) and
            type(width) is int and width in (2, 4) and
            type(workers) is int and 1 <= workers <= 64, 'bounded leaf input required')
    require('strata cpu work:' not in text, 'leaf-only scope cannot claim a static census')
    lines = text.splitlines()
    headers = [(i, match) for i, line in enumerate(lines)
               if (match := re.fullmatch(footer.HEADER, line))]
    copies = [(i, match) for i, line in enumerate(lines)
              if (match := re.fullmatch(footer.COPY, line))]
    require(len(headers) == len(copies) == len(samples) and
            sum(line.startswith('strata serve: prompt ') for line in lines) == len(headers) and
            sum(line.startswith('strata serve: SYCL_EXPERT_COPY') for line in lines) == len(copies),
            'complete native request/copy framing required')
    records, offset, previous_copy, owners, result = service.records(text), 0, -1, {}, []
    for request, ((head_line, header), (copy_line, copy), sample) in enumerate(
            zip(headers, copies, samples), 1):
        require(all(type(sample.get(key)) is int and 0 <= sample[key] <= MAX for key in FIELDS) and
                sample['prompt_tokens'] == sample['reused_tokens'] + sample['read_tokens'] and
                sample['read_tokens'] > 0 and sample['generated_tokens'] > 0 and
                sample['drafts_accepted'] <= sample['drafts_offered'] and
                tuple(map(int, header.groups())) == tuple(sample[key] for key in FIELDS),
                'native/client request mismatch')
        require(previous_copy < head_line < copy_line and
                (request == len(headers) or copy_line < headers[request][0]),
                'native copy footer escaped its request')
        counters = dict(zip(footer.COPY_KEYS, (census.integer(v) for v in copy.groups())))
        require(not any(counters[k] for k in footer.COPY_KEYS[:6]),
                'leaf-only plan requires the qualified pcie0 route')
        require(offset < len(records) and records[offset][1:] ==
                ('CPU_SERVICE', dict(service.HEADER, request=str(request))) and
                previous_copy < records[offset][0] < head_line, 'missing/bad leaf request header')
        offset += 1
        bins, previous, total = {}, (-1, 0, -1), dict(calls=0, wall_ns=0, cpu_ns=0)
        while offset < len(records) and records[offset][1] == 'CPU_SERVICE_BIN':
            line, _, raw = records[offset]
            offset += 1
            value = {key: census.integer(v) for key, v in raw.items() if key != 'op'}
            pair, nt, op = (value['gu'], value['down']), value['nt'], raw['op']
            require(pair in service.TYPES and 1 <= nt <= width and op in service.OPS and
                    value['request'] == request and line < head_line, 'bad leaf bin identity/order')
            key = service.TYPES.index(pair), nt, service.OPS.index(op)
            require(previous < key and value['calls'] > 0 and
                    0 < value['cpu_ns'] <= value['wall_ns'] and
                    0 < value['rows'] <= MAX // nt and value['token_rows'] == value['rows'] * nt,
                    'duplicate/reordered bin or invalid work/clock')
            previous = key
            bins[pair + (nt, op)] = value
            add(total, value)
        groups, jobs, entries = [], 0, 0
        for gu, down, nt in sorted({key[:3] for key in bins}):
            require(all((gu, down, nt, op) in bins for op in service.OPS), 'missing leaf operation')
            counts = {op: bins[gu, down, nt, op] for op in service.OPS}
            n = counts['HQ']['calls']
            require(n <= MAX // 2560, 'CPU job geometry overflow')
            for op, rows in (('GU', 640), ('HQ', 640), ('DOWN', 2560)):
                value = counts[op]
                require(value['rows'] == n * rows and n <= value['calls'] <= n * rows,
                        'GU/HQ/DOWN job rows disagree')
            jobs += n
            entries += n * nt
            require(jobs <= MAX and entries <= MAX, 'CPU job/entry sum overflow')
            groups.append(dict(gu=gu, down=down, nt=nt, jobs=n,
                               operations={op: {k: counts[op][k] for k in total} for op in service.OPS}))
        require((jobs, entries) == (counters['cpu_unique_misses'], counters['cpu_entries']),
                'leaf work disagrees with independent native CPU footer')
        thread_total, tids, previous_slot = dict.fromkeys(total, 0), set(), -1
        while offset < len(records) and records[offset][1] == 'CPU_SERVICE_THREAD':
            line, _, raw = records[offset]
            offset += 1
            value = {key: census.integer(v) for key, v in raw.items()}
            slot, tid = value['slot'], value['tid']
            require(value['request'] == request and line < head_line and
                    previous_slot < slot and (slot < workers or slot == 64) and
                    tid > 0 and tid not in tids and value['calls'] > 0 and
                    0 < value['cpu_ns'] <= value['wall_ns'], 'bad leaf thread identity/clock')
            require((slot not in owners or owners[slot] == tid) and
                    all(other == slot or owner != tid for other, owner in owners.items()),
                    'CPU pool owner changed')
            owners[slot] = tid
            previous_slot = slot
            tids.add(tid)
            add(thread_total, value)
        require(thread_total == total and (not jobs or 64 in owners and previous_slot == 64),
                'leaf thread/bin accounting mismatch')
        require(offset < len(records) and records[offset][1] == 'CPU_SERVICE_END' and
                records[offset][0] < head_line, 'missing leaf completion')
        complete = {key: census.integer(v) for key, v in records[offset][2].items()}
        offset += 1
        require(complete == dict(request=request, complete=1, bins=len(bins), threads=len(tids)),
                'incomplete leaf record')
        result.append(dict(request=request, cpu_jobs=jobs, cpu_entries=entries,
                           format_nt=groups, thread_count=len(tids), **total))
        previous_copy = copy_line
    require(offset == len(records), 'unbound trailing leaf records')
    return result
