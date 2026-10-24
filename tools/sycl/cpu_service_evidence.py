"""Conserve CPU leaf service against the same text's independent job census.

Repository LICENSE; independently authored from the two native output contracts.
Caller must separately qualify source, executable, request clocks and ownership.
"""
import cpu_work_census as census
from measurement_guards import require

TYPES = ((18, 20), (18, 42), (21, 20), (21, 42), (22, 20), (22, 42), (23, 20))
OPS = ('GU', 'HQ', 'DOWN')
HEADER = dict(valid='1', error_mask='0', formats='7', nt_max='4', operations='GU,HQ,DOWN',
    wall_clock='CLOCK_MONOTONIC', cpu_clock='CLOCK_THREAD_CPUTIME_ID', scope='decode_pool_leaves',
    overhead='charged', physical_RAM_bytes='unknown', critical_path='unknown')
BIN = {'request', 'gu', 'down', 'nt', 'op', 'calls', 'rows', 'token_rows', 'wall_ns', 'cpu_ns'}
THREAD = {'request', 'slot', 'tid', 'calls', 'wall_ns', 'cpu_ns'}


def records(text):
    found = []
    for line_index, line in enumerate(text.splitlines()):
        if not line.startswith('strata CPU_SERVICE'):
            continue
        words = line.split()
        require(len(words) >= 3 and words[0] == 'strata' and
            all(word.count('=') == 1 for word in words[2:]), 'malformed CPU service row')
        kind = words[1]
        values = dict(word.split('=', 1) for word in words[2:])
        expected = {'CPU_SERVICE': {*HEADER, 'request'}, 'CPU_SERVICE_BIN': BIN,
            'CPU_SERVICE_THREAD': THREAD}.get(kind)
        require(expected is not None and set(values) == expected and len(values) == len(words) - 2,
            'unknown/duplicate/missing CPU service fields')
        found.append((line_index, kind, values))
    return found


def extract(text, bindings, formats, membership, workers=5):
    """Read both observers afresh; never join a historical census to new timing.

    Independent census admission checks source-format geometry, static residency,
    native request/footer counts and pool clocks. This reader adds leaf work and
    per-thread timing conservation, not physical RAM traffic or CPU saturation.
    """
    require(len(text) <= 64 << 20 and 0 < len(bindings) <= 64 and
        type(workers) is int and 1 <= workers <= 64, 'bounded CPU service input required')
    jobs = census.extract(text, bindings, formats, membership)
    rows = records(text)
    offset = 0
    output = []
    lines = text.splitlines()
    owners = {}
    for work in jobs:
        request = work['request']
        require(offset < len(rows) and rows[offset][1] == 'CPU_SERVICE', 'missing CPU service header')
        first, _, header = rows[offset]
        offset += 1
        require(header == dict(HEADER, request=str(request)), 'CPU service request invalid or unbound')
        expected = {}
        for bin_ in work['format_bins']:
            pair = bin_['gu'], bin_['down']
            require(pair in TYPES and bin_['H'] == 2560 and bin_['FF'] == 640 and
                not any(bin_['jobs_NT'][4:]), 'unsupported CPU service census format')
            for nt, count in enumerate(bin_['jobs_NT'][:4], 1):
                if count:
                    expected[pair + (nt,)] = expected.get(pair + (nt,), 0) + count
        bins = {}
        previous_key = (-1, 0, -1)
        sums = dict(calls=0, wall_ns=0, cpu_ns=0)
        while offset < len(rows) and rows[offset][1] == 'CPU_SERVICE_BIN':
            _, _, raw = rows[offset]
            offset += 1
            require(raw['op'] in OPS, 'unknown CPU leaf operation')
            value = {k: census.integer(v) for k, v in raw.items() if k != 'op'}
            pair = value['gu'], value['down']
            nt, op = value['nt'], raw['op']
            require(value['request'] == request and pair in TYPES and 1 <= nt <= 4,
                'CPU leaf request/format/width mismatch')
            key = TYPES.index(pair), nt, OPS.index(op)
            require(key > previous_key, 'duplicate or reordered CPU leaf bins')
            previous_key = key
            count = expected.get(pair + (nt,), 0)
            wanted_rows = count * (2560 if op == 'DOWN' else 640)
            require(count > 0 and value['rows'] == wanted_rows and value['token_rows'] == wanted_rows * nt and
                count <= value['calls'] <= wanted_rows and
                (op != 'HQ' or value['calls'] == count), 'CPU leaf/job row conservation differs')
            require(0 < value['cpu_ns'] <= value['wall_ns'], 'invalid CPU leaf clocks')
            for k in sums:
                sums[k] += value[k]
                require(sums[k] <= (1 << 64) - 1, 'CPU leaf sum overflow')
            bins[pair + (nt, op)] = dict(value, op=op, jobs=count)
        require(set(bins) == {(*key, op) for key in expected for op in OPS}, 'missing CPU leaf bins')
        threads = {}
        thread_sums = dict(calls=0, wall_ns=0, cpu_ns=0)
        while offset < len(rows) and rows[offset][1] == 'CPU_SERVICE_THREAD':
            _, _, raw = rows[offset]
            offset += 1
            value = {k: census.integer(v) for k, v in raw.items()}
            slot, tid = value['slot'], value['tid']
            require(value['request'] == request and (slot < workers or slot == 64) and
                slot not in threads and tid > 0 and tid not in threads.values() and value['calls'] > 0 and
                0 < value['cpu_ns'] <= value['wall_ns'], 'CPU service thread identity/clocks differ')
            require(slot not in owners or owners[slot] == tid, 'CPU service pool slot owner changed')
            require(all(other == slot or owner != tid for other, owner in owners.items()),
                'CPU service TID moved between pool slots')
            owners[slot] = tid
            threads[slot] = tid
            for k in thread_sums:
                thread_sums[k] += value[k]
        require(thread_sums == sums and (not expected or 64 in threads), 'CPU service thread/bin totals differ')
        begin = f'strata cpu work: begin request={request} '
        end = f'strata cpu work: end request={request} '
        begins = [i for i, line in enumerate(lines) if line.startswith(begin)]
        ends = [i for i, line in enumerate(lines) if line.startswith(end)]
        require(len(begins) == len(ends) == 1 and begins[0] < first <= rows[offset - 1][0] < ends[0],
            'CPU leaf observer lies outside its independent census request')
        output.append(dict(request=request, cpu_jobs=work['cpu_jobs'], cpu_entries=work['cpu_entries'],
            same_request_census=True, thread_count=len(threads), **sums,
            format_bins=[bins[k] for k in sorted(bins, key=lambda k: (TYPES.index(k[:2]), k[2], OPS.index(k[3])))],
            limits='Concurrent leaf sums are not an additive critical path. CPU time includes memory stalls. '
                'Clock/counter costs remain charged; physical RAM traffic and model overhead are unqualified.'))
    require(offset == len(rows), 'unbound trailing CPU service rows')
    return output
