"""Offline strict pinned-PTI Linux MetricQuery text decoder. Repository LICENSE.

Original code; API/schema design references are recorded in source-provenance.json.
No runtime import, subprocess, environment modification or device access.
"""
from collections import Counter
import math
from pathlib import Path
import re
import time

SHAPE = re.compile(r'\[SIMD(?:[1-9][0-9]*|_ANY) \{([0-9]+); ([0-9]+); ([0-9]+)\} \{([0-9]+); ([0-9]+); ([0-9]+)\}\]$')

def need(value, message):
    if not value:
        raise ValueError(message)


def finite_text(path, deadline, cap=62 << 20):
    p = Path(path)
    need(time.monotonic() < deadline and p.is_file() and not p.is_symlink(), 'decoder deadline/regular file')
    need(0 < p.stat().st_size <= cap, 'finite nonempty decoder input')
    raw = p.read_text(encoding='utf-8', errors='strict')
    need(raw.endswith('\n') and '\x00' not in raw and time.monotonic() < deadline, 'complete finite UTF-8 input')
    return raw


def decode(path, catalog, expected_calls, deadline):
    """Catalog is complete, ordered driver metadata, never inferred from CSV names.

    Official PTI wraps names in quotes without escaping embedded quotes, so
    split from the right by the known column count. Reject extra device
    sections, partial files, nonfinite values and incomplete kernel coverage.
    Only a single B570 sample (SubDeviceId=0) is accepted: pinned PTI overwrites
    earlier sample strings when a query yields multiple samples.
    """
    need(len(catalog) == 83 and [m['metric'] for m in catalog] == list(range(83)), 'complete ComputeBasic EVENT_BASED schema')
    labels = []
    for m in catalog:
        units = m['units']
        if 'null' in units:
            units = ''
        elif 'percent' in units:
            units = '%'
        label = m['name'] + ('[' + units + ']' if units else '')
        need(label and ',' not in label and '\n' not in label, 'unambiguous official metric label')
        labels.append(label)
        need(m['result_type_name'] in {'ZET_VALUE_TYPE_UINT32', 'ZET_VALUE_TYPE_UINT64',
             'ZET_VALUE_TYPE_FLOAT32', 'ZET_VALUE_TYPE_FLOAT64', 'ZET_VALUE_TYPE_BOOL8'}, 'unsupported PTI typed-value schema')
    need(len(set(labels)) == 83, 'unique ordered metric labels')
    header = 'Kernel,GlobalInstanceId,SubDeviceId,' + ','.join(labels)
    state, rows, ids, counts = 0, [], set(), Counter()
    for number, line in enumerate(finite_text(path, deadline).splitlines(), 1):
        need(time.monotonic() < deadline and number <= 10000 and len(line) <= 65536, 'finite metric decode budget')
        if not line:
            continue
        if state == 0:
            need(line == '=== Device #0 Metrics ===', 'one exact target-device query section')
            state = 1
            continue
        if state == 1:
            need(line == header, 'unknown/incomplete/reordered metric schema')
            state = 2
            continue
        need(not line.startswith(('===', 'Kernel,')), 'duplicate/device/unknown query section')
        f = line.rsplit(',', 85)
        need(len(f) == 86 and f[0].startswith('"') and f[0].endswith('"'), 'quoted kernel and exact 83-value row')
        name, instance, subdevice = f[0][1:-1], f[1], f[2]
        need(SHAPE.search(name), 'full detailed compute kernel shape')
        normalized = SHAPE.sub('', name)
        need(normalized in expected_calls, 'known detailed compute kernel name')
        need(re.fullmatch(r'[1-9][0-9]*', instance) and 0 < int(instance) < 2**64 and
             subdevice == '0' and instance not in ids, 'unique finite instance/single sample')
        ids.add(instance)
        values = []
        for m, text in zip(catalog, f[3:]):
            kind = m['result_type_name']
            if kind in ('ZET_VALUE_TYPE_UINT32', 'ZET_VALUE_TYPE_UINT64', 'ZET_VALUE_TYPE_BOOL8'):
                need(re.fullmatch(r'[0-9]+', text), 'integer metric encoding')
                value = int(text)
                bound = {'ZET_VALUE_TYPE_UINT32': 2**32, 'ZET_VALUE_TYPE_UINT64': 2**64,
                         'ZET_VALUE_TYPE_BOOL8': 2}[kind]
                need(value < bound, 'typed integer metric range')
            else:
                need(re.fullmatch(r'-?[0-9]+\.[0-9]{6}', text), 'official std::to_string floating encoding')
                value = float(text)
                need(math.isfinite(value), 'finite metric value')
            values.append(value)
        need(values[0] > 0 and labels[0] == 'GpuTime[ns]', 'positive decoded measured query duration')
        counts[normalized] += 1
        need(len(rows) < 4096, 'finite query row count')
        rows.append(dict(kernel=name, summary_kernel=normalized, global_instance_id=int(instance), subdevice_id=0, values=values))
    need(state == 2 and rows and dict(counts) == dict(expected_calls), 'complete two-way compute dispatch/query counts')
    return dict(decoded=True, group='ComputeBasic', sampling='EVENT_BASED', device=0,
                metric_columns=labels, rows=rows, kernel_calls=dict(counts),
                collection_qualified=False, numerical_precision='PTI decimal text; float values rounded to six decimal places',
                limitation='No exact internal API-status evidence; no multi-sample support, model/performance/occupancy claim')
