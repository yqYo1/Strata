"""Read the pinned Intel PTI device table, retaining names with template commas."""
import math
import re


def device_summary(text):
    sections = text.split('=== Device Timing Summary ===')
    if len(sections) != 2:
        raise ValueError('one device timing summary required')
    body = sections[1].split('=== Kernel Properties ===')
    if len(body) != 2:
        raise ValueError('complete kernel table footer required')
    totals = re.findall(r'Total Device Time for L0 backend \(ns\):\s*(\d+)', body[0])
    if len(totals) != 1 or int(totals[0]) <= 0:
        raise ValueError('one positive L0 total required')
    rows = []
    for line in body[0].splitlines():
        # Numeric columns are fixed. Splitting from the right preserves PTI's
        # quoted demangled names, including their template commas.
        parts = [p.strip() for p in line.rsplit(',', 6)]
        if len(parts) != 7 or parts[0] == 'Kernel':
            continue
        name = parts[0]
        calls, ns, avg, lo, hi = (int(parts[i]) for i in (1, 2, 4, 5, 6))
        percent = float(parts[3])
        # PTI CollectKernelCommandTimeStats records end-start verbatim, including
        # zero for a barrier. Keep positive total/average and compute intervals;
        # admit a zero minimum only for these explicit synchronization commands.
        minimum_valid = lo > 0 or (lo == 0 and name.strip('"') in
            ('zeCommandListAppendBarrier', 'zeCommandListAppendMemoryRangesBarrier'))
        if not (name and calls > 0 and ns > 0 and avg > 0 and minimum_valid and lo <= avg <= hi and
                avg == ns // calls and calls * lo <= ns <= calls * hi and
                math.isfinite(percent) and 0 <= percent <= 100.0001):
            raise ValueError('invalid device table row: ' + name)
        rows.append(dict(name=name, calls=calls, total_ns=ns,
                         average_ns=avg, min_ns=lo, max_ns=hi))
    if not rows or len({r['name'] for r in rows}) != len(rows) or sum(r['total_ns'] for r in rows) != int(totals[0]):
        raise ValueError('device table rows do not conserve the total')
    return dict(total_device_ns=int(totals[0]), rows=rows,
                limits='Whole-process summed service; phases, overlap, critical path and hardware saturation are not inferred.')
