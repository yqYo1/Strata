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


def fused_moe_service(summary, histogram):
    """Count/shape reconciliation for the pinned6e40 wide routed families only.

    Projection FLOPs omit fused epilogue arithmetic. Their equivalent rate uses
    the entire fused kernel's service; it is not a peak or utilization metric.
    Caller must qualify source exclusivity, request extent and collector inputs.
    """
    ns = 'strata::experimental::vllm_xe2_native16::'
    prefix = ns + '(anonymous namespace)::'
    bases = dict(gu=prefix + 'GuSwiGLUKernel<(' + ns + 'Tile)1, false>',
                 down=prefix + 'DownScatterKernel<(' + ns + 'Tile)1, false, (' + ns + 'ScatterLayout)2>')
    bins = {}
    previous = 0
    for m, count in histogram:
        if type(m) is not int or type(count) is not int or not previous < m <= 8192 or count <= 0:
            raise ValueError('invalid fresh source job histogram')
        previous = m
        if m >= 2334:
            group = (m + 127) // 128
            cell = bins.setdefault(group, dict(calls=0, logical_rows=0))
            cell['calls'] += count
            cell['logical_rows'] += m * count
    if not bins:
        raise ValueError('no wide fused jobs in the fresh request')
    result = {}
    for family, base in bases.items():
        selected = []
        for row in summary['rows']:
            name = row['name'].strip('"')
            if not name.startswith(base): continue
            match = re.fullmatch(re.escape(base) +
                r'\[SIMD16 \{1; (10|20); ([1-9][0-9]?)\} \{128; 1; 1\}\]', name)
            if not match or int(match[1]) != (10 if family == 'gu' else 20):
                raise ValueError('unexpected fused kernel launch geometry')
            group = int(match[2])
            if group not in bins or row['calls'] != bins[group]['calls']:
                raise ValueError('fresh source and family shape counts differ')
            n, k = (1280, 2560) if family == 'gu' else (2560, 640)
            flops = 2 * bins[group]['logical_rows'] * n * k
            selected.append(dict(groups_m=group, calls=row['calls'],
                logical_rows=bins[group]['logical_rows'], padded_rows=group * 128 * row['calls'],
                projection_FLOPs=flops, service_ns=row['total_ns'], min_ns=row['min_ns'],
                max_ns=row['max_ns']))
        if len(selected) != len(bins) or {r['groups_m'] for r in selected} != set(bins):
            raise ValueError('missing or duplicate fused family shapes')
        total = sum(r['service_ns'] for r in selected)
        result[family] = dict(calls=sum(r['calls'] for r in selected),
            logical_rows=sum(r['logical_rows'] for r in selected), service_ns=total,
            projection_FLOPs=sum(r['projection_FLOPs'] for r in selected),
            projection_equivalent_TFLOP_s=sum(r['projection_FLOPs'] for r in selected) / total / 1000,
            shapes=sorted(selected, key=lambda r: r['groups_m']))
    known = tuple(bases.values())
    if any(('GuSwiGLUKernel<' in r['name'] or 'DownScatterKernel<' in r['name']) and
           not r['name'].strip('"').startswith(known) for r in summary['rows']):
        raise ValueError('unqualified fused family specialization')
    return dict(families=result, source_shape_counts_match=True,
        complete_model_coverage=False, hardware_saturation_qualified=False,
        limits='Unique-family aggregate service with fresh source count/shape checks;no per-execution UR join,critical path,physical traffic,observer-cost,quality or clean-speed qualification.')
