"""Validate per-request placement from the pinned 6e40/pair log emitters.

Repository LICENSE. These are the selected model's source-derived byte counts,
not physical traffic measurements. A valid fallback placement remains visible;
the whole-prefill timing, rather than sample exclusion, decides its cost.
"""
import re

PAIR_BYTES = 84025344
CHUNK, WIDTH, PAGE, EXPERTS = 8192, 10240, 4, 512

PATTERNS = {
    'prefix': (
        'strata prefill: QSA BF16 operands, ',
        r'strata prefill: QSA BF16 operands, FP32 online state/accumulation; '
        r'KV=(?P<KV>\S+) transient_bytes=(?P<transient_bytes>\d+); '
        r'source=(?P<source>\S+) score_pitch=(?P<score_pitch>\d+) '
        r'operand_layout=(?P<operand_layout>\S+) softmax=(?P<softmax>\S+)'),
    'mode': (
        'strata MoE window configuration:',
        r'strata MoE window configuration: requested_mode=(?P<requested_mode>\d+) '
        r'effective_mode=(?P<effective_mode>\d+) maximum_rows=(?P<maximum_rows>\d+) '
        r'mixed_fp16_bytes=(?P<mixed_fp16_bytes>\d+)'),
    'bank': (
        'strata MoE expert bank:',
        r'strata MoE expert bank: requested_slots=(?P<requested_slots>\d+) '
        r'effective_slots=(?P<effective_slots>\d+) staged=(?P<staged>\d+) '
        r'host_slab_bytes=(?P<host_slab_bytes>\d+) device_bytes=(?P<device_bytes>\d+) '
        r'residual_gpu_rows=(?P<residual_gpu_rows>\d+) '
        r'allocation_limit_bytes=(?P<allocation_limit_bytes>\d+) '
        r'general_map=(?P<general_map>\S+)'),
    'placement': (
        'strata prefill: layer-major candidate, ',
        r'strata prefill: layer-major candidate, tokens=(?P<tokens>\d+) '
        r'chunk=(?P<chunk>\d+) residual_dtype=(?P<residual_dtype>\S+) '
        r'residual_element_bytes=(?P<residual_element_bytes>\d+) '
        r'residual_total=(?P<residual_total>\d+) residual_gpu_rows=(?P<residual_gpu_rows>\d+) '
        r'residual_gpu=(?P<residual_gpu>\d+) residual_host=(?P<residual_host>\d+) '
        r'residual_staging=(?P<residual_staging>\d+) layer_cache=(?P<layer_cache>\d+); '
        r'extra allocations, GPU headroom=536870912, RAM headroom=8589934592'),
}
TEXT = {'KV', 'source', 'operand_layout', 'softmax', 'general_map', 'residual_dtype'}


def require(condition, message):
    if not condition:
        raise ValueError('pair-model placement: ' + message)


def validate(log, engine_rows, arm, environment):
    """Bind each complete emitter group to one fresh, position-zero request.

    Prefill's 384-slot bank is distinct from the unchanged 1256-slot decode
    cache. Accept a coherent native 512-slot fallback, retaining its placement
    for the timing decision; never silently treat it as the requested bank.
    """
    require(arm in ('original', 'candidate'), 'unknown arm')
    require(engine_rows and all(type(n) is int and n > CHUNK for n in engine_rows),
            'actual long engine extents required')
    require(environment.get('STRATA_SYCL_PREFILL_EXPERT_BANK_SLOTS') == '384' and
            environment.get('STRATA_SYCL_PREFILL_MIXED_SCRATCH') == '1' and
            environment.get('STRATA_SYCL_PREFILL_NATIVE16') == 'bf16',
            'pinned prefill bank/scratch/precision contract changed')
    records = {kind: [] for kind in PATTERNS}
    for line_number, line in enumerate(log.splitlines()):
        for kind, (prefix, pattern) in PATTERNS.items():
            if line.startswith(prefix):
                match = re.fullmatch(pattern, line)
                require(match is not None, 'malformed ' + kind + ' record')
                records[kind].append((line_number, {
                    key: value if key in TEXT else int(value)
                    for key, value in match.groupdict().items()}))
    require(all(len(rows) == len(engine_rows) for rows in records.values()),
            'one complete mode/bank/prefix/placement group per request required')
    result, stride = [], None
    for index, n in enumerate(engine_rows):
        ordered = [records[kind][index][0] for kind in PATTERNS]
        require(ordered == sorted(set(ordered)) and
                (index == 0 or records['placement'][index-1][0] < ordered[0]),
                'request record order differs')
        row = {kind: rows[index][1] for kind, rows in records.items()}
        prefix, mode, bank, placement = (row[k] for k in PATTERNS)
        require(prefix == dict(KV='active-logical-prefix',
            transient_bytes=((n + PAGE - 1) // PAGE * PAGE) * 2056 +
                (PAIR_BYTES if arm == 'candidate' else 0),
            source='resident-Mode0', score_pitch=33, operand_layout='row-major',
            softmax='parallel-exp-ascending-sum'), 'prefix route or exact reservation differs')
        gpu_rows = placement['residual_gpu_rows']
        require(CHUNK < gpu_rows <= n and (gpu_rows == n or gpu_rows % CHUNK == 0),
                'invalid GPU residual extent')
        require(mode == dict(requested_mode=10, effective_mode=10,
            maximum_rows=min(10 * CHUNK, gpu_rows), mixed_fp16_bytes=0),
                'per-request mode/scratch admission differs')
        require(bank['requested_slots'] == 384 and bank['staged'] in (0, 1) and
                bank['effective_slots'] == (384 if bank['staged'] else EXPERTS) and
                bank['general_map'] == 'private_scratch_never_published', 'bank admission differs')
        if bank['staged']:
            require(gpu_rows == n and not (n > 81920 and 0 < n % 81920 <= CHUNK),
                    'staged bank unsupported for actual residual/tail')
        require(bank['host_slab_bytes'] > 0 and bank['host_slab_bytes'] % EXPERTS == 0,
                'invalid host slab')
        actual_stride = bank['host_slab_bytes'] // EXPERTS
        require(actual_stride % 256 == 0 and (stride is None or stride == actual_stride),
                'expert stride changed across requests')
        stride = actual_stride
        cache = bank['effective_slots'] * stride
        require(bank['device_bytes'] == cache and bank['residual_gpu_rows'] == gpu_rows,
                'bank/residual extents disagree')
        require(placement == dict(tokens=n, chunk=CHUNK, residual_dtype='bf16',
            residual_element_bytes=2, residual_total=n * WIDTH * 2,
            residual_gpu_rows=gpu_rows, residual_gpu=gpu_rows * WIDTH * 2,
            residual_host=(n - gpu_rows) * WIDTH * 2, residual_staging=0, layer_cache=cache),
                'residual/cache byte conservation differs')
        require(bank['allocation_limit_bytes'] >= max(cache, placement['residual_gpu'],
                prefix['transient_bytes']), 'single-allocation limit exceeded')
        result.append(dict(engine_rows=n, position=0, **row))
    return result
