"""SOURCE ONLY synthetic fixture constructors; Main owns all execution.

These are parser/configuration fixtures, never model measurements or evidence
of hardware completion. Parent fixture/source/model receipts remain mandatory.
"""


def protocol_case(n, generated=16, finish='length'):
    lines = ['RESUME 0']
    for position in list(range(6144, n - 1, 6144)) + [n - 1]:
        lines.append(f'PP {position} {n} 25000 100')
    lines.append('REUSED 0')
    for _ in range(generated):
        lines += ['T 123', 'LP -0.1 123:-0.1 124:-1 125:-2 126:-3 127:-4']
    lines.append(f'DONE {generated} {n} 50000 2000 {finish} 8 12 0 774 46560 0 0 0.0 {n} 0')
    return lines


def trace_case(n):
    return ''.join(f'strata trace: prompt chunk {offset} of {n-1}\n' for offset in range(0,n-1,6144))


GEOMETRY = {
    64: {'batched_positions':63, 'offsets':[0], 'sizes':[63],
         'modes':['routed'], 'ring':[8], 'ledger_rows':48},
    2049: {'batched_positions':2048, 'offsets':[0], 'sizes':[2048],
           'modes':['full_nonresident_stream'], 'ring':[16], 'ledger_rows':48},
    32768: {'batched_positions':32767, 'offsets':[0,6144,12288,18432,24576,30720],
            'sizes':[6144,6144,6144,6144,6144,2047],
            'modes':['full_nonresident_stream']*5+['routed'], 'ring':[16]*5+[8], 'ledger_rows':288},
}
GROUPS = [1,8,16,17,32,64]
REJECTION_CASES = [
    'ON without reference', 'OFF with reference', 'invalid N/G/batch',
    'existing immutable output directory', 'active/incomplete/failed reference',
    'reference from other boot/N/G/input/binary/source/controller/helper',
    'old FUSED default16-output reference', 'changed allowlisted actual environ/argv',
    'missing37 pin release/ASAN receipt or mode mismatch', 'missing actual lintv15',
    'missing MainreviewV3/controller Main review/controller CPU/observer CPU gate',
    'missing capability receipt or changed HOST_USM/atomic/capability fields',
    'altered hard-disabled host_handoff_concurrent or CPU share source',
    'G buffer estimate or actual READY reserve below512MiB',
    'incorrect fullstream threshold96; correct threshold2048',
    'missing/duplicate/swapped context ledger or route row',
    'MMQ1 in any of48 native layer scopes', 'missing/late/old-v1 group diagnostic',
    'group-v2-derived pin or mismatching independent OFF streamed count',
    'normal-return GU/Down incorrectly counted as GPU completion',
    'missing core runtime/changed mapped runtime/foreign oneAPI runtime',
    'abnormal owner exit/nonempty session/dump/journal fault/reset/retry',
    'incomplete PP or T/LP5 or nonfinite LP value',
    'normal early EOS: normal QUIT first, strict16 qualification FAIL afterwards',
    'same T IDs but changed selected LP5', 'input stat/source/binary/helper receipt changed',
    'long model additional full UR tracing layer', 'budget stop claimed complete',
]
