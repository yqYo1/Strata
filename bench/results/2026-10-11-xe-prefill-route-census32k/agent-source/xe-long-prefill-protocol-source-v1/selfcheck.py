# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Finite CPU cases; source construction is not an executed qualification."""
import argparse
import json
from pathlib import Path
import protocol


def fixture(n=32768, chunk=4096, generated=16, maximum=16, finish='length',
            zero=False, lp5=True, physical=262144):
    options = dict(n_prompt=n, batched_n=n - 1, chunk_limit=chunk,
                   max_generated=maximum, physical_context=physical, vocabulary=248320,
                   output_requirement='exhausted_zero' if zero else 'positive', require_lp5=lp5)
    count = (n - 2) // chunk + 1
    starts = [i * chunk for i in range(count)]
    ends = [min(start + chunk, n - 1) for start in starts]
    lines = ['RESUME 0']
    lines += [f'PP {end} {n} {1000 + i * 1000} 100.0' for i, end in enumerate(ends)]
    lines += ['REUSED 0']
    for _ in range(generated):
        lines.append('T 123')
        if lp5:
            lines.append('LP -1.0 123:-1.0 124:-2.0 125:-3.0 126:-4.0 127:-5.0')
    counters = '0 0 0 0 0 0 0 0.0' if zero else '8 12 0 774 46560 0 0 0.0'
    lines += [f'DONE {generated} {n} {count * 1000 + 100}.0 {0 if zero else 2000}.0 '
              f'{finish} {counters} {n} 0']
    # Verified accepted drafts may exceed emitted outputs after an early EOS.
    trace = [f'strata trace: prompt chunk {start} of {n - 1}' for start in starts]
    return lines, options, trace


def cases():
    positives = [fixture(), fixture(chunk=6144), fixture(n=65536, chunk=6144),
                 fixture(n=65536, chunk=8192), fixture(n=32768, chunk=2048),
                 fixture(n=16384, chunk=4096), fixture(n=16384, chunk=8192),
                 fixture(generated=3, finish='stop'), fixture(lp5=False),
                 fixture(n=262144, chunk=8192, generated=0, zero=True),
                 fixture(n=4096, chunk=4096), fixture(generated=16, finish='stop')]
    base, opts, trace = fixture()
    negatives = []
    def add(lines, config=None):
        negatives.append((lines, opts if config is None else config))
    add(base[1:])
    add(base[:-1])
    add(base + [base[-1]])
    add(base + ['T 123'])
    add([base[0]] + base)
    add(base[:2] + [base[1]] + base[2:])
    add(base[:1] + base[2:])
    add(base[:9] + ['REUSED 0'] + base[9:])
    add(base[:10] + base[11:])
    add(base[:11] + base[12:])
    swapped = base.copy()
    swapped[1], swapped[2] = swapped[2], swapped[1]
    add(swapped)
    swapped = base.copy()
    swapped[10], swapped[11] = swapped[11], swapped[10]
    add(swapped)
    add(base + ['ERR failed'])
    add(['READY 262144'] + base)
    add(base[:-1] + [base[-1] + ' extra'])
    replacements = [
        ('RESUME 0', 'RESUME 1'), ('REUSED 0', 'REUSED 1'),
        ('PP 4096 32768 1000 100.0', 'PP 4095 32768 1000 100.0'),
        ('PP 4096 32768 1000 100.0', 'PP 4096 32767 1000 100.0'),
        ('PP 4096 32768 1000 100.0', 'PP 4096 32768 nan 100.0'),
        ('PP 4096 32768 1000 100.0', 'PP 4096 32768 1000 inf'),
        ('PP 4096 32768 1000 100.0', 'PP 4096 32768 -1 100.0'),
        ('PP 8192 32768 2000 100.0', 'PP 8192 32768 999 100.0'),
        ('T 123', 'T 248320'), ('T 123', 'T -1'), ('T 123', 'T +123'),
        ('T 123', 'T 0123'), ('T 123', 'T 9223372036854775808'),
        ('T 123', 'T 18446744073709551616'), ('T 123', 'T 1.0'),
        ('T 123', 'T\t123'), ('T 123', 'T  123'),
        ('LP -1.0', 'LP nan'), ('LP -1.0', 'LP inf'), ('LP -1.0', 'LP 0.1'),
        ('124:-2.0', '123:-2.0'), ('124:-2.0', '248320:-2.0'),
        ('124:-2.0', '124:1.0'), ('124:-2.0', '124:-0.5'),
        ('123:-1.0', '123:-1.1'), ('123:-1.0', '123:nan'),
        ('127:-5.0', '127:-5.0 extra'), ('127:-5.0', ''),
        ('length 8 12', 'cancel 8 12'), ('length 8 12', 'error 8 12'),
        ('length 8 12', 'length 13 12'), ('length 8 12', 'length 17 16'),
        (' 774 46560 ', ' 46561 46560 '), (' 0 0 0.0 32768 0', ' 0 0 1.0 32768 0'),
        (' 32768 0', ' 32767 0'), ('DONE 16 ', 'DONE 15 '),
        ('DONE 16 32768 ', 'DONE 16 32767 '), ('DONE 16 ', 'DONE 016 '),
        ('DONE 16 ', 'DONE 18446744073709551615 '),
        ('8100.0 2000.0', 'nan 2000.0'), ('8100.0 2000.0', '8100.0 inf'),
        ('8100.0 2000.0', '0.0 2000.0'), ('8100.0 2000.0', '7999.0 2000.0'),
        ('8100.0 2000.0', '8100.0 0.0'), ('8100.0 2000.0', '8100.0 -1.0'),
        (' 774 46560 ', ' 774 9223372036854775808 '),
        (' 774 46560 ', ' 774 9223372036854775807 '),
        ('length 8 12 0', 'length 8 12 1'),
        ('46560 0 0 0.0', '46560 -1 0 0.0'),
        ('46560 0 0 0.0', '46560 0 18446744073709551616 0.0'),
        ('46560 0 0 0.0', '46560 0 0 nan'),
        ('0.0 32768 0', '0.0 32768 -1'),
        ('PP 4096 32768 1000 100.0', 'PP 4096 32768 1000 -1.0'),
        ('PP 4096 32768 1000 100.0', 'PP 4096 32768 1000 1e309'),
        ('123:-1.0', '123:-1e309'),
    ]
    for old, new in replacements:
        changed = base.copy()
        index = next(i for i, line in enumerate(changed) if old in line)
        changed[index] = changed[index].replace(old, new, 1)
        if old == ' 774 46560 ' and new == ' 774 9223372036854775807 ':
            changed[index] = changed[index].removesuffix(' 0') + ' 1'
        add(changed)
    mass = base.copy()
    mass[11] = 'LP 0.0 123:0.0 124:0.0 125:0.0 126:0.0 127:0.0'
    add(mass)
    for overrides in [dict(n_prompt=32767), dict(batched_n=32768), dict(chunk_limit=6144),
                      dict(chunk_limit=0), dict(max_generated=15), dict(max_generated=0),
                      dict(max_generated=4097), dict(physical_context=32767),
                      dict(physical_context=(1 << 63)), dict(n_prompt=(1 << 64)),
                      dict(vocabulary=123), dict(vocabulary=4), dict(require_lp5=False),
                      dict(require_lp5=1), dict(output_requirement='unknown'),
                      dict(output_requirement='exhausted_zero'), dict(n_prompt=True),
                      dict(chunk_limit=1)]:
        add(base, dict(opts, **overrides))
    early, early_opts, _ = fixture(generated=3, finish='length')
    add(early, early_opts)
    zero, zero_opts, _ = fixture(n=262144, chunk=8192, generated=0, zero=True)
    add(zero, dict(zero_opts, output_requirement='positive'))
    changed_zero = zero.copy()
    changed_zero[-1] = changed_zero[-1].replace('length 0 0', 'stop 0 0')
    add(changed_zero, zero_opts)
    changed_zero = zero.copy()
    changed_zero[-1] = changed_zero[-1].replace('length 0 0', 'length 0 1')
    add(changed_zero, zero_opts)
    too_much, too_opts, _ = fixture(n=32768, chunk=4096, physical=32768)
    add(too_much, too_opts)
    add(base + ['X' * 4097])
    add(base + [''] * 10000)
    add('X' * ((1 << 20) + 1))
    bad_trace = [trace[:-1], trace + trace, trace[::-1],
                 [trace[0].replace(' of 32767', ' of 32768')] + trace[1:],
                 [trace[0].replace('chunk 0', 'chunk 00')] + trace[1:],
                 [trace[0] + ' extra'] + trace[1:],
                 ['strata trace: prompt chunk truncated'],
                 ['strata trace: prompt chunk 9223372036854775808 of 32767'] + trace[1:]]
    return positives, negatives, bad_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actual-protocol')
    parser.add_argument('--expectations')
    parser.add_argument('--actual-trace')
    args = parser.parse_args()
    positives, negatives, traces = cases()
    for lines, opts, trace in positives:
        protocol.validate(lines, **opts)
        protocol.validate_trace(trace, **{k: opts[k] for k in
                                ('n_prompt', 'batched_n', 'chunk_limit', 'physical_context')})
    for index, (lines, opts) in enumerate(negatives):
        try:
            protocol.validate(lines, **opts)
        except protocol.ValidationError:
            pass
        else:
            raise AssertionError(f'accepted negative protocol {index}')
    _, opts, _ = positives[0]
    trace_opts = {k: opts[k] for k in ('n_prompt', 'batched_n', 'chunk_limit', 'physical_context')}
    for index, lines in enumerate(traces):
        try:
            protocol.validate_trace(lines, **trace_opts)
        except protocol.ValidationError:
            pass
        else:
            raise AssertionError(f'accepted negative trace {index}')
    actual = None
    if bool(args.actual_protocol) != bool(args.expectations) or args.actual_trace and not args.actual_protocol:
        parser.error('actual protocol and explicit expectations required together')
    if args.actual_protocol:
        opts = json.loads(Path(args.expectations).read_text())
        actual = {'protocol': protocol.validate(Path(args.actual_protocol).read_text(), **opts)}
        if args.actual_trace:
            actual['trace'] = protocol.validate_trace(Path(args.actual_trace).read_text(),
                                                     **{k: opts[k] for k in trace_opts})
    print(json.dumps({'positive_protocol': len(positives), 'negative_protocol': len(negatives),
                      'positive_trace': len(positives), 'negative_trace': len(traces),
                      'actual': actual}, sort_keys=True))


if __name__ == '__main__':
    main()
