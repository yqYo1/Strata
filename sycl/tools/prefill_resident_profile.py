#!/usr/bin/env python3
"""Repeat a CLI prefill measurement in one resident engine, checking complete heads.

The input is a completed prefill_profile.py record. It supplies model paths,
fixed memory settings and accepted first-logit hashes. Prompt and conversation
caches are disabled; each request must reread the whole prefix. Resident prompt
wall time includes request setup, so compare resident configurations with each
other and keep the CLI confirmation separate.
"""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import statistics
import subprocess
import time


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--exe', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mtp', type=Path, help='Override the reference MTP pack; without MTP the tool uses explicit diagnostic mode')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--reverse-order', action='store_true')
    parser.add_argument('--mode', choices=('wall', 'profile'), default='profile')
    parser.add_argument('--set-env', action='append', default=[], metavar='STRATA_NAME=VALUE')
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--cwd', type=Path, default=Path(__file__).resolve().parents[2])
    opts = parser.parse_args()
    if opts.repeats < 1 or opts.timeout < 1:
        parser.error('repeats and timeout must be positive')
    reference = json.loads(opts.reference.read_text())
    tokens_file = Path(reference['fixture'])
    if sha256(tokens_file) != reference['fixture_sha256']:
        parser.error('reference token fixture changed')
    tokens = list(map(int, tokens_file.read_text().split()))
    expected = {}
    for r in reference['runs']:
        head = (r['logits_sha256'], r['logits_count'], r['output_ids'])
        if r['tokens'] in expected and expected[r['tokens']] != head:
            parser.error('reference heads disagree at the same input length')
        if r['exit_code'] != 0 or not r['logits_finite'] or r.get('chunks') != 1:
            parser.error('reference contains a failed or invalid CLI run')
        expected[r['tokens']] = head
    lengths = sorted(expected)
    if not lengths or max(lengths) >= len(tokens):
        parser.error('reference fixture does not contain every input prefix')
    args = list(reference['runs'][0]['args'][1:])
    for flag in ('--tokens-file', '--max-new'):
        index = args.index(flag)
        del args[index:index + 2]
    if opts.mtp:
        if '--mtp' in args:
            index = args.index('--mtp')
            del args[index:index + 2]
        args += ['--mtp', str(opts.mtp.resolve())]
    resident_mtp = args[args.index('--mtp') + 1] if '--mtp' in args else None
    args += ['--serve', '--prompt-cache', '0', '--conversation-cache-mib', '0', '--turn-token', '-1',
             '--short-read', '0']
    exe = opts.exe.resolve()
    env = dict(os.environ, **reference['env'])
    env.pop('STRATA_SERVE_NO_MTP', None)
    if resident_mtp is None:
        env['STRATA_SERVE_NO_MTP'] = '1'
    for key in ('STRATA_PREFILL_TIMING', 'STRATA_PREFILL_TRANSFER_TIMING', 'STRATA_PREFILL_PRELOAD_PLE'):
        env.pop(key, None)
    if opts.mode == 'profile':
        env.update(STRATA_PREFILL_TIMING='1', STRATA_PREFILL_TRANSFER_TIMING='1')
    for assignment in opts.set_env:
        key, sep, value = assignment.partition('=')
        if not sep or not key.startswith('STRATA_'):
            parser.error('--set-env requires STRATA_NAME=VALUE')
        if key in ('STRATA_PREFILL_TIMING', 'STRATA_PREFILL_TRANSFER_TIMING', 'STRATA_PREFILL_PRELOAD_PLE',
                   'STRATA_DUMP_FIRST_LOGITS', 'STRATA_SERVE_NO_MTP'):
            parser.error('timing and head-dump settings are controlled by this tool')
        env[key] = value
    opts.output.mkdir(parents=True, exist_ok=True)
    dump = (opts.output / 'current-head.bin').resolve()
    env['STRATA_DUMP_FIRST_LOGITS'] = str(dump)
    report = {'exe': str(exe), 'binary_sha256': sha256(exe), 'args': args,
              'reference': str(opts.reference.resolve()), 'reference_sha256': sha256(opts.reference),
              'fixture_sha256': reference['fixture_sha256'], 'lengths': lengths,
              'mtp': resident_mtp, 'reference_mtp': reference['mtp'],
              'mode': opts.mode, 'warmup': opts.warmup, 'reverse_order': opts.reverse_order,
              'env': {key: value for key, value in env.items() if key in reference['env'] or key.startswith('STRATA_')},
              'notes': ['No prompt or conversation cache; every request must report RESUME 0 and a full read.',
                        'Resident prompt wall time includes request setup and any enabled MTP prompt work; keep CLI wall comparisons separate.',
                        'DMA, CPU staging and GPU timeline overlap; do not add or subtract their times.',
                        'Complete first-head hashes and output IDs must match the CLI reference on every request.',
                        'Warmups are recorded and validated but excluded from medians.'],
              'startup': [], 'runs': []}
    target = opts.output / 'run.json'
    def save():
        target.write_text(json.dumps(report, indent=2) + '\n')
    save()
    log = opts.output / 'engine.log'
    with log.open('wb') as handle:
        child = subprocess.Popen([str(exe)] + args, cwd=opts.cwd, env=env,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=handle)
        selector = selectors.DefaultSelector()
        selector.register(child.stdout, selectors.EVENT_READ)
        pending = bytearray()
        def line():
            deadline = time.monotonic() + opts.timeout
            while True:
                if b'\n' in pending:
                    raw, _, tail = pending.partition(b'\n')
                    pending[:] = tail
                    return raw.decode().strip()
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError('resident engine protocol timeout')
                data = os.read(child.stdout.fileno(), 65536)
                if not data:
                    raise RuntimeError('resident engine exited before protocol response')
                pending.extend(data)
        try:
            while True:
                s = line()
                report['startup'].append(s)
                if s.startswith('ERR '):
                    raise RuntimeError(s)
                if s.startswith('READY '):
                    break
            save()
            for repeat in range(-int(opts.warmup), opts.repeats):
                warmup = repeat < 0
                ascending = (repeat % 2 == 0) != opts.reverse_order
                for n in lengths if ascending else list(reversed(lengths)):
                    name = f'{n}-warmup' if warmup else f'{n}-r{repeat + 1}'
                    dump.unlink(missing_ok=True)
                    offset = log.stat().st_size
                    child.stdin.write(('GEN 1 ' + ','.join(map(str, tokens[:n + 1])) + '\n').encode())
                    child.stdin.flush()
                    protocol = []
                    ids = []
                    while True:
                        s = line()
                        protocol.append(s)
                        if s.startswith('ERR '):
                            raise RuntimeError(s)
                        if s.startswith('T '):
                            ids.append(int(s.split()[1]))
                        if s.startswith('DONE '):
                            done = s.split()
                            break
                    diagnostics = log.read_bytes()[offset:].decode()
                    (opts.output / f'{name}.log').write_text(diagnostics)
                    head = opts.output / f'{name}-logits.bin'
                    dump.rename(head)
                    values = array.array('f')
                    values.frombytes(head.read_bytes())
                    record = {'name': name, 'tokens': n, 'repeat': repeat + 1, 'warmup': warmup,
                              'protocol': protocol, 'output_ids': ids, 'logits_count': len(values),
                              'logits_sha256': sha256(head), 'logits_finite': bool(values) and all(map(math.isfinite, values)),
                              'request_prompt_wall_ms': float(done[3]), 'request_decode_wall_ms': float(done[4]),
                              'exit_code': 0}
                    pp = [s.split() for s in protocol if s.startswith('PP ')]
                    record['pp_records'] = pp
                    for prefix, key in (('strata prefill transfer: ', 'transfer'), ('strata prefill phases: ', 'phases')):
                        lines = [s[len(prefix):] for s in diagnostics.splitlines() if s.startswith(prefix)]
                        if lines:
                            record[key] = json.loads(lines[-1])
                    report['runs'].append(record)
                    save()
                    assert [s for s in protocol if s.startswith('RESUME ')] == ['RESUME 0'], name
                    assert int(done[1]) == 1 and int(done[2]) == n + 1 and int(done[8]) == 0 and int(done[14]) == n + 1, name
                    assert len(pp) == 1 and int(pp[0][1]) == n and int(pp[0][2]) == n + 1, name
                    assert record['logits_finite'] and (record['logits_sha256'], len(values), ids) == expected[n], name
                    if opts.mode == 'profile':
                        assert record.get('transfer', {}).get('chunks') == 1 and record['transfer']['tokens'] == n, name
                        assert 'phases' in record, name
                    print(json.dumps({'name': name, 'prompt_ms': record['request_prompt_wall_ms'],
                                      'whole_head_bit_identical': True}), flush=True)
            child.stdin.write(b'QUIT\n')
            child.stdin.flush()
            child.wait(timeout=30)
            if child.returncode:
                raise RuntimeError(f'resident engine exit {child.returncode}')
            report['exit_code'] = child.returncode
            report['medians'] = []
            for n in lengths:
                rows = [r for r in report['runs'] if r['tokens'] == n and not r['warmup']]
                report['medians'].append({'tokens': n, 'request_prompt_wall_ms': statistics.median(r['request_prompt_wall_ms'] for r in rows)})
            save()
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            selector.close()


if __name__ == '__main__':
    main()
