#!/usr/bin/env python3
"""Four bounded host-only cases for Main to review and execute; no GPU imports.

Original project code under the repository LICENSE. Leaves its tiny fixtures and
receipt in the explicitly supplied fresh root; does not perform cleanup.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
D = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    options = parser.parse_args()
    options.root.mkdir(mode=0o700, parents=False, exist_ok=False)
    source = D / 'compare_screen.py'
    pin = json.loads((D / 'manifest.json').read_text())['proposal_files']['compare_screen.py']['sha256']
    if hashlib.sha256(source.read_bytes()).hexdigest() != pin:
        raise RuntimeError('comparator pin')
    spec = importlib.util.spec_from_file_location('fixture_screen_comparator', source)
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)
    tokens, positions, vocab = [0, 1, 2, 3], [0, 2], 4
    groups = []

    def capture(name, rows, *, header=None, suffix=b'', metadata_edit=None, partial_only=False):
        raw = options.root / (name + '.bin')
        payload = struct.pack('<ii', *(header or (vocab, len(positions))))
        payload += b''.join(struct.pack('<4f', *row) for row in rows) + suffix
        raw.write_bytes(payload)
        lines = ['strata-prefill-logits-v1', 'format=native-endian-int32-vocab-count-float32-rows',
                 'vocab=4', 'count=2', 'batched=3', 'head=native', 'head_ggml_type=14',
                 'row\tposition\tnext_token_id', '0\t0\t1', '1\t2\t3', 'complete\t2\t40']
        if metadata_edit:
            metadata_edit(lines)
        meta = Path(str(raw) + '.meta' + ('.partial' if partial_only else ''))
        meta.write_text('\n'.join(lines) + '\n', encoding='ascii')
        return raw, Path(str(raw) + '.meta')

    def analyze(candidate):
        paths = dict(zip(c.ARMS, (control_a, control_b, candidate)))
        return c.analyze_triplet(paths, tokens, positions, vocab=vocab)

    def rejected(label, candidate):
        try:
            analyze(candidate)
        except (ValueError, FileNotFoundError) as error:
            return dict(label=label, rejected=True, reason=str(error))
        raise AssertionError('accepted malformed fixture: ' + label)

    record = dict(passed=False, complete=False, GPU_executed=False, groups=groups,
                  scope='Host parser/oracle fault-injection only; not production model qualification')
    try:
        base = [[0.0] * 4, [0.0] * 4]
        control_a = capture('control-a', base)
        control_b = capture('control-b', base)
        shifted = capture('shifted', [[64.0] * 4, [64.0] * 4])
        result = analyze(shifted)
        assert all(abs(v - math.log(4)) < 1.e-14 for v in result['mean_NLL'].values())
        assert all(row['arms']['bf16']['top1'] == 0 and row['arms']['bf16']['top1_ties'] == 4
                   for row in result['individual_positions'])
        assert all(pair['mean_NLL_delta'] == 0 and pair['mean_L1'] == 0 and
                   pair['mean_KL_control_to_candidate'] == 0 for pair in result['pairs'].values())
        assert result['pairs']['fp32-a-vs-bf16']['logit_max_absolute_delta'] == 64
        groups.append(dict(case='hand-computable uniform distribution, ties and additive shift', passed=True))
        bad_headers = [rejected('zero committed count', capture('zero-count', base, header=(4, 0))),
                       rejected('wrong vocabulary', capture('wrong-vocab', base, header=(5, 2))),
                       rejected('trailing payload byte', capture('trailing', base, suffix=b'x')),
                       rejected('truncated last row', capture('truncated', base[:1]))]
        groups.append(dict(case='header/length integrity', passed=True, checks=bad_headers))
        bad_metadata = [
            rejected('wrong next-token target', capture('bad-target', base, metadata_edit=lambda s: s.__setitem__(8, '0\t0\t2'))),
            rejected('duplicate position', capture('bad-position', base, metadata_edit=lambda s: s.__setitem__(9, '1\t0\t3'))),
            rejected('wrong completion count', capture('bad-footer', base, metadata_edit=lambda s: s.__setitem__(10, 'complete\t1\t40'))),
            rejected('partial metadata only', capture('partial', base, partial_only=True)),
        ]
        groups.append(dict(case='position/target/completion publication integrity', passed=True, checks=bad_metadata))
        bad_finite = []
        for index, value in enumerate((math.nan, math.inf, -math.inf)):
            # Position 2 targets token 3: the corrupt token 2 is neither target
            # nor a chosen top1 assertion. Full-vocabulary validation must find it.
            rows = [[0.0] * 4, [0.0, 0.0, value, 0.0]]
            bad_finite.append(rejected('non-target nonfinite ' + str(index), capture('nonfinite-' + str(index), rows)))
        groups.append(dict(case='every vocabulary value finite', passed=True, checks=bad_finite))
        record.update(passed=True, complete=True)
    except BaseException as error:
        record['error'] = repr(error)
    record['fixture_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record['comparator_sha256'] = pin
    record['artifacts'] = [c.artifact(p) for p in options.root.iterdir() if p.is_file()]
    with (options.root / 'fixture-result.json').open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=record['passed'], complete=record['complete'], groups=len(groups))))
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
