# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Strict CPU numerical evidence; capture adapter and task qualification are separate."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


class ContractError(ValueError):
    """Capture evidence violates the comparison contract."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    with path.open(encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=unique,
                         parse_constant=lambda _: require(False, 'nonfinite JSON'))


def pinned(root, entry):
    require(isinstance(entry, dict) and set(entry) == {'path', 'sha256'}, 'pin fields')
    sha = entry['sha256']
    require(isinstance(sha, str) and len(sha) == 64
            and all(c in '0123456789abcdef' for c in sha), 'SHA256 syntax')
    path = root / entry['path']
    require(path.is_file() and digest(path) == sha, 'pin mismatch')
    return path


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def validate(root, manifest):
    require(manifest['schema'] == 'strata.fixed-history-comparison.v1', 'schema')
    common = manifest['common']
    p, n, vocab = (common[k] for k in ('P', 'N', 'vocab'))
    require(all(integer(v, 1) for v in (p, n, vocab, common['max_context'])), 'dimensions')
    require(vocab >= 2 and p + n - 1 <= common['max_context'], 'context/vocab')
    require(common['endian'] in ('little', 'big') and common['stride'] == 1
            and type(common['stride']) is int, 'layout')
    require(common['position_convention'] == 'P-1+i predicts follow[i]', 'positions')
    expected_policy = {'bos': 'prompt-explicit', 'eos': 'score-all-stop-off', 'mask': 'all-follow'}
    require(common['policy'] == expected_policy, 'mask/BOS/EOS policy')
    for key, count in (('prompt_ids', p), ('follow_ids', n)):
        ids = common[key]
        require(isinstance(ids, list) and len(ids) == count
                and all(integer(i) and i < vocab for i in ids), 'input IDs')
        require(ids_hash(ids) == common[key + '_sha256'], 'input identity')
    for key in ('tokenizer', 'model'):
        pinned(root, common[key])
    budget = manifest['budget']
    require(integer(budget['max_vocab'], 2) and integer(budget['max_rows'], 1)
            and integer(budget['max_capture_bytes'], 8), 'finite budget')
    require(vocab <= budget['max_vocab'] and n <= budget['max_rows']
            and 8 + 4 * vocab * n <= budget['max_capture_bytes'], 'budget exceeded')
    require(isinstance(manifest['consumer'], str) and bool(manifest['consumer'])
            and isinstance(manifest['retire_after'], str) and bool(manifest['retire_after']),
            'retention consumer/review')
    paths = []
    for name in ('baseline', 'candidate'):
        arm = manifest[name]
        require(json.dumps(arm['common'], sort_keys=True) == json.dumps(common, sort_keys=True),
                'arm history/tokenizer/model mismatch')
        paths.append(pinned(root, arm['capture']))
        for key in ('binary', 'source', 'runtime'):
            pinned(root, arm[key])
        receipt = load_json(pinned(root, arm['receipt']))
        require(receipt['schema'] == 'strata.fixed-history-capture-receipt.v1', 'receipt schema')
        expected_bindings = {key: arm[key]['sha256'] for key in ('capture', 'binary', 'source', 'runtime')}
        require(receipt['bindings'] == expected_bindings, 'receipt bindings')
        require(json.dumps(receipt['common'], sort_keys=True) == json.dumps(common, sort_keys=True), 'receipt identity')
        require(receipt['positions'] == list(range(p - 1, p + n - 1))
                and receipt['targets'] == common['follow_ids']
                and type(receipt['observed_rows']) is int and receipt['observed_rows'] == n, 'row coverage')
        state = receipt['lifecycle']
        require(all(state[key] is True for key in
                    ('observed', 'reaped', 'owned_processes_empty', 'closed',
                     'capture_close_ok', 'normal_exit', 'fault_check_complete')), 'incomplete lifecycle')
        require(all(state[key] is False for key in
                    ('forced', 'canceled', 'deadline_hit', 'budget_hit')), 'abnormal lifecycle')
        require(type(state['exit_code']) is int and state['exit_code'] == 0
                and type(state['fault_count']) is int and state['fault_count'] == 0,
                'exit/fault')
        require(isinstance(state['boot_before'], str) and bool(state['boot_before'])
                and state['boot_before'] == state['boot_after'], 'boot')
        require(integer(state['owned_pid'], 1) and isinstance(state['owned_start'], str)
                and bool(state['owned_start']), 'ownership')
        require(receipt['source_provider'] == 'standalone-native-segmented-spec-follow-dump-logits',
                'source provider')
        # Evidence must be pinned, but only a reviewed producer adapter can attest semantics.
        for key in ('supervisor_evidence', 'fault_evidence', 'provider_evidence'):
            pinned(root, receipt[key])
    endian = '<' if common['endian'] == 'little' else '>'
    for path in paths:
        require(path.stat().st_size == 8 + 4 * vocab * n, 'exact capture size')
        with path.open('rb') as stream:
            require(struct.unpack(endian + 'ii', stream.read(8)) == (vocab, n), 'header')
    return common, paths, endian


def distribution(row, target):
    require(all(math.isfinite(x) for x in row), 'nonfinite vocabulary value')
    peak = max(row)
    shifted = [x - peak for x in row]
    normalizer = math.log(math.fsum(math.exp(x) for x in shifted))
    logp = [x - normalizer for x in shifted]
    order = sorted(range(len(row)), key=lambda i: (-row[i], i))
    return logp, {'target_logprob': logp[target], 'target_nll': -logp[target],
                  'top1': order[0], 'top2': order[1],
                  'top1_top2_margin': row[order[0]] - row[order[1]],
                  'target_rank': order.index(target) + 1}


def quantiles(values):
    values = sorted(values)

    def at(q):
        x = (len(values) - 1) * q
        lo = int(x)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (x - lo)
    return {str(q): at(q) for q in (0, .25, .5, .75, .95, 1)}


def perplexity(nll):
    try:
        value = math.exp(nll)
    except OverflowError:
        return {'value': None, 'overflow': True, 'log_value': nll}
    return {'value': value, 'overflow': False, 'log_value': nll}


def compare(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = load_json(manifest_path)
    if manifest.get('schema') == 'strata.serve-first-row-comparison.v1':
        from serve_first_row_capture_receipt import compare_serve
        return compare_serve(manifest_path)
    common, paths, endian = validate(manifest_path.parent, manifest)
    rows = []
    with paths[0].open('rb') as left, paths[1].open('rb') as right:
        left.read(8)
        right.read(8)
        size = 4 * common['vocab']
        fmt = endian + str(common['vocab']) + 'f'
        for i, target in enumerate(common['follow_ids']):
            a, am = distribution(struct.unpack(fmt, left.read(size)), target)
            b, bm = distribution(struct.unpack(fmt, right.read(size)), target)
            rows.append({'position': common['P'] - 1 + i, 'target': target,
                         'baseline': am, 'candidate': bm,
                         'target_logprob_delta_candidate_minus_baseline': b[target] - a[target],
                         'target_nll_delta_candidate_minus_baseline': a[target] - b[target],
                         'KL_baseline_to_candidate': math.fsum(
                             math.exp(x) * (x - y) for x, y in zip(a, b)),
                         'KL_candidate_to_baseline': math.fsum(
                             math.exp(y) * (y - x) for x, y in zip(a, b)),
                         'top1_agrees': am['top1'] == bm['top1']})
        require(not left.read(1) and not right.read(1), 'trailing bytes')
    # Recheck immutable input assumption after streaming: concurrent mutation is rejected.
    for name, path in zip(('baseline', 'candidate'), paths):
        require(digest(path) == manifest[name]['capture']['sha256'], 'capture changed')
    summaries = {}
    for name in ('baseline', 'candidate'):
        mean = math.fsum(r[name]['target_nll'] for r in rows) / len(rows)
        summaries[name] = {'mean_nll': mean, 'perplexity': perplexity(mean),
                           'target_rank_quantiles': quantiles([r[name]['target_rank'] for r in rows]),
                           'margin_quantiles': quantiles([r[name]['top1_top2_margin'] for r in rows])}
    for key in ('target_nll_delta_candidate_minus_baseline',
                'target_logprob_delta_candidate_minus_baseline',
                'KL_baseline_to_candidate', 'KL_candidate_to_baseline'):
        values = [r[key] for r in rows]
        summaries[key] = {'mean': math.fsum(values) / len(values), 'quantiles': quantiles(values)}
    summaries['top1_agreement'] = sum(r['top1_agrees'] for r in rows) / len(rows)
    return {'schema': 'strata.fixed-history-numerical-evidence.v1',
            'manifest_sha256': digest(manifest_path), 'rows': rows, 'aggregates': summaries,
            'tie_policy': 'descending logit; ascending token ID; ordinal rank starts at 1',
            'source_contract': 'validated declared standalone row mapping',
            'operational_contract': 'validated adapter-schema evidence; adapter production qualification required',
            'production_capture_qualified': False, 'quality_qualified': False,
            'task_quality': 'unevaluated', 'adoption': 'no automatic decision'}


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: fixed_history_logits.py comparison-manifest.json')
    try:
        result = compare(sys.argv[1])
        print(json.dumps(result, allow_nan=False, sort_keys=True))
    except (ContractError, KeyError, TypeError, OSError, ValueError, struct.error) as error:
        raise SystemExit('comparison rejected: ' + str(error)) from error


if __name__ == '__main__':
    main()
