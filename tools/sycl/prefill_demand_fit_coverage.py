"""Reconcile partial-bank issue order with independently observed MoE routing."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import moe_upload_census as census
import prefill_workspace_admission as workspace


require = workspace.require
POLICIES = ('jit', 'eager', 'demand-fit-only', 'eager-demand-fit')
BACKING = 1022361600


def fields(text, marker, pattern):
    rows = re.findall('^' + re.escape(marker) + pattern + '$', text, re.MULTILINE)
    require(text.count(marker) == len(rows), 'malformed ' + marker)
    return [tuple(map(int, row)) for row in rows]


def routed_layers(text, rows):
    """Extract model-specific stride and true demand from an observed request."""
    work = census.census(text, rows + 1)
    layouts = workspace.records(text, 'strata packed expert bank: ')
    histograms = [json.loads(line, object_pairs_hook=census.unique_object) for line in
                  re.findall(r'^strata MoE planned job census: (.+)$', text, re.MULTILINE)]
    require(len(layouts) == len(histograms) == 48, 'missing IQ3_S layer')
    layers = []
    for i, (layout, histogram) in enumerate(zip(layouts, histograms, strict=True)):
        stride, slots = int(layout['stride']), int(layout['slots'])
        require(int(layout['layer']) == histogram['layer'] == i and stride > 0 and stride % 256 == 0
                and int(layout['backing_bytes']) == BACKING and slots == min(512, BACKING // stride)
                and int(layout['used_bytes']) == slots * stride and layout['full_fit'] == str(int(slots == 512)),
                'wrong packed expert bank geometry')
        buckets = census.histogram(histogram['jobs_per_expert_histogram'], 0, 32)
        demand = 512 - buckets.get(0, 0)
        require(10 <= demand <= 512, 'invalid unique expert count')
        layers.append((slots, stride, demand))
    require(work['nonempty_experts_across_layers'] == sum(demand for _, _, demand in layers),
            'planned expert demand does not add up')
    return layers


def check_dispatch(text, rows, policy, layers):
    """Use the matching prompt's traced layout/demand in clean or traced runs."""
    require(policy in POLICIES and rows > 0 and len(layers) == 48, 'unplanned prompt or upload policy')
    full = fields(text, 'strata prefill fullfit upload: ',
                  r'policy=eager layer=(\d+) first_row=(\d+) logical_experts=(\d+) accepted_copy_commands=(\d+) '
                  r'requested_H2D_bytes=(\d+) publication=accepted_same_queue native_completion=unchecked_until_outer_drain')
    partial = fields(text, 'strata prefill demand-fit: ',
                     r'layer=(\d+) rows=(\d+) slots=(\d+) accepted_copies=(\d+) requested_H2D_bytes=(\d+)')
    expected_full, expected_partial = [], []
    eligible = []
    for i, (slots, stride, demand) in enumerate(layers):
        require(stride > 0 and stride % 256 == 0 and slots == min(512, BACKING // stride)
                and 10 <= demand <= 512, 'invalid traced layout or demand')
        if slots == 512:
            if policy in ('eager', 'eager-demand-fit'):
                expected_full.append((i, 0, demand, demand, demand * stride))
        else:
            copies = demand if demand < slots else 0
            if copies:
                eligible.append((i, copies, copies * stride))
            if policy in ('demand-fit-only', 'eager-demand-fit'):
                expected_partial.append((i, rows, slots, copies, copies * stride))
    require(sum(s == 512 for s, _, _ in layers) == 24 and full == expected_full and partial == expected_partial,
            'measured issue order differs from this prompt’s routed expert counts')
    end = workspace.one(text, 'strata MoE expert bank complete: ')
    counts = sum(n for _, _, n in layers)
    size = sum(stride * n for _, stride, n in layers)
    require(end['final_drain'] == 'checked' and int(end['permanent_hits']) == 0
            and int(end['uploads']) == counts and int(end['requested_H2D_bytes']) == size,
            'checked bank totals differ from the single-window routed work')
    return dict(policy=policy, uploads=counts, requested_bytes=size, eligible_layers=[i for i, _, _ in eligible],
                eligible_copies=sum(n for _, n, _ in eligible), eligible_bytes=sum(b for _, _, b in eligible),
                partial_issued_copies=sum(v[3] for v in partial), partial_issued_bytes=sum(v[4] for v in partial),
                routed_layout_sha256=hashlib.sha256(json.dumps(layers, separators=(',', ':')).encode()).hexdigest())


def server_requests(path, policy):
    """Read the traced 32769 / unrelated 33 / RESTORE +8209 native sequence."""
    require(0 < path.stat().st_size <= 64 << 20, 'missing or oversized native server capture')
    text = path.read_text()
    frames = re.split(r'^strata serve: STATE_HASH L=.*$', text, flags=re.MULTILINE)
    require(len(frames) == 4, 'expected three state publications')
    for frame in (frames[1], frames[3]):
        require(all(tag not in frame for tag in ('strata MoE planned job census:', 'strata packed expert bank:',
                'strata prefill demand-fit:', 'strata prefill fullfit upload:', 'strata MoE expert bank complete:')),
                'expert work outside long-prefill requests')
    results = []
    for frame, rows in ((frames[0], 32768), (frames[2], 8208)):
        results.append(check_dispatch(frame, rows, policy, routed_layers(frame, rows)))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare expert upload coverage in matched native RESTORE logs.')
    for policy_name in POLICIES:
        parser.add_argument(policy_name.replace('-', '_'), metavar=policy_name.upper(), type=Path)
    args = parser.parse_args()
    results = {p: server_requests(getattr(args, p.replace('-', '_')), p) for p in POLICIES}
    for n in range(2):
        for key in ('uploads', 'requested_bytes', 'routed_layout_sha256'):
            require(len({results[p][n][key] for p in POLICIES}) == 1, 'changed routed work across arms')
    print(json.dumps(results, indent=2))
