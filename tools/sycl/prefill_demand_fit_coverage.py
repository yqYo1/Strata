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
    return layers, work


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


def verify_routed_work(text, rows, work):
    routers = workspace.records(text, 'strata MoE router census: ')
    require(len(routers) == 48, 'need one audited router window per layer')
    hashes = []
    for layer, router in enumerate(routers):
        require((int(router['layer']), int(router['first']), int(router['rows'])) == (layer, 0, rows)
                and all(re.fullmatch('[0-9a-f]{16}', router[k]) for k in ('ids', 'prefix_weights')),
                'incomplete router ID or prefix-weight audit')
        hashes.append((router['ids'], router['prefix_weights']))
    window = workspace.one(text, 'strata MoE accumulator window: ')
    require(int(window['windows']) == 48 and int(window['maximum_rows']) == rows
            and int(window['operator_max']) == 8192 and int(window['GPU_resident_windows_only']) == 1
            and int(window['dequants']) == work['nonempty_experts_across_layers']
            and int(window['jobs']) == work['planned_jobs']
            and int(window['source_H2D_bytes']) == rows * 10 * 4 * 48,
            'completed accumulator work differs from planned routing')
    return dict(jobs=work['planned_jobs'], census_sha256=work['original_census_sha256'],
                router_sha256=hashlib.sha256(json.dumps(hashes, separators=(',', ':')).encode()).hexdigest())


def server_requests(path, policy):
    """Read the traced 32769 / unrelated 33 / RESTORE +8209 native sequence."""
    require(0 < path.stat().st_size <= 64 << 20, 'missing or oversized native server capture')
    text = path.read_text()
    frames = re.split(r'^strata serve: STATE_HASH L=.*$', text, flags=re.MULTILINE)
    require(len(frames) == 4, 'expected three state publications')
    for frame in (frames[1], frames[3]):
        require(all(tag not in frame for tag in ('strata MoE planned job census:', 'strata packed expert bank:',
                'strata prefill demand-fit:', 'strata prefill fullfit upload:', 'strata MoE expert bank complete:',
                'strata MoE router census:', 'strata MoE accumulator window:')),
                'expert work outside long-prefill requests')
    results = []
    for frame, rows in ((frames[0], 32768), (frames[2], 8208)):
        layers, work = routed_layers(frame, rows)
        result = check_dispatch(frame, rows, policy, layers)
        result['routed_work'] = verify_routed_work(frame, rows, work)
        results.append(result)
    return results


def check_reader():
    # Full banks; a partial bank exactly fills its permanent slots; the next
    # one would consume its scratch slot, so all of its copies stay at first use.
    layers = [(512, 1510400, 32)] * 24 + [(469, 2176000, 468)] * 12 + [(438, 2329600, 438)] * 12
    rows = 65536
    footer = ('strata MoE expert bank complete: uploads=11640 requested_H2D_bytes='
              + str(sum(stride * n for _, stride, n in layers)) + ' permanent_hits=0 final_drain=checked')
    full = [f'strata prefill fullfit upload: policy=eager layer={i} first_row=0 logical_experts=32 '
            f'accepted_copy_commands=32 requested_H2D_bytes={32 * 1510400} '
            'publication=accepted_same_queue native_completion=unchecked_until_outer_drain' for i in range(24)]
    partial = [f'strata prefill demand-fit: layer={i} rows=65536 slots=469 accepted_copies=468 '
               f'requested_H2D_bytes={468 * 2176000}' for i in range(24, 36)]
    partial += [f'strata prefill demand-fit: layer={i} rows=65536 slots=438 accepted_copies=0 requested_H2D_bytes=0'
                for i in range(36, 48)]
    for policy, markers in zip(POLICIES, ([], full, partial, full + partial), strict=True):
        result = check_dispatch('\n'.join([*markers, footer]), rows, policy, layers)
        require(result['eligible_layers'] == list(range(24, 36)) and result['eligible_copies'] == 5616,
                'wrong synthetic capacity boundary')
    invalid = [partial[1:], [*partial, partial[0]], [line.replace('accepted_copies=0', 'accepted_copies=438')
               for line in partial], full + partial, [line.replace('rows=65536', 'rows=32768') for line in partial]]
    for markers in invalid:
        try:
            check_dispatch('\n'.join([*markers, footer]), rows, 'demand-fit-only', layers)
        except RuntimeError:
            continue
        raise RuntimeError('invalid partial-bank dispatch receipt passed')
    work = dict(planned_jobs=11640, nonempty_experts_across_layers=11640, original_census_sha256='fixture')
    routers = [f'strata MoE router census: layer={i} first=0 rows={rows} ids={i:016x} prefix_weights={i:016x}'
               for i in range(48)]
    complete = (f'strata MoE accumulator window: windows=48 dequants=11640 jobs=11640 '
                f'source_H2D_bytes={rows * 10 * 4 * 48} GPU_resident_windows_only=1 operator_max=8192 maximum_rows={rows}')
    trace = '\n'.join([*routers, complete])
    reference = verify_routed_work(trace, rows, work)
    changed = verify_routed_work(trace.replace('ids=0000000000000000', 'ids=0000000000000001'), rows, work)
    require(changed['router_sha256'] != reference['router_sha256'], 'changed router IDs must change the comparison')
    for bad in (trace.removeprefix(routers[0]), trace.replace('jobs=11640', 'jobs=11639')):
        try:
            verify_routed_work(bad, rows, work)
        except RuntimeError:
            continue
        raise RuntimeError('incomplete router work accepted')
    return dict(policies=len(POLICIES), rejected=len(invalid), exact_fit=468, overflow=438,
                routed_work_controls=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare expert upload coverage in matched native RESTORE logs.')
    for policy_name in POLICIES:
        parser.add_argument(policy_name.replace('-', '_'), metavar=policy_name.upper(), type=Path)
    args = parser.parse_args()
    results = {p: server_requests(getattr(args, p.replace('-', '_')), p) for p in POLICIES}
    for index in range(2):
        control = results['jit'][index]
        for policy in POLICIES[1:]:
            candidate = results[policy][index]
            require(all(candidate[key] == control[key] for key in
                        ('uploads', 'requested_bytes', 'routed_layout_sha256', 'routed_work')),
                    f'{policy} changed routed work on request {index}')
    require(any(request['partial_issued_copies'] for request in results['demand-fit-only']),
            'no partial bank used the candidate on either traced request')
    print(json.dumps(results, indent=2))
