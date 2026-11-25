"""Attribute current width-four DFlash2 waits with its retained executable.

Independent Main controller, repository LICENSE. Keep caller intervals and
event-query wrapper timings in separate observed processes.
"""

from pathlib import Path
import math
import re
import statistics
import sys

sys.dont_write_bytecode = True
import dflash2_snapshot_decode as reference
import decode_service_evidence as service
import query_profile as query

bank, basis, require = reference.bank, reference.basis, reference.require
leaves = reference.leaves
THIS = str(Path(__file__).resolve().relative_to(bank.P))
KEY = 'dflash2_width4_wait_attribution_v1'
ROOT = bank.B / KEY.replace('_', '-')
REFERENCE = 'd6f2b82947018f85e8e6169bd2072fb076c03f19'
ORDER = ('clean-before', 'caller', 'query', 'clean-after')
OBSERVERS = {
    'caller': 'STRATA_SYCL_DECODE_SERVICE_PROFILE',
    'query': 'STRATA_SYCL_QUERY_PROFILE',
}
OVERHEAD_LIMIT = 1.10
PRODUCERS = ('src/program/generate.cpp', 'src/core/verify.cpp',
             'src/sycl/compat.cpp', 'include/strata/sycl/decode_service_profile.hpp')


def work_signature(process):
    samples = process['evidence']['samples']
    output_hashes = tuple((sample['output_sha256'], sample['reasoning_sha256']) for sample in samples)
    return bank.signature((reference.work(process), output_hashes))


def measured_reference():
    measured = bank.previous.historic(REFERENCE, reference.KEY)
    selected = [p for p in measured['processes'] if p['arm'] == 'incumbent']
    require(measured['stage'] == 'closed' and measured['passed']
            and measured['complete'] and measured['ownership_closed'] and not measured['active']
            and measured['decision']['matched_work'] and len(selected) == 3
            and all(p['complete'] and p['passed'] and p['ownership_closed'] and p['native_exit_code'] == 0
                    and not p['fresh_faults'] for p in selected)
            and len({work_signature(p) for p in selected}) == 1,
            'recover the completed matched width-four reference')
    return measured, work_signature(selected[0])


def baseline():
    row = reference.baseline()
    measured, expected_work = measured_reference()
    require(all(row[k] == measured[k] for k in ('binary', 'native_source_commit', 'native_build_sha256',
                                                'boot', 'driver', 'runtime_manifest')),
            'recover the exact width-four executable, configuration and timing reference')
    root = leaves.speed.build.DFLASH2_KV_ROOT
    compiled = bank.previous.read(root / 'record.json')
    hashes = {path: compiled['source_files'][path] for path in PRODUCERS}
    require(compiled['binaries']['strata'] == row['binary']
            and all(bank.sha(root / 'source' / path) == value for path, value in hashes.items())
            and OBSERVERS['caller'] in (root / 'source' / PRODUCERS[0]).read_text()
            and OBSERVERS['query'] in (root / 'source' / PRODUCERS[1]).read_text(),
            'the compiled width-four observer inputs must remain exact')
    return dict(row, width4_wait_reference=REFERENCE, width4_wait_producer_sha256=hashes,
                width4_clean_work_sha256=expected_work)


def environment(row, plan):
    env = reference.confidence.environment(row, plan)
    require(plan['arm'] in ORDER and env.get('STRATA_SYCL_VERIFY_OVERLAP', '1') == '1'
            and env.get('STRATA_SYCL_VERIFY_PCIE_EARLY', '0') == '0'
            and 'STRATA_DECODE_TIMING' not in env
            and all(env.get(flag, '0') == '0' for flag in OBSERVERS.values()),
            'unexpected observer, overlap or early-PCIe setting')
    if plan['arm'] in OBSERVERS:
        env[OBSERVERS[plan['arm']]] = '1'
    return env


def routes(text, samples, arm):
    request_numbers = list(range(1, len(samples) + 1))
    caller_rows = service.extract(text, request_numbers if arm == 'caller' else [])
    # The established route reader expects quiet logs. Pass only the checked
    # caller records to the ledger reader, leaving all normal route rows intact.
    route_text = '\n'.join(line for line in text.splitlines() if not line.startswith(service.PREFIX)) + '\n'
    evidence = reference.routes(route_text, samples, 'incumbent')
    if arm == 'caller':
        require(all(abs(v['wall_ns'] / 1e6 - sample['decode_ms']) <= 2.0
                    for v, sample in zip(caller_rows, samples, strict=True)),
                'caller ledger escaped the actual request clocks')
        evidence['caller'] = service.compact(caller_rows[-1])

    if arm == 'query':
        headers = list(re.finditer(leaves.clock.footer.HEADER, text, re.M))
        require(len(headers) == len(samples) and text.count('SYCL_EVENT_QUERY') == 4 * len(samples),
                'one query accounting group required per native response')
        last, = query.query_evidence(text[headers[-1].end():], 1)
        window_range = evidence['window_work']
        count = last['resident_pool_return']['calls']
        require(count <= 48 * window_range['windows_max'],
                'resident query calls exceed the full verifier window extent')
        evidence['query'] = dict(groups=len(headers), long_request=last)
    else:
        require('SYCL_EVENT_QUERY' not in text, 'unexpected event-query observation')
    return evidence


def decision(processes):
    require(tuple(p['arm'] for p in processes) == ORDER
            and len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ORDER)
            and all(p['passed'] and p['complete'] and p['ownership_closed'] and p['native_exit_code'] == 0
                    and not p['fresh_faults'] for p in processes), 'finish four independent width-four processes')
    _, expected_work = measured_reference()
    work = [work_signature(p) for p in processes]
    matched = len(set(work)) == 1
    matched_reference = all(signature == expected_work for signature in work)
    samples = [p['evidence']['samples'][-1] for p in processes]
    milliseconds = [sample['decode_ms'] for sample in samples]
    require(all(math.isfinite(v) and v > 0 for v in milliseconds), 'invalid sustained decode timings')
    elapsed = dict(zip(ORDER, milliseconds, strict=True))
    overhead = {arm: [elapsed[arm] / elapsed[clean] for clean in (ORDER[0], ORDER[-1])]
                for arm in OBSERVERS}
    qualified = {arm: matched and matched_reference and all(1 / OVERHEAD_LIMIT <= ratio <= OVERHEAD_LIMIT for ratio in ratios)
                 for arm, ratios in overhead.items()}
    rates = {arm: 1000 * sample['generated_tokens'] / sample['decode_ms']
             for arm, sample in zip(ORDER, samples, strict=True)}
    query_stats = processes[2]['routes']['query']['long_request']
    return dict(matched_work=matched, matched_reference_work=matched_reference,
        observed_over_clean_latency=overhead, attribution_qualified=qualified,
        TG=rates, clean_median_TG=statistics.median((rates[ORDER[0]], rates[ORDER[-1]])),
        resident_pool_query_fraction=query_stats['resident_pool_return']['wall_ns'] / (elapsed['query'] * 1e6),
        combined_resident_query_fraction=sum(query_stats[k]['wall_ns'] for k in
            ('resident_pool_return', 'resident_launch')) / (elapsed['query'] * 1e6),
        hardware_saturation_qualified=False, adopted=False)


def experiment():
    ctx = leaves.experiment()
    ctx.ROOT, ctx.KEY, ctx.THIS, ctx.ORDER = ROOT, KEY, THIS, ORDER
    ctx.baseline, ctx.configuration, ctx.environment, ctx.routes = baseline, reference.configuration, environment, routes
    ctx.evidence_reader.decide = decision
    ctx.protocol = dict(order=list(ORDER), workload=leaves.WORKLOAD, overhead_factor_limit=OVERHEAD_LIMIT,
        only_factor='Use the measured 16.694 tok/s width-four executable and placement. Enable one compiled observer at a time.',
        evidence='Use the four official warmups, then the same 33,217-input thinking-on regular-code workload. Bind caller '
                 'ledgers or query footers to the actual request. Preserve the verified route, frontier, output and expert work.',
        decision='Keep each clean and observed rate. For each observer require matched work and both observed/clean '
                 'latency ratios within the 1.10 factor before attributing request costs. Do not add query wrapper '
                 'wall time to nested caller intervals or infer device saturation.',
        stop='Stop at the first changed source, route, configuration, ownership, budget or device fault.')
    original_pins, original_current = ctx.pins, ctx.current

    def pins(revision):
        sources = original_pins(revision)
        for path in (THIS, reference.THIS, reference.confidence.THIS,
                     str(Path(service.__file__).resolve().relative_to(bank.P)),
                     str(Path(query.__file__).resolve().relative_to(bank.P))):
            require((bank.P / path).read_bytes() == basis.validation.git('show', revision + ':' + path),
                    'changed width-four attribution input: ' + path)
            sources[path] = bank.sha(bank.P / path)
        return sources

    def current(row):
        normalized = [{k: value.replace(plan['root'], '<process>')
                       for k, value in environment(row, plan).items() if k not in OBSERVERS.values()}
                      for plan in row['plans']]
        require(all(env == normalized[0] for env in normalized), 'another native setting differs between observers')
        return original_current(row)

    ctx.pins, ctx.current = pins, current
    return ctx


if __name__ == '__main__':
    reference.clock.main(experiment())
