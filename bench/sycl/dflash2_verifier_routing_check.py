"""Check per-expert routing evidence on the retained DFlash2 executable."""

import fcntl
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import dflash2_decode_confidence as confidence

import cpu_work_census as census
import decode_footer as footer
import dflash2_expert_routing as routing

width, model, bank, basis = confidence.width_check, confidence.model, confidence.bank, confidence.basis
require = confidence.require

KEY = 'dflash2_verifier_routing_check_v1'
ROOT = bank.B / KEY.replace('_', '-')
CASE = 'near'
THIS = str(Path(__file__).resolve().relative_to(bank.P))
ROUTING_FILE = ROOT / CASE / 'expert-routing.dat'
LIMITS = dict(model.LIMITS)
PRODUCERS = ('src/program/generate.cpp', 'src/core/expert_source.cpp',
             'include/strata/core/cpu_work_census.hpp')
READERS = (THIS, 'tools/sycl/dflash2_expert_routing.py', 'tests/core/test_dflash2_expert_routing.py')


def inputs(revision):
    prior = basis.validation.committed(bank.RESULT, (confidence.KEY,))
    require(prior['stage'] == 'closed' and prior['passed'] and prior['complete']
            and prior['ownership_closed'] and not prior['active']
            and len(prior['processes']) == 4 and all(p['native_exit_code'] == 0 and p['passed']
                and not p['fresh_faults'] for p in prior['processes']),
            'finish the original four-process confidence comparison')
    base = confidence.baseline()
    build = confidence.leaves.speed.build.DFLASH2_KV_ROOT
    receipt = bank.previous.read(build / 'record.json')
    require(all(base[k] == prior[k] for k in ('binary', 'native_source_commit', 'native_build_sha256',
                                             'boot', 'driver', 'runtime_manifest'))
            and bank.sha(base['binary']['path']) == base['binary']['sha256'],
            'require the same qualified DFlash2 image and runtime')
    producer_pins = {p: receipt['source_files'][p] for p in PRODUCERS}
    require(all(bank.sha(build / 'source' / p) == h for p, h in producer_pins.items()),
            'actual routing or census producer differs')
    pins = confidence.experiment().pins(revision)
    for name in READERS:
        path = bank.P / name
        require(path.read_bytes() == basis.validation.git('show', revision + ':' + name),
                'commit all routing evidence inputs first')
        pins[name] = bank.sha(path)
    case_root = ROOT / CASE
    choice = dict(root=str(case_root), arm='full')
    config = confidence.configuration(base, choice)
    env = model.native_environment(base, dict(confidence.environment(base, choice),
        STRATA_TRACE='1', STRATA_DECODE_TIMING='1', STRATA_CPU_WORK_CENSUS='1',
        STRATA_SYCL_CACHE_LOAN_AUDIT='1'))
    args = config['args']
    require('--dump-routing' not in args and env['STRATA_COMMIT_SYNC'] == '1'
            and env['STRATA_WATCHDOG_S'] == '240', 'unused routing output and qualified runtime required')
    profile = Path(args[args.index('--expert-profile') + 1])
    manifest = Path(args[args.index('--pack') + 1]) / 'native_experts.txt'
    near = confidence.CHECK_PROMPT
    short = model.request_ids(Path(config['tokenizer']))['ids']
    require(len(short) == 82, 'the checked prompt suffix changed')
    request = ([i % 4096 for i in range(near - len(short))] + short, confidence.CHECK_OUTPUT)
    plan = dict(source_commit=revision, sources=dict(files=len(pins), sha256=bank.signature(pins)),
        native_source_commit=base['native_source_commit'], native_build_sha256=base['native_build_sha256'],
        binary=base['binary'], producer_pins=producer_pins, runtime_manifest=base['runtime_manifest'],
        confidence_reference=bank.signature(prior), boot=base['boot'], driver=base['driver'],
        validation=base['native_validation_probe'], drafter_assets=base['dflash2_assets'], limits=LIMITS,
        routes=dict(file=str(ROUTING_FILE), profile=str(profile), profile_sha256=bank.sha(profile),
                    manifest=str(manifest), manifest_sha256=bank.sha(manifest)),
        scope='One qualified near-context width-four diagnostic. Compare reconstructed expert jobs and '
              'logical blob visits with independent native counters before any clean speed or placement measurement.',
        cases={CASE: dict(command=[base['binary']['path'], '--serve', *args, '--dump-routing', str(ROUTING_FILE)],
                          environment_sha256=bank.signature(env), wire=dict(prompt=near, output=request[1],
                              sha256=model.restore.signature_bytes(model.wire(*request))))})
    return plan, env, [request]


def inspect(log, plan, pid, samples):
    proof = basis.validation.preload_evidence(log, pid)
    require(proof['loader'] == plan['validation']['loader']['path']
            and not re.search(r'(?m)^strata mtp: .*loaded|UR_RESULT_ERROR_|ZE_RESULT_ERROR_|teardown uncertain', log)
            and log.count('strata native head DPAS: ready=1 columns=4 layout=canonical') == 1,
            'actual runtime validation or native head differs')
    sample, = samples
    require(sample['prompt'] == confidence.CHECK_PROMPT and sample['generated'] == confidence.CHECK_OUTPUT,
            'unexpected actual prompt or output extent')
    windows = [(int(p), int(t)) for p, t in re.findall(r'^strata trace: window (\d+) (\d+)$', log, re.M)]
    positions, widths = zip(*windows, strict=True)
    require(windows[0] == (sample['prompt'] - 1, 1) and len(widths) > 1 and all(w == 4 for w in widths[1:])
            and all(1 <= b[0] - a[0] <= a[1] for a, b in zip(windows, windows[1:]))
            and sum(widths) - len(widths) == sample['offered'], 'expected the measured width-four route')
    header, = re.findall(footer.HEADER, log, re.M)
    require(tuple(map(int, header)) == (sample['prompt'], 0, sample['prompt'], sample['generated'],
                                         sample['accepted'], sample['offered']), 'native GEN footer differs')
    timing, = re.findall(footer.TIMING, log, re.M)
    copies, = re.findall(footer.COPY, log, re.M)
    clocks = dict(zip(footer.LABELS, map(float, timing[1:]), strict=True))
    require(int(timing[0]) == len(widths) and not any(map(int, copies[:6])) and clocks['PCIe'] == 0,
            'native window count or PCIe route differs')
    binding = dict(prompt_tokens=sample['prompt'], windows=len(widths), rows=sum(widths),
                   pool_ms=clocks['CPU_pool_run'] * len(widths),
                   cpu_jobs=int(copies[-2]), cpu_entries=int(copies[-1]))
    paths = plan['routes']
    require(all(bank.sha(paths[k]) == paths[k + '_sha256'] for k in ('profile', 'manifest')),
            'cache or expert formats changed during the trace')
    formats = census.manifest_formats(paths['manifest'])
    membership = census.expected_membership(paths['profile'])
    observed, = census.extract(log, [binding], formats, membership)
    usage = routing.verifier_routing_usage(ROUTING_FILE, widths, observed, formats, membership,
                                           max_bytes=plan['limits']['file_cap'])
    require('payload_match=1' in log and re.search(
        r'^strata cache loan: phase=ready generation=\d+ chunk=8192 first=132 lent=1124 decode_ready=1$',
        log, re.M), 'exact post-prefill expert cache handoff required')
    return dict(actual=observed, distinct_layer_expert_pairs=len(usage), routing_sha256=bank.sha(ROUTING_FILE),
                routing_bytes=ROUTING_FILE.stat().st_size, window_count=len(widths), last_position=positions[-1],
                usage_sha256=bank.signature(usage))


def main(action):
    require(sys.executable == basis.reference.model.PYTHON and action in ('prepare', 'run'),
            'use the pinned Python with prepare or run')
    with (bank.B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bank.builder.guards.no_model_owners([])
        if action == 'prepare':
            require(not ROOT.exists() and KEY not in basis.validation.committed(bank.RESULT),
                    'inspect the existing diagnostic first')
            revision = basis.validation.git('rev-parse', 'HEAD').decode().strip()
            plan, _, _ = inputs(revision)
            for directory in ('cache', 'tmp'):
                (ROOT / CASE / directory).mkdir(mode=0o700, parents=True)
            save(dict(stage='prepared', complete=False, passed=False, active=False,
                      ownership_closed=True, plan=plan))
            return
        record = bank.previous.read(ROOT / 'record.json')
        plan, env, requests = inputs(record['plan']['source_commit'])
        path = ROOT / CASE
        require(record == basis.validation.committed(bank.RESULT, (KEY,)) and record['plan'] == plan
                and record['stage'] == 'prepared' and record['ownership_closed'] and not record['active']
                and not any((path / name).exists() for name in ('execution.json', 'owned', ROUTING_FILE.name)),
                'require the committed unused diagnostic plan')
        outcome = width.run_case(plan, path, CASE, env, requests,
            lambda log, pid, answers: inspect(log, plan, pid, answers), limits=LIMITS)
        record.update(stage='closed' if outcome['passed'] else 'failed',
                      complete=outcome['complete'], passed=outcome['passed'],
                      ownership_closed=outcome['ownership_closed'], active=not outcome['ownership_closed'],
                      outcome=outcome)
        save(record)
        print({k: outcome.get(k) for k in ('passed', 'native_exit_code', 'fresh_faults', 'error')})
        require(outcome['passed'], 'close the original diagnostic outcome before collecting training samples')


def save(record):
    bank.stage.write(ROOT / 'record.json', record)
    results = bank.previous.read(bank.RESULT)
    results[KEY] = record
    bank.stage.write(bank.RESULT, results)


if __name__ == '__main__':
    action, = sys.argv[1:]
    main(action)
