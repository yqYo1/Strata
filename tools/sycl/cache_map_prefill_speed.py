#!/usr/bin/env python3
"""Clean repeated prefill of the corrected worker/map engine; repository LICENSE.

Reuse this fork's owned bench.sh supervision and actual long-route checks.
Three fresh processes revalidate the selected117K route,not a paired speedup.
Native inputs,decode placement and runtime safety are unchanged. Main owns tests.
"""
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import socket
import statistics
import sys

sys.dont_write_bytecode = True
import cache_map_prefill as extent
import compare_prefill_strategies as routes

entry, comparison = extent.entry, routes.comparison
base, shared = entry.base, entry.template.shared
P, B, RESULT = entry.P, entry.B, entry.RESULT
SOURCE = 'tools/sycl/cache_map_prefill_speed.py'
ROOT = entry.model.ROOT / 'prefill-speed117489-v1'
KEY = 'cache_map_prefill_speed117489_v1'
READ, RUNS = 117489, 4
ORDER = [(READ, 1)] * 3
require, digest, write = base.require, base.digest, base.write
compact = entry.template.compact


def save(row):
    write(ROOT / 'record.json', row)
    evidence = json.loads(RESULT.read_text())
    evidence[KEY] = compact(row)
    write(RESULT, evidence)


def pinned_sources(revision):
    pins = entry.pinned_sources(revision)
    pins.update(base.controls(revision, controller=Path(__file__).resolve()))
    return pins


def proof(revision):
    native, gate, providers = entry.proof()
    # Private entry context: reuse this call's fully verified native receipt,
    # without changing any imported shared context or process environment.
    ctx = extent.context()
    ctx.proof = lambda: (native, gate, providers)
    prior = json.loads((extent.ROOT / 'record.json').read_text())
    files, manifest = ctx.current(prior)
    contract, checked_manifest = ctx.python_manifest()
    require(manifest == checked_manifest and contract == prior['python_contract']
            and manifest['selected'] == prior['python_selected'], 'qualified frontend provider changed')
    require(files == providers and prior['stage'] == 'closed' and prior['passed']
            and prior['complete'] and prior['host_passed'] and prior['ownership_closed']
            and not prior['active'] and prior['binary'] == native['binaries']['strata']
            and prior['native_source_commit'] == native['source_commit']
            and prior['outcome']['native_exit_code'] == 0 and not prior['outcome']['fresh_faults']
            and len(prior['outcome']['samples']) == len(prior['outcome']['routes']) == 2
            and (prior['read_tokens'], prior['rows'], prior['runs']) == (READ, READ-1, 2),
            'closed same-engine V3 full117K lifecycle/extent qualification required')
    recovered = {name: hashlib.sha256(raw).hexdigest()
                 for name, raw in shared.gen.blobs(P, prior['source_commit'], prior['sources'])}
    current = pinned_sources(revision)
    require(recovered == prior['sources'] and all(current.get(n) == h for n, h in recovered.items())
            and set(current) - set(recovered) == {SOURCE},
            'only this committed clean-speed adapter may extend the unchanged extent inputs')
    return native, providers, manifest, prior


def native_context(row, plan):
    require(type(plan['index']) is int and 0 <= plan['index'] < len(ORDER)
            and (plan['read'], plan['heads']) == ORDER[plan['index']]
            and plan['kernel'] == 'candidate', 'unplanned clean process selection')
    return {k: row[k] for k in ('binary', 'native_source_commit', 'native_validation_probe', 'boot')}


def environment(row, plan):
    native_context(row, plan)
    env = comparison.environment(row, plan)
    expected = dict(STRATA_SYCL_PREFILL_GDN_BARRIERS3='1', STRATA_SYCL_PREFILL_GDN_FP16='1',
        STRATA_SYCL_PREFILL_GDN_COLUMNS='4', STRATA_SYCL_PREFILL_GDN_CHUNK64='0',
        STRATA_MTP_BATCH_F16='1', STRATA_SYCL_ADAPT_SYNC='1',
        STRATA_SYCL_PREFILL_DOWN_SPLIT_KERNEL='original', STRATA_PREFILL_LAYER_MAJOR='1')
    require(all(env.get(k) == v for k, v in expected.items()) and 'STRATA_TRACE' not in env,
            'immutable qualified route and clean observers required')
    return env


def request_route(text, plan):
    require((plan['read'], plan['heads'], plan['kernel']) == (READ, 1, 'candidate'),
            'unexpected request selection')
    result = routes.request_route(text, plan)
    require(result['split'] and result['split_guard_checked'] and
            result['staged_bank_checked_drain'] and result['prefill_slots'] == 128,
            'qualified full-GPU117K split/bank route required')
    return result


validate_model_text = routes.validate_model_text


def decision(processes):
    require(len(processes) == len(ORDER) and all(p['index'] == i and p['read'] == READ
            and p['heads'] == 1 and p['kernel'] == 'candidate' and p['passed'] and p['complete']
            and p['ownership_closed'] and not p['fresh_faults'] and len(p['samples']) == RUNS
            for i, p in enumerate(processes)), 'complete ordered three-process cohort required')
    require(len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes})
            == len(ORDER), 'native process identity reused')
    for p in processes:
        require(all(math.isfinite(s['prompt_ms']) and s['prompt_ms'] > 10000 and
                    math.isfinite(s['ttft_s']) and s['ttft_s'] > 0 for s in p['samples']) and
                p['subsequent_median_ms'] == statistics.median(s['prompt_ms'] for s in p['samples'][1:]),
                'invalid or inconsistent process clock')
    pp = [READ*1000/p['subsequent_median_ms'] for p in processes]
    warmed = [s for p in processes for s in p['samples'][1:]]
    return dict(read_tokens=READ, process_count=3, subsequent_samples=9,
        subsequent_process_pp=pp, median_process_pp=statistics.median(pp),
        subsequent_sample_peak_pp=max(READ*1000/s['prompt_ms'] for s in warmed),
        subsequent_process_TTFT=[statistics.median(s['ttft_s'] for s in p['samples'][1:]) for p in processes],
        first_use=[dict(prompt_ms=p['samples'][0]['prompt_ms'],
                        pp=READ*1000/p['samples'][0]['prompt_ms'], ttft_s=p['samples'][0]['ttft_s']) for p in processes],
        all_process_medians_at_least1000=all(v >= 1000 for v in pp),
        paired_gain_qualified=False, inference_adopted=False, sustained_decode_qualified=False,
        physical262144_qualified=False, physical_resource_saturation_qualified=False,
        limits='Same-engine clean117K revalidation only. First use and subsequent requests are separate;no matched gain against the retired historical engine,no sustained TG/task/agent/physical-context/RESTORE or resource-ceiling claim.')


def selfcheck():
    checks = routes.selfcheck()
    def sample(ms): return dict(prompt_ms=ms, ttft_s=ms/1000+.2)
    def process(i):
        return dict(index=i, read=READ, heads=1, kernel='candidate', passed=True, complete=True,
            ownership_closed=True, fresh_faults=[], native_identity=dict(pid=i+1, start_ticks=i+10),
            samples=[sample(125000), sample(112000), sample(112100), sample(112200)],
            subsequent_median_ms=112100)
    rows = [process(i) for i in range(3)]
    result = decision(rows)
    require(result['all_process_medians_at_least1000'] and result['subsequent_samples'] == 9 and
            len(result['first_use']) == 3 and not result['paired_gain_qualified'], 'separate repeated/first-use evidence')
    slower = copy.deepcopy(rows)
    slower[0]['samples'][1:] = [sample(118000)]*3
    slower[0]['subsequent_median_ms'] = 118000
    require(not decision(slower)['all_process_medians_at_least1000'], 'slower process hidden by aggregate')
    invalid = [rows[:2]]
    for patch in (dict(read=READ-1), dict(ownership_closed=False), dict(fresh_faults=['new fault']),
                  dict(subsequent_median_ms=1), dict(native_identity=rows[1]['native_identity'])):
        bad = copy.deepcopy(rows); bad[0].update(patch); invalid.append(bad)
    for bad in invalid:
        try: decision(bad)
        except (RuntimeError, KeyError, ValueError): continue
        raise RuntimeError('invalid repeated-speed cohort accepted')
    return dict(passed=True, inherited=checks, decision_negative=len(invalid),
        slower_process_retained=True, GPU_executed=False)


def prepare():
    controls = selfcheck(); revision = shared.gen.git('rev-parse', 'HEAD').decode().strip()
    native, providers, manifest, prior = proof(revision)
    require(not ROOT.exists() and KEY not in base.committed(RESULT), 'fresh clean cohort only')
    assets = shared.models.model_pins(prior['configuration'], prior['model_assets'], full=True)
    ROOT.mkdir(mode=0o700); write(ROOT/'runtime-inputs.json', providers)
    env = dict(prior['environment']); env.pop('STRATA_TRACE', None)
    row = dict(stage='prepared', source_commit=revision, sources=pinned_sources(revision), model_assets=assets,
        binary=native['binaries']['strata'], native_source_commit=native['source_commit'],
        native_validation_probe=prior['native_validation_probe'], configuration=copy.deepcopy(prior['configuration']),
        environment_template=env, python_contract=prior['python_contract'], python_selected=manifest['selected'],
        template=prior['template'], boot=prior['boot'], driver=prior['driver'],
        runtime_manifest=dict(path=str(ROOT/'runtime-inputs.json'), sha256=digest(ROOT/'runtime-inputs.json'), files=len(providers)),
        qualification_basis=dict(extent_result_key=extent.KEY, native_result_key=entry.model.KEY,
            worker_device_result_key=entry.device.KEY, original_extent_inputs_unchanged=True,
            native_recompiled=False, only_new_controller=SOURCE),
        controls=controls, plans=[], processes=[], request_hashes={}, host_passed=False,
        passed=False, complete=False, active=False, ownership_closed=True, inference_adopted=False,
        whole_bytes=256 << 20, budgets=dict(wall=1200, cpu=3600, rss_cap=104 << 30,
            text_cap=16 << 20, file_cap=16 << 20, total_cap=64 << 20),
        plan=dict(question='Does the corrected worker/map engine reproduce>=1000prefill tok/s at117489fresh input tokens across three independent processes?',
            order='Three fresh same-route native processes;four bench.sh exact-input requests per process,thinking on,zero reuse,1..24outputs. First request separate;median of subsequent3 is the process unit.',
            configuration='Unchanged552b engine,GDN16/MTP16,throughput,native16 BF16 HC,full-GPU residuals,128staged prefill slots,checked65536-row primary/tail split,head retirement1. Preserve1256decode slots,5workers,spec4,8192chunk,131072INT8resident context,zero adaptive swaps and immutable ADAPT_SYNC1/runtime safety.',
            observers='Remove STRATA_TRACE only;retain quiet parameter validation and the native loader probe until process exit. Actual always-on native16/GDN3/bank/split/Down headers plus immutable selectors bind the prior traced dispatch;clean timing has no per-launch GDN16/MTP16 count.',
            acceptance='Every sample/body/read/reuse/thinking/clock,request-route,provider/source/binary,owner/native-exit/fault and quiet-final-validation gate must pass. Same request-body vector across processes. Retain all12samples. The117K throughput milestone requires all3warmed process medians>=1000;no minimum gain or comparison with an unmatched historical engine.',
            stopping='Check each actual route before continuing. Stop on source/provider/budget/owner/clock/route/validation/fault failure;no retry,replacement,speed-based exclusion or hash refresh. Commit each process result before the next. Full15model hashes before/after cohort;quick identities throughout.',
            limits='Prefill/TTFT revalidation only,not inference adoption,sustained TG,task/agent,full262144/RESTORE or hardware-saturation evidence. Keep both explicit prefill strategies and independent decode.',
            retention='Keep compact individual samples,route conservation and exact source/config/native decisions. Retire successful raw output after closure/source recovery. Keep V3record/config/plan/runtime inputs while this proof consumes them;no new build generation.'))
    for i, (read, heads) in enumerate(ORDER):
        root = ROOT/str(i); root.mkdir(mode=0o700)
        for part in ('tmp', 'cache'): (root/part).mkdir()
        config = copy.deepcopy(row['configuration']); config['log'] = str(root/'engine.stderr')
        require(config['args'][-2:] == ['--prefill-strategy', 'throughput'], 'explicit throughput selection required')
        write(root/'config.json', config)
        write(root/'plan.json', dict(binary=row['binary'], native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        plan = dict(index=i, kernel='candidate', read=read, heads=heads, root=str(root),
            generated={n:digest(root/n) for n in ('config.json', 'plan.json')},
            command=['/bin/sh', str(P/'bench.sh'), '--config', str(root/'config.json'), '--port', str(port),
                '--prefill-exact-tokens', str(read), '--think', '--runs', str(RUNS), '--profile-request-gates',
                '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root/'bench')])
        plan['environment_sha256'] = comparison.fingerprint(environment(row, plan)); row['plans'].append(plan)
    save(row); print('Prepared three clean same-engine117K processes;commit before host probes.', flush=True)


def current(row):
    require(base.committed(RESULT)[KEY] == compact(row), 'commit exact clean-cohort state')
    native, providers, manifest, _ = proof(row['source_commit'])
    require(row['binary'] == native['binaries']['strata'] and row['native_source_commit'] == native['source_commit']
            and pinned_sources(row['source_commit']) == row['sources'], 'frozen clean inputs changed')
    inv = row['runtime_manifest']
    require(digest(inv['path']) == inv['sha256'] and json.loads(Path(inv['path']).read_text()) == providers
            and len(providers) == inv['files'], 'runtime inventory changed')
    require(digest(row['template']['path']) == row['template']['sha256'] and
            base.guards.driver_guard() == row['driver'] and manifest['selected'] == row['python_selected'],
            'frontend/template/platform changed')
    require(base.supervision.artifact_bytes(ROOT) <= row['whole_bytes'], 'whole campaign output budget')
    shared.models.model_pins(row['configuration'], row['model_assets'], full=False)
    return providers, manifest


if __name__ == '__main__':
    args = sys.argv[1:]
    require(args in (['selfcheck'], ['prepare'], ['host']) or
            len(args) == 1 and args[0].isdigit() and int(args[0]) < len(ORDER),
            'use selfcheck|prepare|host|INDEX0..2')
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); base.guards.no_model_owners([])
        if args == ['selfcheck']: print(json.dumps(selfcheck()), flush=True)
        elif args == ['prepare']: prepare()
        elif args == ['host']: comparison.host(sys.modules[__name__])
        else: comparison.run(int(args[0]), sys.modules[__name__])
