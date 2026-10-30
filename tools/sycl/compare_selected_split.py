#!/usr/bin/env python3
"""Compare retained selected6e40 with the original split long-prefill policy.

Main-authored from this fork's compare_moe_down/compare_qsa_k_model controllers;
repository LICENSE applies. No XeStrata implementation input. Reuse admitted
native artifacts, not old frontend pins or a rejected row-cache policy.
"""
import fcntl
import json
import math
from pathlib import Path
import re
import select
import socket
import statistics
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import compare_moe_down as down
import engine_once

d = down.diagnostic
P, B, RESULT = down.P, down.B, down.RESULT
ROOT = B/'selected-split-comparison-v2'
KEY, FRONTEND = 'selected_fallback_comparison', 'selected_fallback_frontend_v2'
ORDER = ('selected', 'split', 'split', 'selected', 'selected', 'split')
EXACT, RUNS = down.EXACT, down.RUNS
require, digest, write = down.require, down.digest, down.write


def native():
    record, scope, built, rt = down.qualified()
    selected, selected_rt = d.native()
    require(selected_rt == rt and built['selected_binary'] == selected['binary'],
            'independently qualified selected/candidate runtime or artifact differs')
    previous = scope['clean_down_comparison']
    require(previous['complete'] and previous['passed'] and previous['ownership_closed'] and
            not previous['active'] and len(previous['processes']) == 9 and
            all(p['passed'] and p['ownership_closed'] and not p['fresh_faults'] for p in previous['processes']),
            'closed original-split same-build model comparison required')
    return record, scope, built, selected, rt


def environment(root, rt, arm):
    require(arm in ('selected', 'split'), 'unknown comparison arm')
    env = down.environment(root, rt, 'original')
    env['STRATA_SYCL_PREFILL_MOE_SPLIT_ACCUMULATOR'] = str(int(arm == 'split'))
    return env


def compact(row):
    return {k: v for k, v in row.items() if k not in ('sources', 'model_assets')}


def prepare():
    record, scope, built, selected, rt = native()
    host, _ = down.frontend(root=ROOT, key=FRONTEND)
    require(KEY not in scope and not (ROOT/'record.json').exists(), 'fresh comparison only')
    config = down.configuration(ROOT)
    prior = json.loads((down.ROOT/'basis.json').read_text())
    assets = d.models.model_pins(config, prior['model_assets'], full=True)
    binaries = {'selected': selected['binary'], 'split': built['binaries']['strata']}
    native_sources = {'selected': selected['source_commit'], 'split': built['source_commit']}
    probe = down.native_probe(rt)
    plans = []
    for index, arm in enumerate(ORDER):
        root = ROOT/str(index)
        require(not root.exists(), 'fresh independent process root required')
        root.mkdir(mode=0o700)
        for part in ('tmp', 'cache'):
            (root/part).mkdir()
        write(root/'config.json', down.configuration(root))
        write(root/'plan.json', dict(binary=binaries[arm], native_validation_probe=probe))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        command = ['/bin/sh', str(P/'bench.sh'), '--config', str(root/'config.json'), '--port', str(port),
            '--prefill-exact-tokens', str(EXACT), '--think', '--runs', str(RUNS), '--profile-request-gates',
            '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600',
            '--out', str(root/'bench')]
        plans.append(dict(index=index, arm=arm, root=str(root), command=command, binary=binaries[arm],
            native_source_commit=native_sources[arm],
            generated={n: digest(root/n) for n in ('config.json', 'plan.json')},
            environment_sha256=down.fingerprint(environment(root, rt, arm))))
    hashes = [s['request_sha256'] for s in scope['clean_down_comparison']['processes'][0]['samples']]
    require(len(hashes) == RUNS and len(set(hashes)) == RUNS and
            all(re.fullmatch('[0-9a-f]{64}', s) for s in hashes), 'exact-count request reference missing')
    profile = P/'data/expert-profile.bin'
    row = dict(stage='prepared', source_commit=host['source_commit'], sources=host['sources'],
        model_assets=assets, runtime_sha256=digest(d.runtime.RESULT), boot=rt['boot'],
        native_validation_probe=probe, binaries=binaries, native_sources=native_sources,
        profile=dict(path=str(profile), identity=d.models.file_identity(profile), sha256=digest(profile)),
        plans=plans, request_sha256=hashes, processes=[],
        budgets=dict(wall=1800, cpu=5400, rss_cap=104<<30, text_cap=64<<20,
                     file_cap=128<<20, total_cap=512<<20),
        plan=dict(order=list(ORDER), runs_per_process=RUNS, input_tokens=EXACT, output_cap=24,
            question='Does the original-split whole long-prefill policy improve on selected6e40 now?',
            mechanism='A full117488-row split accumulator reduces48-layer MoE windows from96 to48, '
                'with128 versus384 effective prefill expert slots. This is a path/placement policy '
                'comparison,not an isolated accumulation or row-cache gain;hardware saturation is unknown.',
            constants='Same exact117489-read/zero-reuse bodies,117488engine rows,131072 INT8 resident '
                'context,1256 decode slots,8192chunks,BF16/vLLM/mode10,5workers,spec4,model/profile,'
                'frontend,current runtime and immutable safety settings. Candidate uses original Down '
                'kernel only. Parameter validation and retained loader remain;heavy observers stay off.',
            samples='Three adjacent opposite-order pairs,six fresh independent native processes,four '
                'requests each. Keep first long use separately and median of subsequent3 per process. '
                'Full hashes before/after cohort;count-only preflight warms tokenizer,not cold-disk TTFT.',
            acceptance='Complete all6processes/24requests. All3 paired split/selected median-time ratios '
                'must be below1 and geometric mean<=0.995 to advance. Target1000 requires all3 split '
                'warmed medians>=1000tok/s. Neither screen adopts the route.',
            stopping='Stop first source/runtime/owner/validation/fault/budget/route/sample failure. '
                'No speed-based stop,retry,replacement,exclusion or changed threshold. A late session '
                'failure discards that session;binding Restore is not tensor rollback.',
            routes='Observe each expected route,not route equality: selected384slots/81920maximum/96windows;'
                'split128slots/117488maximum/48windows with65536primary+51952tail,checked guard/drain.'
                'Both must hold all residual rows on GPU and restore1256-slot decode configuration.',
            limits='24 outputs are not sustained decode,agent session,task quality,long thinking or '
                'physical262144 fill/tail/RESTORE. Repeated decode preservation remains required. '
                'Requested bytes/service percentages do not prove physical traffic or saturation.',
            retention='Keep individual compact samples,per-arm routes,identities/gates and decision. '
                'Retire successful raw logs and consumed research after verification;retain current '
                'two native generations only for this comparison and its concrete next qualification.'),
        complete=False, passed=False, active=False, ownership_closed=True)
    write(ROOT/'record.json', row); scope[KEY] = compact(row)
    scope.update(status='selected-fallback-comparison-prepared', next_gate='Run the frozen six-process comparison.')
    write(RESULT, record)
    print('Prepared six-process selected/split comparison;commit before execution.', flush=True)


def basis():
    record, scope, built, selected, rt = native()
    host, manifest = down.frontend(root=ROOT, key=FRONTEND)
    row = json.loads((ROOT/'record.json').read_text())
    require(scope[KEY] == compact(row) and row['source_commit'] == host['source_commit'] and
            row['sources'] == host['sources'] and row['binaries'] ==
            dict(selected=selected['binary'], split=built['binaries']['strata']) and row['native_sources'] ==
            dict(selected=selected['source_commit'], split=built['source_commit']) and
            row['native_validation_probe'] == down.native_probe(rt) and row['boot'] == rt['boot'] and
            row['runtime_sha256'] == digest(d.runtime.RESULT), 'frozen comparison basis changed')
    profile = row['profile']
    require(d.models.file_identity(profile['path']) == profile['identity'] and
            digest(profile['path']) == profile['sha256'], 'placement profile changed')
    d.models.model_pins(down.configuration(ROOT), row['model_assets'])
    return record, scope, row, rt, manifest


def selected_routes(log, count):
    prefixes = ['strata prefill: layer-major candidate, ', 'strata MoE expert bank: ',
        'strata MoE window configuration: ', 'strata MoE accumulator window: ',
        'strata MoE expert bank complete: ']
    groups = [[line for line in log.splitlines() if line.startswith(prefix)] for prefix in prefixes]
    require(all(len(rows) == count for rows in groups) and 'strata MoE split ' not in log and
            [s for s in log.splitlines() if s.startswith('[native16-down] ')] ==
            ['[native16-down] fusion=large min_rows=2334'],
            'selected route records missing/duplicated or unexpected Down selector')
    result = []
    for i in range(count):
        route = d.route('\n'.join(rows[i] for rows in groups), dict(rows=EXACT-1, read=EXACT, split=False))
        require(route['residual_gpu_rows'] == EXACT-1 and route['residual_host_bytes'] == 0 and
                route['maximum_window_rows'] == 81920 and route['work']['windows'] == 96 and
                route['requested_expert_upload']['final_drain'] == 'checked', 'selected placement/window/drain differs')
        result.append(route)
    return result


def decision(processes):
    require(len(processes) == len(ORDER) and [p['arm'] for p in processes] == list(ORDER) and
            all(p['passed'] and p['complete'] and p['ownership_closed'] and not p['fresh_faults'] for p in processes),
            'six successful ordered independent processes required')
    require(len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ORDER),
            'native process was reused')
    require(all(math.isfinite(p['subsequent_median_ms']) and p['subsequent_median_ms'] > 10000 for p in processes),
            'invalid process median')
    ratios = []
    for start in (0, 2, 4):
        pair = {p['arm']: p['subsequent_median_ms'] for p in processes[start:start+2]}
        ratios.append(pair['split']/pair['selected'])
    ratio = math.exp(statistics.mean(math.log(v) for v in ratios))
    return dict(paired_time_ratios=ratios, geometric_mean_time_ratio=ratio,
        performance_screen_passed=all(v < 1 for v in ratios) and ratio <= .995,
        all_three_split_warm_medians_ge1000=all(EXACT*1000/p['subsequent_median_ms'] >= 1000
            for p in processes if p['arm'] == 'split'), inference_adopted=False,
        sustained_decode_qualified=False, physical262144_qualified=False)


def run(index):
    record, scope, row, rt, manifest = basis()
    require(row['stage'] in ('prepared', 'running') and len(row['processes']) == index < len(ORDER) and
            all(p['passed'] for p in row['processes']), 'fresh committed ordered process required;no retry')
    plan = row['plans'][index]; root = Path(plan['root']); env = environment(root, rt, plan['arm'])
    require(plan['arm'] == ORDER[index] and plan['binary'] == row['binaries'][plan['arm']] and
            not (root/'execution.json').exists() and down.fingerprint(env) == plan['environment_sha256'] and
            all(digest(root/n) == sha for n, sha in plan['generated'].items()) and
            digest(plan['binary']['path']) == plan['binary']['sha256'], 'process inputs changed')
    cursors = re.findall(r'^-- cursor: (.+)$', subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15), re.M)
    require(len(cursors) == 1, 'fresh GPU fault cursor required')
    outcome = dict(index=index, arm=plan['arm'], complete=False, passed=False, host_probe_accepted=False,
                   model_executed=None, clean_speed=True, adoption_qualified=False)
    owner = None
    def persist():
        write(root/'execution.json', dict(outcome, active=bool(owner and owner.active),
              commands=owner.commands if owner else []))
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
    def interact(io, entry):
        identity = None
        for request in range(RUNS):
            for stage in ('begin', 'end'):
                while True:
                    line = io.line(900)
                    if line.startswith('STRATA_PROFILE_GATE '):
                        break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) ==
                        dict(stage=stage, phase='prefill-only', run=request), 'unexpected request gate')
                delta = [a-b for a, b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()), initial)]
                require(min(delta) >= 0 and abs(delta[0]-delta[1]) <= 50000000, 'suspend/clock discontinuity')
                if stage == 'begin':
                    actual = json.loads((root/'engine-start.json').read_text())['native']; pid = actual['pid']
                    live = d.runtime.supervision.identity(pid)
                    require(live and live['start_ticks'] == actual['start_ticks'] and live['sid'] == entry['pid'] and
                            str(Path('/proc', str(pid), 'exe').resolve()) == str(Path(plan['binary']['path']).resolve()) and
                            (identity is None or identity == actual), 'owned native identity differs')
                    identity = actual
                    expected = engine_once.native_environment(row, env)
                    actual_env = dict(v.split(b'=', 1) for v in Path('/proc', str(pid), 'environ').read_bytes().split(b'\0') if b'=' in v)
                    require(all(actual_env.get(k.encode()) == v.encode() for k, v in expected.items()) and
                            set(actual_env)-{k.encode() for k in expected} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'},
                            'native pre-exec environment differs')
                    mapped = d.runtime.executable_mappings(Path('/proc', str(pid), 'maps').read_text())
                    require(mapped <= set(rt['runtime_files']) | {str(Path(plan['binary']['path']).resolve())} and
                            {row['native_validation_probe']['loader']['path'], row['native_validation_probe']['library']['path'],
                             env['ZE_ENABLE_ALT_DRIVERS']} <= mapped, 'native provider differs')
                    outcome['physical_device'] = down.device.device.live_device(pid, 'Intel(R) Arc(TM) B570 Graphics',
                        down.device.device.device_inventory(rt), rt)
                    allowed = {v['realpath'] for v in manifest['inventory']['files'].values()}
                    d.python_runtime.check_running(entry['pid'], P/'tools/strata_bench.py',
                        manifest['selected'], env, allowed, plan['command'][2:])
                    outcome.update(native_identity=actual, model_executed=True)
                require(not (root/'validation.log').exists() or root.joinpath('validation.log').stat().st_size == 0,
                        'parameter-validation warning/error')
                print(f'Process{index} {plan["arm"]} request{request} {stage}', flush=True)
                io.send(b'continue\n')
        io.proc.stdin.close(); deadline = time.monotonic()+180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll(); require(time.monotonic() < deadline, 'teardown deadline')
            if select.select([io.proc.stdout], [], [], .05)[0]:
                io.read(); io.pending.clear()
        io.exit(1)
    try:
        owner = d.runtime.supervision.Owner(root/'owned', persist)
        entry, _, _ = owner.run('bench', plan['command'], env, P, interaction=interact, **row['budgets'])
        require(d.runtime.supervision.closed(entry) and entry['exit_code'] == 0, 'model process failed')
        files = list((root/'bench').glob('*/results.json')); require(len(files) == 1, 'missing/duplicate official result')
        result = json.loads(files[0].read_text()); cohort = result['results']['prefill_only']
        samples, preflight = cohort['requests'], cohort['count_preflight']
        require(result['complete'] and not result['results']['failures'] and len(samples) == len(preflight) == RUNS,
                'incomplete official cohort')
        for i, sample in enumerate(samples):
            require(all(sample.get(k) is True for k in ('ok', 'eligible', 'stream_complete', 'thinking_requested', 'reasoning_observed')) and
                    sample['request_sha256'] == preflight[i]['body_sha256'] == row['request_sha256'][i] and
                    preflight[i]['counted_input_tokens'] == sample['actual_prompt_tokens'] == sample['actual_read_tokens'] == EXACT and
                    sample['actual_reused_tokens'] == 0 and 1 <= sample['generated_tokens'] <= 24 and
                    sample['prompt_tps_source'] == 'engine' and math.isfinite(sample['prompt_ms']) and
                    sample['prompt_ms'] > 10000, 'workload/clock differs')
        require(d.client.verify_request_journal(files[0].parent)['valid'], 'request journal invalid')
        basis()
        config = json.loads((root/'config.json').read_text())
        shared = dict(binary=plan['binary'], native_validation_probe=row['native_validation_probe'])
        end = d.gen.native_end(root, shared, dict(config=config), row['boot'], entry['pid'])
        log = (root/'engine.stderr').read_text()
        proof = d.runtime.validation.preload_evidence(log, end['native']['pid'])
        require(proof['loader'] == row['native_validation_probe']['loader']['path'] and
                not any(t in log for t in ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called', 'teardown uncertain',
                    'strata trace:', 'strata prefill dense:', 'strata loan audit:', 'strata MoE mixed owners:',
                    'SYCL_EVENT_QUERY', 'strata decode cpu:', 'strata decode timing:')), 'runtime/clean observer differs')
        if plan['arm'] == 'split':
            require([s for s in log.splitlines() if s.startswith('[native16-down] ')] ==
                    ['[native16-down] fusion=large min_rows=2334 split_kernel=original'], 'candidate Down policy differs')
            routes = down.route_samples(log, RUNS)
        else:
            routes = selected_routes(log, RUNS)
        require(all(digest(root/n) == sha for n, sha in plan['generated'].items()), 'generated inputs changed')
        fields = ('request_sha256', 'actual_prompt_tokens', 'actual_read_tokens', 'actual_reused_tokens',
                  'generated_tokens', 'prompt_ms', 'prompt_tps', 'prompt_tps_source', 'ttft_s', 'total_s',
                  'thinking_requested', 'reasoning_observed', 'ok', 'eligible', 'stream_complete')
        outcome.update(complete=True, host_probe_accepted=True, native_exit_code=0, validation=proof, routes=routes,
            samples=[{k: s[k] for k in fields if k in s} for s in samples],
            subsequent_median_ms=statistics.median(s['prompt_ms'] for s in samples[1:]))
        if index == len(ORDER)-1:
            d.models.model_pins(config, row['model_assets'], full=True)
    except BaseException as error:
        outcome.update(error=repr(error), complete=False, host_probe_accepted=False)
    finally:
        if owner:
            d.runtime.validation.finish(outcome, owner, cursors[0], row['boot'])
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete']); persist()
        row['processes'].append(outcome)
        row.update(stage='running' if outcome['passed'] else 'failed', ownership_closed=outcome.get('ownership_closed', False),
                   active=bool(owner and owner.active))
        if outcome['passed'] and len(row['processes']) == len(ORDER):
            row.update(stage='closed', complete=True, passed=True, decision=decision(row['processes']))
        write(ROOT/'record.json', row); scope[KEY] = compact(row)
        scope.update(status='selected-fallback-comparison-'+row['stage'],
            next_gate='Run the next frozen process;then independent decode/task/physical-context gates.' if outcome['passed']
                else 'Preserve failed outcome;resolve its concrete cause before a separately committed new scope.')
        write(RESULT, record)
    print(json.dumps({k: outcome.get(k) for k in ('index', 'arm', 'passed', 'complete', 'ownership_closed', 'fresh_faults', 'error')}), flush=True)
    require(outcome['passed'], 'selected/split process did not qualify')


if __name__ == '__main__':
    with (B/'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args = sys.argv[1:]
        if args == ['prepare-frontend']:
            native(); down.prepare_frontend(ROOT, FRONTEND)
        elif args == ['frontend']:
            native(); down.run_frontend(ROOT, FRONTEND)
        elif args == ['prepare']:
            prepare()
        elif len(args) == 2 and args[0] == 'run':
            run(int(args[1]))
        else:
            raise RuntimeError('prepare-frontend|frontend|prepare|run INDEX')
