#!/usr/bin/env python3
"""Matched official-engine prefill comparison; repository LICENSE.

Main-authored from this fork's owned benchmark controls. No native build or
XeStrata inputs. Keep diagnostic qualification, clean timing and adoption apart.
"""
import copy
import fcntl
import hashlib
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
import official_model_entry as entry
import engine_once

build, shared = entry.build, entry.shared
P, B, RESULT = entry.P, entry.B, entry.RESULT
ROOT = B / 'official-prefill-speed-v1'
KEY, READS, RUNS = 'official_prefill_speed_v1', (65537, 117489), 4
ARMS = (0, 1, 1, 0, 0, 1)
ORDER = [(read, arm) for pair in range(3) for read in READS for arm in ARMS[2*pair:2*pair+2]]
require, digest, write = entry.require, entry.digest, entry.write


def compact(row):
    return entry.compact(row)


def save(row):
    write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text())
    data[KEY] = compact(row)
    write(RESULT, data)


def environment(row, plan):
    old = row['environment_template']['STRATA_MEASUREMENT_ROOT']
    env = {k: v.replace(old, plan['root']) for k, v in row['environment_template'].items()}
    env.pop('STRATA_TRACE', None)  # Presence, including "0", enables detailed logs.
    env['STRATA_SYCL_PREFILL_TOPK_EARLY'] = str(plan['topk'])
    require(not env.get('LD_PRELOAD') and all(v == '0' for k, v in env.items()
            if '_PROFILE' in k or '_AUDIT' in k), 'clean observers required')
    require(env['EnableDirectSubmission'] == '0' and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1'
            and env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1', 'runtime safety settings differ')
    return env


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare():
    controls = selfcheck()
    built, _, files = entry.proof()
    data = build.committed(RESULT)
    prior = data[entry.KEY]
    require(prior['passed'] and prior['complete'] and prior['ownership_closed'] and not prior['active']
            and prior['binary'] == built['binaries']['strata'] and not prior['outcome']['fresh_faults'],
            'closed same-build model-entry qualification required')
    require(data['qsa_topk_early_component_v2']['decision']['mechanism_screen_passed'], 'component speed gate required')
    require(not ROOT.exists() and KEY not in data, 'fresh comparison only')
    contract, manifest = entry.python_manifest()
    config = copy.deepcopy(prior['configuration'])
    expected = json.loads(shared.gen.git('show', '1e676d554:' + str(shared.TEMPLATE.relative_to(P))))['model_assets']
    assets = shared.models.model_pins(config, expected, full=True)
    revision = shared.gen.git('rev-parse', 'HEAD').decode().strip()
    sources = shared.pins(revision)
    ROOT.mkdir(mode=0o700)
    write(ROOT / 'runtime-inputs.json', files)
    row = dict(stage='prepared', source_commit=revision, sources=sources, model_assets=assets,
        native_source_commit=built['source_commit'], binary=built['binaries']['strata'],
        native_validation_probe=prior['native_validation_probe'], configuration=config,
        environment_template=prior['environment'], python_contract=contract, python_selected=manifest['selected'],
        template=prior['template'], boot=built['boot'], driver=built['driver'],
        runtime_manifest=dict(path=str(ROOT / 'runtime-inputs.json'), sha256=digest(ROOT / 'runtime-inputs.json'), files=len(files)),
        plans=[], processes=[], request_hashes={}, controller_checks=controls, host_passed=False, complete=False, passed=False,
        active=False, ownership_closed=True, inference_adopted=False,
        budgets=dict(wall=1800, cpu=5400, rss_cap=104 << 30, text_cap=16 << 20, file_cap=16 << 20, total_cap=64 << 20),
        plan=dict(question='Does exact-completion Top-k improve the same official engine at64K and117K?',
            mechanism='The measured Top-k component improves about3.94%;previous117K Top-k service is4.4289s. '
                'If that service reduction transferred entirely to a117s critical path,the effect would be '
                'about0.15%. Overlap,clock and shape differences prevent treating this as a hardware ceiling.',
            constants='Same native ELF,official2026.2 providers,BF16/vLLM,split policy enabled,original Down,'
                'eight-job waits,8192chunks,1256decode slots,5workers,spec4,INT8 resident131072context. '
                'Only Top-k0/1 differs before native exec;no runtime environment mutation.',
            samples='12fresh processes:3adjacent opposite-order pairs per length,interleaved by length. '
                'Each has4exact-count thinking-on fresh requests;first long use separate,median of subsequent3 '
                'is the process unit. Warmup chat and count-only tokenizer preflight are outside prompt timing. '
                'This is not cold-disk TTFT. Keep all individual prompt times,PP and client TTFT.',
            acceptance='Finish all12processes/48requests. Per length,all3paired on/off warmed-time ratios<1 '
                'and geometric mean<=0.999 permit further Top-k qualification. Report TTFT and first use '
                'separately. Target1000 requires all3on warmed process medians>=1000 at a length>=64K. '
                'Neither screen adopts the engine or proves decode preservation.',
            stopping='Stop first input,owner,clock,warning,fault,budget,route or sample failure. No speed-based '
                'stop,rerun,replacement,exclusion or post-hoc threshold changes. Commit every closed process '
                'before the next. Full model-asset hashes before/after cohort;identities throughout.',
            routes='Actual layer-major/GPU residual/work/Down counters and checked drain/guard per request. '
                '64K:384prefill slots,ordinary accumulator,48windows;117K:128slots,65536primary+51952tail,48windows. '
                'Top-k dispatch is source-bound to the immutable prefill policy and actual layer-major route; '
                'no per-kernel trace or arithmetic/VRAM/PCIe saturation claim.',
            limits='Two lengths supply local observations,not a validated global scaling law. Short24token '
                'outputs do not qualify sustained decode,task quality,agent session,long thinking or '
                'physical262144fill/tail/RESTORE. Selected6e40fallback remains unchanged.',
            retention='Compact individual samples,routes,identities and decisions only. Retire successful raw '
                'journals,traces and caches after verification. Retain current official candidate and selected '
                'fallback for the imminent phase-specific comparison and remaining qualification.'))
    for index, (read, arm) in enumerate(ORDER):
        root = ROOT / str(index)
        root.mkdir(mode=0o700)
        for name in ('tmp', 'cache'):
            (root / name).mkdir()
        cfg = copy.deepcopy(config)
        cfg['log'] = str(root / 'engine.stderr')
        write(root / 'config.json', cfg)
        write(root / 'plan.json', dict(binary=row['binary'], native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        plan = dict(index=index, read=read, topk=arm, root=str(root),
            generated={n: digest(root / n) for n in ('config.json', 'plan.json')},
            command=['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(port),
                '--prefill-exact-tokens', str(read), '--think', '--runs', str(RUNS), '--profile-request-gates',
                '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root / 'bench')])
        plan['environment_sha256'] = fingerprint(environment(row, plan))
        row['plans'].append(plan)
    save(row)
    print('Prepared12matched official-engine processes;commit before host checks.', flush=True)


def current(row):
    require(build.committed(RESULT)[KEY] == compact(row), 'commit exact comparison state')
    built, _, _ = entry.proof()
    require(built['binaries']['strata'] == row['binary'] and built['source_commit'] == row['native_source_commit'], 'native generation differs')
    require(shared.pins(row['source_commit']) == row['sources'], 'comparison inputs changed')
    inv = row['runtime_manifest']
    require(digest(inv['path']) == inv['sha256'], 'provider inventory changed')
    files = json.loads(Path(inv['path']).read_text())
    require(len(files) == inv['files'] and all(digest(p) == sha for p, sha in files.items()), 'provider bytes changed')
    require(digest(row['template']['path']) == row['template']['sha256'] and
            build.guards.driver_guard() == row['driver'], 'tokenizer/driver/fault state differs')
    shared.models.model_pins(row['configuration'], row['model_assets'], full=False)
    contract, manifest = entry.python_manifest()
    require(contract == row['python_contract'] and manifest['selected'] == row['python_selected'], 'Python contract differs')
    return files, manifest


def host():
    row = json.loads((ROOT / 'record.json').read_text())
    _, manifest = current(row)
    require(row['stage'] == 'prepared' and not (ROOT / 'host').exists(), 'fresh host probes only')
    owner = build.supervision.Owner(ROOT / 'host')
    env = environment(row, row['plans'][0])
    reports = []
    try:
        for role, target in (('benchmark', 'tools/strata_bench.py'), ('server', 'serve/server.py')):
            command = [manifest['selected']['path'], str(shared.python_runtime.SOURCE), 'probe', str(P / target), row['template']['path']]
            run, out, err = owner.run(role, command, env, P, wall=45, cpu=30, rss_cap=512 << 20,
                text_cap=512 << 10, file_cap=2 << 20, total_cap=8 << 20)
            require(build.supervision.closed(run) and run['exit_code'] == 0 and err.stat().st_size == 0, 'host probe failed')
            result = json.loads(out.read_text())
            require(str(Path(result['executable']).resolve()) == manifest['selected']['file']['realpath'], 'host interpreter differs')
            allowed = {v['realpath'] for v in manifest['inventory']['files'].values()}
            for name in [*result['modules'], *result['code_maps']]:
                path = Path(name).resolve()
                if path.is_relative_to(P):
                    require(str(path.relative_to(P)) in row['sources'], 'unfrozen project import')
                else:
                    require(str(path) in allowed, 'host provider differs')
            reports.append(dict(role=role, passed=True))
        row.update(host_passed=True, stage='ready', host_results=reports)
    except BaseException as error:
        row.update(stage='failed', error=repr(error))
    finally:
        row.update(active=bool(owner.active), ownership_closed=owner.active is None and all(build.supervision.closed(v) for v in owner.commands))
        save(row)
    require(row['host_passed'] and row['ownership_closed'], row.get('error', 'host check failed'))
    print('Current-source frontend probes passed;commit before native execution.', flush=True)


def routes(text, read):
    prefixes = ('strata prefill: layer-major candidate, ', 'strata MoE expert bank: ',
        'strata MoE window configuration: ', 'strata MoE accumulator window: ',
        'strata MoE expert bank complete: ', 'strata MoE split admission: ')
    groups = [[dict(re.findall(r'(\w+)=([^\s;]+)', line)) for line in text.splitlines() if line.startswith(prefix)] for prefix in prefixes]
    require(all(len(v) == RUNS for v in groups), 'missing/duplicate route records')
    tails = [dict(re.findall(r'(\w+)=([^\s;]+)', line)) for line in text.splitlines() if line.startswith('strata MoE split complete: ')]
    fused = re.findall(r'^strata MoE Down fusion: fused_jobs=(\d+) fused_rows=(\d+) separate_jobs=(\d+) min_rows=2334$', text, re.M)
    n, split = read - 1, read == 117489
    require(len(tails) == (RUNS if split else 0) and len(fused) == RUNS, 'split/Down route records differ')
    result = []
    for i in range(RUNS):
        main, bank, window, work, end, admission = (v[i] for v in groups)
        require(int(main['tokens']) == int(main['residual_gpu_rows']) == n and main['residual_dtype'] == 'bf16'
                and main['chunk'] == '8192' and main['residual_host'] == '0' and bank['staged'] == '1'
                and int(bank['effective_slots']) == (128 if split else 384)
                and window['requested_mode'] == window['effective_mode'] == '10'
                and int(window['maximum_rows']) == int(work['maximum_rows']) == n and int(work['windows']) == 48
                and end['final_drain'] == 'checked' and int(admission['active']) == int(split), 'actual prefill placement differs')
        if split:
            tail = (n - 65536) * 2560 * 4
            finish = tails[i]
            require(admission['rows'] == finish['rows'] == str(n) and admission['primary_rows'] == finish['primary_rows'] == '65536'
                    and int(admission['tail_payload_bytes']) == int(finish['tail_bytes']) == tail
                    and int(admission['extra_allocation_bytes']) == (tail + 255) // 256 * 256 + 256
                    and int(admission['max_single_allocation_bytes']) >= int(admission['extra_allocation_bytes'])
                    and admission['decode_changed'] == finish['decode_changed'] == '0'
                    and finish['guard'] == finish['final_drain'] == 'checked', 'split bounds/guard/drain differs')
        jobs, rows, separate = map(int, fused[i])
        require(jobs > 0 and rows >= 2334 * jobs and jobs + separate == int(work['jobs']), 'Down work conservation differs')
        result.append(dict(rows=n, residual_gpu_rows=n, prefill_slots=int(bank['effective_slots']), split=split,
            windows=48, jobs=int(work['jobs']), fused_down_jobs=jobs, maximum_window_rows=n,
            checked_drain=True, physical_saturation_qualified=False))
    return result


def decision(processes):
    require(len(processes) == len(ORDER) and all(p['passed'] and p['complete'] and p['ownership_closed']
            and not p['fresh_faults'] for p in processes), 'complete cohort required')
    require(len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ORDER), 'native process reused')
    require(all(math.isfinite(p['subsequent_median_ms']) and p['subsequent_median_ms'] > 10000 for p in processes), 'invalid process median')
    result = {}
    for read in READS:
        selected = [p for p in processes if p['read'] == read]
        require([p['topk'] for p in selected] == list(ARMS), 'comparison order differs')
        ratios = []
        for i in (0, 2, 4):
            pair = {p['topk']: p['subsequent_median_ms'] for p in selected[i:i+2]}
            ratios.append(pair[1] / pair[0])
        gm = math.exp(statistics.mean(math.log(v) for v in ratios))
        result[str(read)] = dict(paired_time_ratios=ratios, geometric_mean_time_ratio=gm,
            mechanism_screen_passed=all(v < 1 for v in ratios) and gm <= .999,
            all_three_on_warm_medians_ge1000=all(read * 1000 / p['subsequent_median_ms'] >= 1000 for p in selected if p['topk'] == 1))
    return dict(lengths=result, inference_adopted=False, sustained_decode_qualified=False, physical262144_qualified=False)


def selfcheck():
    """Reject corrupted placement and comparison evidence without device work."""
    logs = []
    for read in READS:
        n, split = read - 1, read == 117489
        tail = (n - 65536) * 2560 * 4
        parts = [f'strata prefill: layer-major candidate, tokens={n} residual_gpu_rows={n} residual_dtype=bf16 chunk=8192 residual_host=0',
            f'strata MoE expert bank: staged=1 effective_slots={128 if split else 384}',
            f'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows={n}',
            f'strata MoE accumulator window: maximum_rows={n} windows=48 jobs=48',
            'strata MoE expert bank complete: final_drain=checked',
            'strata MoE Down fusion: fused_jobs=2 fused_rows=4668 separate_jobs=46 min_rows=2334',
            f'strata MoE split admission: active={int(split)} rows={n} primary_rows=65536 tail_payload_bytes={tail} '
                f'extra_allocation_bytes={tail+256} max_single_allocation_bytes={1<<30} decode_changed=0']
        if split:
            parts.append(f'strata MoE split complete: rows={n} primary_rows=65536 tail_bytes={tail} guard=checked final_drain=checked decode_changed=0')
        text = '\n'.join(parts * RUNS)
        require(len(routes(text, read)) == RUNS, 'valid route rejected')
        logs.append(text)
    bad = [(logs[0] + '\nstrata MoE accumulator window: jobs=48', READS[0]),
        (logs[0].replace('strata MoE expert bank:', 'missing bank:'), READS[0]),
        (logs[0].replace('effective_slots=384', 'effective_slots=128'), READS[0]),
        (logs[0].replace('residual_gpu_rows=65536', 'residual_gpu_rows=0'), READS[0]),
        (logs[1].replace('guard=checked', 'guard=failed'), READS[1]),
        (logs[1].replace('separate_jobs=46', 'separate_jobs=47'), READS[1])]
    for text, read in bad:
        try:
            routes(text, read)
        except (RuntimeError, ValueError, KeyError):
            continue
        raise RuntimeError('invalid route accepted')
    processes = [dict(read=read, topk=arm, passed=True, complete=True, ownership_closed=True, fresh_faults=[],
        native_identity=dict(pid=i+1, start_ticks=i+100), subsequent_median_ms=20000 if arm == 0 else 19800)
        for i, (read, arm) in enumerate(ORDER)]
    require(all(v['mechanism_screen_passed'] for v in decision(processes)['lengths'].values()), 'passing decision rejected')
    processes[1]['subsequent_median_ms'] = 21000
    require(not decision(processes)['lengths'][str(READS[0])]['mechanism_screen_passed'], 'one slower pair accepted')
    duplicate = copy.deepcopy(processes)
    duplicate[1]['native_identity'] = duplicate[0]['native_identity']
    try:
        decision(duplicate)
    except (RuntimeError, ValueError, KeyError):
        return dict(passed=True, valid_route_shapes=2, invalid_routes_rejected=len(bad), comparison_controls=3)
    raise RuntimeError('reused native identity accepted')


def run(index):
    row = json.loads((ROOT / 'record.json').read_text())
    files, manifest = current(row)
    require(row['host_passed'] and row['stage'] in ('ready', 'running') and len(row['processes']) == index < len(ORDER)
            and all(p['passed'] for p in row['processes']), 'fresh ordered process only;no retry')
    plan = row['plans'][index]
    root, env = Path(plan['root']), environment(row, plan)
    require(not (root / 'execution.json').exists() and (plan['read'], plan['topk']) == ORDER[index]
            and fingerprint(env) == plan['environment_sha256'] and
            all(digest(root / n) == sha for n, sha in plan['generated'].items()), 'process plan changed')
    cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines() if v.startswith('-- cursor: '))
    outcome, owner = dict(index=index, read=plan['read'], topk=plan['topk'], complete=False,
        host_probe_accepted=False, model_executed=False, clean_speed=True), None
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
    def persist():
        write(root / 'execution.json', dict(outcome, active=bool(owner and owner.active), commands=owner.commands if owner else []))
    def interact(io, command):
        identity = None
        for request in range(RUNS):
            for stage in ('begin', 'end'):
                while True:
                    line = io.line(900)
                    if line.startswith('STRATA_PROFILE_GATE '):
                        break
                require(json.loads(line.removeprefix('STRATA_PROFILE_GATE ')) == dict(stage=stage, phase='prefill-only', run=request), 'request gate differs')
                delta = [a - b for a, b in zip((time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns()), initial)]
                require(min(delta) >= 0 and abs(delta[0] - delta[1]) <= 50000000, 'suspend/clock discontinuity')
                claim = json.loads((root / 'engine-start.json').read_text())['native']
                actual = build.supervision.identity(claim['pid'])
                require(actual and actual['start_ticks'] == claim['start_ticks'] and actual['sid'] == command['pid']
                        and str(Path('/proc', str(claim['pid']), 'exe').resolve()) == row['binary']['path']
                        and (identity is None or identity == claim), 'owned native identity differs')
                identity = claim
                expected = engine_once.native_environment(row, env)
                observed = dict(v.split(b'=', 1) for v in Path('/proc', str(claim['pid']), 'environ').read_bytes().split(b'\0') if b'=' in v)
                require(all(observed.get(k.encode()) == v.encode() for k, v in expected.items())
                        and set(observed) - {k.encode() for k in expected} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'}, 'native environment differs')
                mapped = shared.runtime.executable_mappings(Path('/proc', str(claim['pid']), 'maps').read_text())
                require(mapped <= set(files) | {row['binary']['path']} and {row['native_validation_probe']['loader']['path'],
                    row['native_validation_probe']['library']['path'], env['ZE_ENABLE_ALT_DRIVERS']} <= mapped, 'native providers differ')
                shared.python_runtime.check_running(command['pid'], P / 'tools/strata_bench.py', manifest['selected'], env,
                    {v['realpath'] for v in manifest['inventory']['files'].values()}, plan['command'][2:])
                require(not (root / 'validation.log').exists() or (root / 'validation.log').stat().st_size == 0, 'validation warning/error')
                outcome.update(model_executed=True, native_identity=claim)
                print(f'Official process{index} read{plan["read"]} topk{plan["topk"]} request{request} {stage}', flush=True)
                io.send(b'continue\n')
        io.proc.stdin.close()
        deadline = time.monotonic() + 180
        while not io.exit_watch.poll(0) or not io.eof:
            io.poll()
            require(time.monotonic() < deadline, 'native teardown deadline')
            if select.select([io.proc.stdout], [], [], .02)[0]:
                io.read()
                io.pending.clear()
        io.exit(1)
    try:
        owner = build.supervision.Owner(root / 'owned', persist)
        command, _, _ = owner.run('bench', plan['command'], env, P, interaction=interact, **row['budgets'])
        require(build.supervision.closed(command) and command['exit_code'] == 0, 'model process failed')
        outputs = list((root / 'bench').glob('*/results.json'))
        require(len(outputs) == 1, 'missing/duplicate benchmark result')
        result = json.loads(outputs[0].read_text())
        cohort = result['results']['prefill_only']
        samples, counts = cohort['requests'], cohort['count_preflight']
        require(result['complete'] and not result['results']['failures'] and len(samples) == len(counts) == RUNS, 'incomplete benchmark')
        hashes = [s['request_sha256'] for s in samples]
        require(len(set(hashes)) == RUNS and all(re.fullmatch('[0-9a-f]{64}', s) for s in hashes)
                and hashes == row['request_hashes'].get(str(plan['read']), hashes), 'request bodies differ')
        for sample, count in zip(samples, counts):
            require(all(sample[k] is True for k in ('ok', 'eligible', 'stream_complete', 'thinking_requested', 'reasoning_observed'))
                    and sample['actual_read_tokens'] == sample['actual_prompt_tokens'] == count['counted_input_tokens'] == plan['read']
                    and sample['actual_reused_tokens'] == 0 and sample['request_sha256'] == count['body_sha256']
                    and 1 <= sample['generated_tokens'] <= 24 and sample['prompt_tps_source'] == 'engine'
                    and math.isfinite(sample['prompt_ms']) and sample['prompt_ms'] > 10000, 'sample/workload/clock differs')
        require(shared.client.verify_request_journal(outputs[0].parent)['valid'], 'request journal invalid')
        current(row)
        config = json.loads((root / 'config.json').read_text())
        end = shared.gen.native_end(root, row, dict(config=config), row['boot'], command['pid'])
        text = (root / 'engine.stderr').read_text()
        validation = shared.runtime.validation.preload_evidence(text, end['native']['pid'])
        require(validation['loader'] == row['native_validation_probe']['loader']['path'] and not any(s in text for s in
            ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called', 'teardown uncertain', 'strata trace:',
             'strata MoE progress:', 'strata MoE mixed owners:', 'strata prefill dense:', 'strata decode timing:')), 'runtime/observer differs')
        fields = ('actual_read_tokens', 'actual_reused_tokens', 'prompt_ms', 'prompt_tps', 'ttft_s', 'total_s',
            'generated_tokens', 'request_sha256', 'drafts_offered', 'drafts_accepted', 'timing_frontend_state')
        outcome.update(complete=True, host_probe_accepted=True, native_exit_code=0, validation=validation,
            routes=routes(text, plan['read']), samples=[{k: s[k] for k in fields} for s in samples],
            subsequent_median_ms=statistics.median(s['prompt_ms'] for s in samples[1:]))
        row['request_hashes'][str(plan['read'])] = hashes
        if index == len(ORDER) - 1:
            shared.models.model_pins(config, row['model_assets'], full=True)
    except BaseException as error:
        outcome.update(error=repr(error), complete=False, host_probe_accepted=False)
    finally:
        if owner:
            shared.runtime.validation.finish(outcome, owner, cursor, row['boot'])
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete'])
        persist()
        row['processes'].append(outcome)
        row.update(stage='running' if outcome['passed'] else 'failed', active=bool(owner and owner.active),
            ownership_closed=outcome.get('ownership_closed', False))
        if outcome['passed'] and len(row['processes']) == len(ORDER):
            row.update(stage='closed', passed=True, complete=True, decision=decision(row['processes']))
        save(row)
    print(json.dumps({k: outcome.get(k) for k in ('index', 'read', 'topk', 'passed', 'complete', 'fresh_faults', 'error')}), flush=True)
    require(outcome['passed'], 'official model process did not qualify')


if __name__ == '__main__':
    require(len(sys.argv) == 2 and (sys.argv[1] in ('prepare', 'host') or sys.argv[1].isdigit()), 'use prepare|host|index')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.guards.no_model_owners([])
        {'prepare': prepare, 'host': host}.get(sys.argv[1], lambda: run(int(sys.argv[1])))()
