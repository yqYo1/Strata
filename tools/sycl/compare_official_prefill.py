#!/usr/bin/env python3
"""Matched joint-head capacity strategy comparison; repository LICENSE.

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
ROOT = B / 'official-joint-speed-v1'
KEY, READS, RUNS = 'official_joint_speed_v1', (117489, 65537), 4
ARMS = (1, 0, 0, 1, 1, 0)
ORDER = [(read, arm) for pair in range(3) for read in READS for arm in ARMS[2*pair:2*pair+2]]
require, digest, write = entry.require, entry.digest, entry.write
TARGET_BYTES, DRAFT_BYTES, MAP_BYTES = 521472000, 223227900, 425196


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
    env['STRATA_SYCL_PREFILL_TOPK_EARLY'] = '0'
    env['STRATA_PREFILL_RELEASE_HEADS'] = str(plan['heads'])
    require(not env.get('LD_PRELOAD') and all(v == '0' for k, v in env.items()
            if '_PROFILE' in k or '_AUDIT' in k), 'clean observers required')
    require(env['EnableDirectSubmission'] == '0' and env['UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD'] == '1'
            and env['STRATA_SYCL_PREFILL_SAME_QUEUE'] == '1', 'runtime safety settings differ')
    return env


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require_quiet_validation(root):
    path = root / 'validation.log'
    require(not path.exists() or path.stat().st_size == 0, 'validation warning/error')


def require_valid_client_ttft(sample):
    first, total = sample['ttft_s'], sample['total_s']
    require(type(first) in (int, float) and type(total) in (int, float)
            and math.isfinite(first) and math.isfinite(total) and 0 < first <= total,
            'invalid individual client TTFT or duration')


def prepare():
    controls = selfcheck()
    built, _, files = entry.proof()
    data = build.committed(RESULT)
    prior = data[entry.KEY]
    require(prior['passed'] and prior['complete'] and prior['ownership_closed'] and not prior['active']
            and prior['binary'] == built['binaries']['strata'] and not prior['outcome']['fresh_faults'],
            'closed same-build model-entry qualification required')
    require(len(prior['outcome']['routes']) == 2 and all(
        r['target_snapshot_bytes'] == TARGET_BYTES and r['draft_rebuilt_bytes'] == DRAFT_BYTES
        and r['restored_dense_layers'] == 48 for r in prior['outcome']['routes']),
        'two actual same-build joint retirement/restore progressions required')
    require(not ROOT.exists() and KEY not in data, 'fresh comparison only')
    contract, manifest = entry.python_manifest()
    config = copy.deepcopy(prior['configuration'])
    expected = json.loads(shared.gen.git('show', '1e676d554:' + str(shared.TEMPLATE.relative_to(P))))['model_assets']
    assets = shared.models.model_pins(config, expected, full=True)
    revision = shared.gen.git('rev-parse', 'HEAD').decode().strip()
    sources = shared.pins(revision)
    qualified = json.loads((entry.ROOT / 'record.json').read_text())
    require(entry.compact(qualified) == prior and sources.keys() == qualified['sources'].keys(),
            'original model qualification/source inventory differs')
    changed = sorted(k for k in sources if sources[k] != qualified['sources'][k])
    require(changed == ['tools/sycl/compare_official_prefill.py'],
            'only the unimported comparison controller may differ from the model qualification')
    ROOT.mkdir(mode=0o700)
    write(ROOT / 'runtime-inputs.json', files)
    row = dict(stage='prepared', source_commit=revision, sources=sources, model_assets=assets,
        native_source_commit=built['source_commit'], binary=built['binaries']['strata'],
        native_validation_probe=prior['native_validation_probe'], configuration=config,
        qualification_reuse=dict(model_source=prior['source_commit'], changed_controller=changed,
            scope='Same seven native executables and model/benchmark execution inputs;new comparison controller only. '
                'No inherited clean speed,sustained decode,head byte/graph-binding or physical-context qualification.'),
        environment_template=prior['environment'], python_contract=contract, python_selected=manifest['selected'],
        template=prior['template'], boot=built['boot'], driver=built['driver'],
        runtime_manifest=dict(path=str(ROOT / 'runtime-inputs.json'), sha256=digest(ROOT / 'runtime-inputs.json'), files=len(files)),
        plans=[], processes=[], request_hashes={}, controller_checks=controls, host_passed=False, complete=False, passed=False,
        active=False, ownership_closed=True, inference_adopted=False,
        budgets=dict(wall=1800, cpu=5400, rss_cap=104 << 30, text_cap=16 << 20, file_cap=16 << 20, total_cap=64 << 20),
        plan=dict(question='Does releasing decode-only target/draft heads improve total long-prefill time at64K and117K?',
            mechanism='The prior117K capacity miss is234895744B including512MiB headroom. The same-build32K '
                'joint progression retires521472000B target and223227900B draft-head storage. Test whether '
                'this admits all-GPU residuals and split accumulation at117K,including snapshot,retirement,'
                'restore and subset reconstruction cost. No PCIe/VRAM/arithmetic saturation is established.',
            constants='Same native ELF,official2026.2 providers,BF16/vLLM,split policy enabled,original Down,'
                'eight-job waits,8192chunks,1256decode slots,5workers,spec4,INT8 resident131072context. '
                'Top-k fixed0;only head retirement0/1 differs before native exec;no runtime environment mutation.',
            samples='12fresh processes:3adjacent opposite-order pairs per length,interleaved by length. '
                'Head arms1/0,0/1,1/0;117K then64K within each pair. '
                'Each has4exact-count thinking-on fresh requests;first long use separate,median of subsequent3 '
                'is the process unit. Warmup chat and count-only tokenizer preflight are outside prompt timing. '
                'This is not cold-disk TTFT. Keep all individual prompt times,PP and client TTFT.',
            acceptance='Finish all12processes/48requests. Per length,all3paired on/off warmed-time ratios<1 '
                'and geometric mean<=0.99 permit further head-strategy qualification at that length only. '
                'Separately require all candidate requests to admit full-GPU residuals and117K split. Report TTFT and first use '
                'separately. Target1000 requires all3on warmed process medians>=1000 at a length>=64K. '
                'Neither screen adopts the engine or proves decode preservation.',
            stopping='Stop first input,owner,clock,warning,fault,budget,route or sample failure. No speed-based '
                'stop,rerun,replacement,exclusion or post-hoc threshold changes. Commit every closed process '
                'before the next. Full model-asset hashes before/after cohort;identities throughout.',
            routes='Per-request byte windows exclude startup/warmup. All arms conserve HC4 BF16 residual '
                'bytes and GPU/host rows,48GPU windows,job/Down counts and unchanged decode placement. '
                '64K:all65536GPU rows,staged384slots,ordinary accumulator. Candidate117K:all117488GPU rows,'
                'staged128slots,65536primary+51952tail and checked guard/drain. Baseline117K may use that '
                'same admitted split or its legitimate ordinary512-slot partially GPU-resident fallback; '
                'the latter has no staged-bank/split-completion line. Keep its actual capacity/placement '
                'rather than requiring the candidate route. This is a new capacity-strategy question,'
                'not a retry or relaxation of failed Top-k V1. Head detach bytes/map must match the '
                'qualified model entry on and be absent off. No bytewise payload/graph-binding correctness '
                'or per-kernel/physical-resource saturation claim.',
            limits='Length decisions are independent;do not generalize a117K capacity gain to64K. '
                'Two lengths supply local observations,not a validated global scaling law. Short24token '
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
        plan = dict(index=index, read=read, heads=arm, root=str(root),
            generated={n: digest(root / n) for n in ('config.json', 'plan.json')},
            command=['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(port),
                '--prefill-exact-tokens', str(read), '--think', '--runs', str(RUNS), '--profile-request-gates',
                '--workload-id', 'native_gdn_20261018', '--fail-fast-engine-error', '--ready-timeout', '600', '--out', str(root / 'bench')])
        plan['environment_sha256'] = fingerprint(environment(row, plan))
        row['plans'].append(plan)
    save(row)
    print('Prepared12matched head-capacity processes;commit before host checks.', flush=True)


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


def host(experiment=None):
    """Reuse frontend checks with an explicit campaign policy, never global mutation."""
    policy = experiment or sys.modules[__name__]
    row = json.loads((policy.ROOT / 'record.json').read_text())
    _, manifest = policy.current(row)
    require(row['stage'] == 'prepared' and not (policy.ROOT / 'host').exists(), 'fresh host probes only')
    owner = build.supervision.Owner(policy.ROOT / 'host')
    env = policy.environment(row, row['plans'][0])
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
        policy.save(row)
    require(row['host_passed'] and row['ownership_closed'], row.get('error', 'host check failed'))
    print('Current-source frontend probes passed;commit before native execution.', flush=True)


def route(text, read, heads):
    """Validate one actual request window,including a capacity-limited control."""
    require(read in READS and heads in (0, 1), 'unplanned route case')
    def records(prefix):
        result = []
        for line in text.splitlines():
            if not line.startswith(prefix):
                continue
            pairs = re.findall(r'([A-Za-z_][A-Za-z_0-9-]*)=([^\s;]+)', line.split(';', 1)[0])
            require(len({k for k, _ in pairs}) == len(pairs), 'duplicate route field')
            result.append(dict(pairs))
        return result
    def one(prefix):
        result = records(prefix)
        require(len(result) == 1, 'missing/duplicate route record: ' + prefix)
        return result[0]
    main = one('strata prefill: layer-major candidate, ')
    bank = one('strata MoE expert bank: ')
    window = one('strata MoE window configuration: ')
    work = one('strata MoE accumulator window: ')
    admission = one('strata MoE split admission: ')
    n, gpu = read - 1, int(main['residual_gpu_rows'])
    split, staged, slots = int(admission['active']), int(bank['staged']), int(bank['effective_slots'])
    require(split in (0, 1) and staged in (0, 1) and 0 < gpu <= n, 'invalid route shape')
    require(main['tokens'] == str(n) and main['chunk'] == '8192' and main['residual_dtype'] == 'bf16'
            and main['residual_element_bytes'] == '2' and int(main['residual_total']) == n * 10240 * 2
            and int(main['residual_gpu']) == gpu * 10240 * 2
            and int(main['residual_host']) == (n - gpu) * 10240 * 2
            and main['residual_staging'] == '0', 'HC4 residual byte/row conservation differs')
    require(bank['requested_slots'] == '384' and int(bank['residual_gpu_rows']) == gpu
            and int(bank['device_bytes']) == int(main['layer_cache']) > 0
            and int(bank['allocation_limit_bytes']) >= int(bank['device_bytes'])
            and bank['general_map'] == 'private_scratch_never_published'
            and admission['rows'] == str(n) and admission['prefill_slots'] == str(slots)
            and admission['decode_changed'] == '0', 'bank/cache publication differs')
    full = gpu == n
    if read == 65537:
        require(full and staged == 1 and slots == 384 and split == 0, '64K placement differs')
    elif full:
        require(staged == split == 1 and slots == 128, '117K all-GPU split placement differs')
    else:
        require(heads == 0 and staged == split == 0 and slots == 512
                and gpu % 8192 == 0 and gpu <= 81920, 'unqualified partial-residency route')
    require(not heads or full, 'candidate capacity screen failed')
    require(window['requested_mode'] == window['effective_mode'] == '10'
            and int(window['maximum_rows']) == int(work['maximum_rows']) == gpu
            and work['windows'] == '48' and work['operator_max'] == '8192'
            and work['GPU_resident_windows_only'] == '1' and int(work['jobs']) >= 48
            and int(work['dequants']) > 0 and int(work['source_H2D_bytes']) >= 0,
            'actual accumulator/window work differs')
    bank_ends = records('strata MoE expert bank complete: ')
    require(len(bank_ends) == staged, 'staged bank completion differs')
    if staged:
        end = bank_ends[0]
        require(end['final_drain'] == 'checked' and int(end['total_slots']) == slots
                and 0 <= int(end['permanent_slots']) <= slots and int(end['uploads']) >= 0
                and int(end['requested_H2D_bytes']) >= 0, 'bank upload/drain differs')
    tails = records('strata MoE split complete: ')
    require(len(tails) == split, 'split completion differs')
    if split:
        tail, finish = (n - 65536) * 2560 * 4, tails[0]
        require(admission['primary_rows'] == finish['primary_rows'] == '65536'
                and int(admission['tail_payload_bytes']) == int(finish['tail_bytes']) == tail
                and int(admission['extra_allocation_bytes']) == (tail + 255) // 256 * 256 + 256
                and int(admission['max_single_allocation_bytes']) >= int(admission['extra_allocation_bytes'])
                and finish['rows'] == str(n) and finish['decode_changed'] == '0'
                and finish['guard'] == finish['final_drain'] == 'checked', 'split bounds/guard/drain differs')
    else:
        require(admission['primary_rows'] == admission['tail_payload_bytes'] ==
                admission['extra_allocation_bytes'] == '0', 'inactive split carries storage')
    down = one('strata MoE Down fusion: ')
    require(int(down['fused_jobs']) > 0 and down['min_rows'] == '2334'
            and int(down['fused_rows']) >= 2334 * int(down['fused_jobs'])
            and int(down['fused_jobs']) + int(down['separate_jobs']) == int(work['jobs']), 'Down work conservation differs')
    targets, drafts = records('strata prefill heads: target detached '), records('strata prefill heads: draft detached ')
    require(len(targets) == len(drafts) == heads, 'head retirement dispatch differs')
    if heads:
        target, draft = targets[0], drafts[0]
        require(target['source'] == 'owned-RAM' and int(target['bytes']) == int(target['D2H']) ==
                int(target['H2D-on-restore']) == TARGET_BYTES and int(draft['bytes']) == DRAFT_BYTES
                and int(draft['retained-map-bytes']) == MAP_BYTES, 'head snapshot/subset contract differs')
        require(text.index('strata prefill heads: target detached ') < text.index('strata prefill heads: draft detached ')
                < text.index('strata MoE expert bank: ') < text.index('strata prefill: layer-major candidate, '),
                'head retirement/admission order differs')
    return dict(rows=n, residual_gpu_rows=gpu, residual_gpu_bytes=int(main['residual_gpu']),
        residual_host_bytes=int(main['residual_host']), prefill_slots=slots, staged=bool(staged), split=bool(split),
        windows=48, jobs=int(work['jobs']), fused_down_jobs=int(down['fused_jobs']), maximum_window_rows=gpu,
        staged_bank_checked_drain=bool(staged), split_guard_checked=bool(split),
        requested_bank_H2D_bytes=int(bank_ends[0]['requested_H2D_bytes']) if staged else 0,
        requested_dequant_H2D_bytes=int(work['source_H2D_bytes']),
        target_snapshot_bytes=TARGET_BYTES if heads else 0, draft_rebuilt_bytes=DRAFT_BYTES if heads else 0,
        physical_saturation_qualified=False, head_payload_bytes_verified=False)


def decision(processes):
    require(len(processes) == len(ORDER) and all(p['passed'] and p['complete'] and p['ownership_closed']
            and not p['fresh_faults'] for p in processes), 'complete cohort required')
    require(len({(p['native_identity']['pid'], p['native_identity']['start_ticks']) for p in processes}) == len(ORDER), 'native process reused')
    require(all(math.isfinite(p['subsequent_median_ms']) and p['subsequent_median_ms'] > 10000 for p in processes), 'invalid process median')
    result = {}
    for read in READS:
        selected = [p for p in processes if p['read'] == read]
        require([p['heads'] for p in selected] == list(ARMS), 'comparison order differs')
        ratios = []
        for i in (0, 2, 4):
            pair = {p['heads']: p['subsequent_median_ms'] for p in selected[i:i+2]}
            ratios.append(pair[1] / pair[0])
        gm = math.exp(statistics.mean(math.log(v) for v in ratios))
        result[str(read)] = dict(paired_time_ratios=ratios, geometric_mean_time_ratio=gm,
            mechanism_screen_passed=all(v < 1 for v in ratios) and gm <= .99,
            candidate_capacity_screen_passed=all(len(p['routes']) == RUNS and all(
                r['residual_gpu_rows'] == read - 1 and r['residual_host_bytes'] == 0
                and r['split'] == (read == 117489) for r in p['routes']) for p in selected if p['heads'] == 1),
            all_three_on_warm_medians_ge1000=all(read * 1000 / p['subsequent_median_ms'] >= 1000 for p in selected if p['heads'] == 1))
    return dict(lengths=result, inference_adopted=False, sustained_decode_qualified=False, physical262144_qualified=False)


def selfcheck():
    """Check residual conservation,legitimate controls and asymmetric eligibility."""
    def fixture(read, heads, gpu=None):
        n = read - 1
        gpu = n if gpu is None else gpu
        staged, split = int(gpu == n), int(read == 117489 and gpu == n)
        slots = (128 if split else 384) if staged else 512
        tail = (n - 65536) * 2560 * 4 if split else 0
        parts = [f'strata MoE split admission: active={split} rows={n} primary_rows={65536 if split else 0} '
                f'tail_payload_bytes={tail} extra_allocation_bytes={tail+256 if split else 0} '
                f'prefill_slots={slots} max_single_allocation_bytes={4<<30} decode_changed=0',
            f'strata MoE expert bank: requested_slots=384 effective_slots={slots} staged={staged} '
                f'host_slab_bytes=1363148800 device_bytes=1363148800 residual_gpu_rows={gpu} '
                f'allocation_limit_bytes={4<<30} general_map=private_scratch_never_published',
            f'strata prefill: layer-major candidate, tokens={n} residual_gpu_rows={gpu} residual_dtype=bf16 '
                f'chunk=8192 residual_element_bytes=2 residual_total={n*20480} residual_gpu={gpu*20480} '
                f'residual_host={(n-gpu)*20480} residual_staging=0 layer_cache=1363148800; '
                'extra allocations, GPU headroom=536870912, RAM headroom=8589934592',
            f'strata MoE window configuration: requested_mode=10 effective_mode=10 maximum_rows={gpu}',
            f'strata MoE accumulator window: maximum_rows={gpu} windows=48 jobs=48 dequants=48 '
                'source_H2D_bytes=4096 GPU_resident_windows_only=1 operator_max=8192',
            'strata MoE Down fusion: fused_jobs=2 fused_rows=4668 separate_jobs=46 min_rows=2334']
        if heads:
            parts[:0] = [f'strata prefill heads: target detached bytes={TARGET_BYTES} source=owned-RAM '
                f'D2H={TARGET_BYTES} H2D-on-restore={TARGET_BYTES}',
                f'strata prefill heads: draft detached bytes={DRAFT_BYTES} retained-map-bytes={MAP_BYTES}']
        if staged:
            parts.append(f'strata MoE expert bank complete: uploads=2 requested_H2D_bytes=4096 '
                f'permanent_hits=1 permanent_slots=1 total_slots={slots} final_drain=checked')
        if split:
            parts.append(f'strata MoE split complete: rows={n} primary_rows=65536 tail_bytes={tail} '
                'guard=checked final_drain=checked decode_changed=0')
        return '\n'.join(parts)
    cases = [(read, heads, fixture(read, heads)) for read in READS for heads in (0, 1)]
    cases += [(117489, 0, fixture(117489, 0, gpu)) for gpu in (81920, 73728)]
    for read, heads, text in cases:
        route(text, read, heads)
    full, short, partial = fixture(117489, 1), fixture(65537, 1), fixture(117489, 0, 73728)
    bad = [(full + '\nstrata MoE accumulator window: jobs=48', 117489, 1),
        (full.replace('strata MoE expert bank:', 'missing bank:'), 117489, 1),
        (short.replace('effective_slots=384', 'effective_slots=128'), 65537, 1),
        (short.replace('residual_gpu_rows=65536', 'residual_gpu_rows=0'), 65537, 1),
        (full.replace('guard=checked', 'guard=failed'), 117489, 1),
        (full.replace('separate_jobs=46', 'separate_jobs=47'), 117489, 1),
        (full.replace('decode_changed=0', 'decode_changed=1'), 117489, 1),
        (full.replace('residual_host=0', 'residual_host=1'), 117489, 1),
        (full.replace('residual_element_bytes=2', 'residual_element_bytes=4'), 117489, 1),
        (full.replace('residual_total=2406154240', 'residual_total=2406154241'), 117489, 1),
        (full.replace('D2H=521472000', 'D2H=521472001'), 117489, 1),
        (full.replace('retained-map-bytes=425196', 'retained-map-bytes=0'), 117489, 1),
        (full.replace('strata prefill heads: draft detached', 'missing draft'), 117489, 1),
        (full.replace('source=owned-RAM', 'bytes=1 source=owned-RAM'), 117489, 1),
        (full.replace('strata MoE expert bank complete:', 'missing completion:'), 117489, 1),
        (partial + '\nstrata MoE expert bank complete: final_drain=checked', 117489, 0),
        (partial.replace('residual_gpu=1509949440', 'residual_gpu=1509949441'), 117489, 0),
        ('\n'.join(full.splitlines()[:2]) + '\n' + partial, 117489, 1)]
    for text, read, heads in bad:
        try:
            route(text, read, heads)
        except (RuntimeError, ValueError, KeyError):
            continue
        raise RuntimeError('invalid route accepted')
    processes = [dict(read=read, heads=arm, passed=True, complete=True, ownership_closed=True, fresh_faults=[],
        native_identity=dict(pid=i+1, start_ticks=i+100), subsequent_median_ms=20000 if arm == 0 else 19600,
        routes=[route(fixture(read, arm), read, arm) for _ in range(RUNS)])
        for i, (read, arm) in enumerate(ORDER)]
    require(all(v['mechanism_screen_passed'] for v in decision(processes)['lengths'].values()), 'passing decision rejected')
    processes[0]['subsequent_median_ms'] = 21000
    mixed = decision(processes)['lengths']
    require(not mixed[str(READS[0])]['mechanism_screen_passed'] and mixed[str(READS[1])]['mechanism_screen_passed'],
        'one slower pair accepted or length decisions coupled')
    duplicate = copy.deepcopy(processes)
    duplicate[1]['native_identity'] = duplicate[0]['native_identity']
    try:
        decision(duplicate)
    except (RuntimeError, ValueError, KeyError):
        return dict(passed=True, valid_route_shapes=len(cases), invalid_routes_rejected=len(bad), comparison_controls=3)
    raise RuntimeError('reused native identity accepted')


def run(index, experiment=None):
    """Owned clean model case; campaign supplies admission, selection and decision.

    The original head comparison remains the default. A separate campaign may
    reuse the same request/clock/owner/provider/fault checks without replacing
    this module's globals or mutating the native process environment.
    """
    policy = experiment or sys.modules[__name__]
    row = json.loads((policy.ROOT / 'record.json').read_text())
    files, manifest = policy.current(row)
    require(row['host_passed'] and row['stage'] in ('ready', 'running') and len(row['processes']) == index < len(policy.ORDER)
            and all(p['passed'] for p in row['processes']), 'fresh ordered process only;no retry')
    plan = row['plans'][index]
    native = policy.native_context(row, plan) if hasattr(policy, 'native_context') else row
    require(native['native_validation_probe'] == row['native_validation_probe'] and
            native['boot'] == row['boot'], 'campaign native context changed runtime/boot')
    root, env = Path(plan['root']), policy.environment(row, plan)
    require(not (root / 'execution.json').exists() and (plan['read'], plan['heads']) == policy.ORDER[index]
            and fingerprint(env) == plan['environment_sha256'] and
            all(digest(root / n) == sha for n, sha in plan['generated'].items()), 'process plan changed')
    cursor = next(v.split(': ', 1)[1] for v in subprocess.check_output(
        ['journalctl', '-k', '-b', '-n', '0', '--show-cursor', '--no-pager'], text=True, timeout=15).splitlines() if v.startswith('-- cursor: '))
    outcome, owner = dict(index=index, read=plan['read'], heads=plan['heads'], complete=False,
        host_probe_accepted=False, model_executed=False, clean_speed=True), None
    if 'kernel' in plan:
        outcome['kernel'] = plan['kernel']
    if hasattr(policy, 'native_context'):
        outcome.update(binary=native['binary'], native_source_commit=native['native_source_commit'])
    initial = (time.clock_gettime_ns(time.CLOCK_BOOTTIME), time.monotonic_ns())
    log_windows = []
    def persist():
        write(root / 'execution.json', dict(outcome, active=bool(owner and owner.active), commands=owner.commands if owner else []))
    def interact(io, command):
        identity = None
        for request in range(policy.RUNS):
            offsets = {}
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
                        and str(Path('/proc', str(claim['pid']), 'exe').resolve()) == native['binary']['path']
                        and (identity is None or identity == claim), 'owned native identity differs')
                identity = claim
                expected = engine_once.native_environment(native, env)
                observed = dict(v.split(b'=', 1) for v in Path('/proc', str(claim['pid']), 'environ').read_bytes().split(b'\0') if b'=' in v)
                require(all(observed.get(k.encode()) == v.encode() for k, v in expected.items())
                        and set(observed) - {k.encode() for k in expected} <= {b'PWD', b'OLDPWD', b'SHLVL', b'_'}, 'native environment differs')
                mapped = shared.runtime.executable_mappings(Path('/proc', str(claim['pid']), 'maps').read_text())
                require(mapped <= set(files) | {native['binary']['path']} and {native['native_validation_probe']['loader']['path'],
                    native['native_validation_probe']['library']['path'], env['ZE_ENABLE_ALT_DRIVERS']} <= mapped, 'native providers differ')
                shared.python_runtime.check_running(command['pid'], P / 'tools/strata_bench.py', manifest['selected'], env,
                    {v['realpath'] for v in manifest['inventory']['files'].values()}, plan['command'][2:])
                require_quiet_validation(root)
                outcome.update(model_executed=True, native_identity=claim)
                offsets[stage] = (root / 'engine.stderr').stat().st_size
                print(f'Prefill process{index} read{plan["read"]} heads{plan["heads"]} '
                      f'kernel={plan.get("kernel", "original")} request{request} {stage}', flush=True)
                io.send(b'continue\n')
            require(offsets['end'] > offsets['begin'], 'missing native request log window')
            log_windows.append(offsets)
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
        # The final request gate precedes native/runtime destruction. Only an
        # empty owned session proves that all validation-log producers exited.
        require_quiet_validation(root)
        outputs = list((root / 'bench').glob('*/results.json'))
        require(len(outputs) == 1, 'missing/duplicate benchmark result')
        result = json.loads(outputs[0].read_text())
        cohort = result['results']['prefill_only']
        samples, counts = cohort['requests'], cohort['count_preflight']
        require(result['complete'] and not result['results']['failures'] and
                len(samples) == len(counts) == len(log_windows) == policy.RUNS, 'incomplete benchmark')
        hashes = [s['request_sha256'] for s in samples]
        require(len(set(hashes)) == policy.RUNS and all(re.fullmatch('[0-9a-f]{64}', s) for s in hashes)
                and hashes == row['request_hashes'].get(str(plan['read']), hashes), 'request bodies differ')
        for sample, count in zip(samples, counts):
            require_valid_client_ttft(sample)
            require(all(sample[k] is True for k in ('ok', 'eligible', 'stream_complete', 'thinking_requested', 'reasoning_observed'))
                    and sample['actual_read_tokens'] == sample['actual_prompt_tokens'] == count['counted_input_tokens'] == plan['read']
                    and sample['actual_reused_tokens'] == 0 and sample['request_sha256'] == count['body_sha256']
                    and 1 <= sample['generated_tokens'] <= 24 and sample['prompt_tps_source'] == 'engine'
                    and math.isfinite(sample['prompt_ms']) and sample['prompt_ms'] > 10000, 'sample/workload/clock differs')
        require(shared.client.verify_request_journal(outputs[0].parent)['valid'], 'request journal invalid')
        policy.current(row)
        config = json.loads((root / 'config.json').read_text())
        end = shared.gen.native_end(root, native, dict(config=config), row['boot'], command['pid'])
        raw_text = (root / 'engine.stderr').read_bytes()
        text = raw_text.decode()
        validation = shared.runtime.validation.preload_evidence(text, end['native']['pid'])
        require(validation['loader'] == row['native_validation_probe']['loader']['path'] and not any(s in text for s in
            ('UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_', 'terminate called', 'teardown uncertain', 'strata trace:',
             'strata MoE progress:', 'strata MoE mixed owners:', 'strata prefill dense:', 'strata decode timing:')), 'runtime/observer differs')
        if hasattr(policy, 'validate_model_text'):
            policy.validate_model_text(text, plan)
        fields = ('actual_read_tokens', 'actual_reused_tokens', 'prompt_ms', 'prompt_tps', 'ttft_s', 'total_s',
            'generated_tokens', 'request_sha256', 'drafts_offered', 'drafts_accepted', 'timing_frontend_state')
        routes = [policy.request_route(raw_text[w['begin']:w['end']].decode(), plan)
                  if hasattr(policy, 'request_route') else
                  policy.route(raw_text[w['begin']:w['end']].decode(), plan['read'], plan['heads'])
                  for w in log_windows]
        outcome.update(complete=True, host_probe_accepted=True, native_exit_code=0, validation=validation,
            routes=routes,
            samples=[{k: s[k] for k in fields} for s in samples],
            subsequent_median_ms=statistics.median(s['prompt_ms'] for s in samples[1:]))
        row['request_hashes'][str(plan['read'])] = hashes
        if index == len(policy.ORDER) - 1:
            shared.models.model_pins(config, row['model_assets'], full=True)
    except BaseException as error:
        outcome.update(error=repr(error), complete=False, host_probe_accepted=False)
    finally:
        if owner:
            shared.runtime.validation.finish(outcome, owner, cursor, row['boot'])
        if owner and outcome.get('driver_after') != row['driver']:
            outcome['passed'] = False
            outcome.setdefault('error', 'Post-run driver differs from the frozen platform')
        outcome['passed'] = bool(outcome.get('passed') and outcome['complete'])
        persist()
        row['processes'].append(outcome)
        row.update(stage='running' if outcome['passed'] else 'failed', active=bool(owner and owner.active),
            ownership_closed=outcome.get('ownership_closed', False))
        if outcome['passed'] and len(row['processes']) == len(policy.ORDER):
            row.update(stage='closed', passed=True, complete=True, decision=policy.decision(row['processes']))
        policy.save(row)
    print(json.dumps({k: outcome.get(k) for k in ('index', 'read', 'heads', 'passed', 'complete', 'fresh_faults', 'error')}), flush=True)
    require(outcome['passed'], 'official model process did not qualify')


if __name__ == '__main__':
    require(len(sys.argv) == 2 and (sys.argv[1] in ('prepare', 'host') or sys.argv[1].isdigit()), 'use prepare|host|index')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build.guards.no_model_owners([])
        {'prepare': prepare, 'host': host}.get(sys.argv[1], lambda: run(int(sys.argv[1])))()
