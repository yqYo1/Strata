#!/usr/bin/env python3
"""Matched ordinary-Arena decode work distribution on the coherent engine.

Main-authored from this fork's owned benchmark controls; repository LICENSE.
No native rebuild, math change, runtime environment mutation or XeStrata input.
"""
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import socket
import sys

sys.dont_write_bytecode = True
import coherent_native_reuse as reuse
import coherent_decode_transfer as admission
import coherent_prefill_bench as prefill
import pool_spin as runner

P, B, RESULT = reuse.P, reuse.B, reuse.RESULT
ROOT, KEY = B / 'coherent-decode-pcie-speed-v2', 'coherent_decode_pcie_speed_v2'
THIS = 'tools/sycl/coherent_decode_bench.py'
PYTHON = prefill.model.PYTHON
ARMS = ('reference', 'candidate', 'candidate', 'reference', 'reference', 'candidate')
WORKLOADS = ('regular-code', 'varied-prose')
WORKLOAD_ID = 'coherent_decode_pcie_20261112'
require, sha, signature = reuse.require, reuse.sha, reuse.fingerprint
core, shared, build = prefill.core, runner.shared, runner.build
INDEX = None


def compact(row):
    return core.compact(row)


def save(row):
    core.write(ROOT / 'record.json', row)
    data = json.loads(RESULT.read_text()); data[KEY] = compact(row); core.write(RESULT, data)


def basis(expected=None):
    native, proof = reuse.verify(observers={THIS, admission.THIS}, expected=expected)
    prior = json.loads((B / 'coherent-decode-transfer-long-v2/record.json').read_text())
    require(reuse.build.validation.committed(RESULT, ('coherent_decode_transfer_long_v2',))
        == admission.original.original.previous.compact(prior)
        and prior['stage'] == 'closed' and all(prior[k] for k in
            ('passed', 'complete', 'ownership_closed', 'model_executed')) and not prior['active']
        and prior['native_build']['binaries'] == native['binaries']
        and prior['native_source_commit'] == native['native_source_commit']
        and sha(P / admission.THIS) == prior['native_observer_admission']['declared_observers'][admission.THIS],
        'closed corrected long-copy admission and its original controller required')
    outcome, = prior['results']
    require(outcome['native_exit_code'] == 0 and outcome['ownership_closed']
        and not outcome['fresh_faults'] and outcome['routes']['request_footer_pairing_checked']
        and [(x['prompt_tokens'], x['generated_tokens']) for x in outcome['routes']['copy_records']]
            == [(32769, 32), (82, 32), (32769, 32)], 'actual long/short/long binding required')
    speed = json.loads((prefill.ROOT / 'record.json').read_text())
    require(reuse.build.validation.committed(RESULT, (prefill.KEY,)) == prefill.core.compact(speed)
        and speed['stage'] == 'closed' and speed['passed'] and speed['complete']
        and speed['ownership_closed'] and not speed['active']
        and speed['binary'] == native['binaries']['strata']
        and speed['native_source_commit'] == native['native_source_commit'],
        'same coherent generation and clean prefill configuration required')
    providers = json.loads(Path(speed['runtime_manifest']['path']).read_text())
    require(sha(speed['runtime_manifest']['path']) == speed['runtime_manifest']['sha256']
        and all(sha(p) == h for p, h in providers.items())
        and build.guards.driver_guard() == speed['driver'] == prior['driver']
        and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == speed['boot'] == prior['boot'],
        'runtime, boot or driver changed')
    return native, proof, speed, providers


def configuration(row, plan):
    value = copy.deepcopy(row['configuration'])
    value['log'] = str(Path(plan['root']) / 'engine.stderr')
    args = value['args']
    require(args.count('--pcie-frac') == 1 and args[args.index('--pcie-frac')+1] == '0'
        and args[args.index('--adapt-swaps')+1] == '0', 'preserve static qualified configuration')
    args[args.index('--pcie-frac')+1] = '0.0625' if plan['kernel'] == 'candidate' else '0'
    return value


def environment(row, plan):
    env = prefill.context().environment(row, plan)
    env.update(STRATA_SYCL_VERIFY_ARENA_COPY='1', STRATA_SYCL_VERIFY_PCIE_EARLY='0',
        STRATA_SYCL_VERIFY_OVERLAP='1', STRATA_IQ3_PAIR_AUDIT='0',
        STRATA_SYCL_Q6_ROUTE_AUDIT='0', STRATA_CPU_SERVICE='0', STRATA_CPU_WORK_CENSUS='0')
    require(not any(k in env for k in runner.OBSERVERS)
        and all(v == '0' for k, v in env.items() if '_PROFILE' in k or '_AUDIT' in k)
        and env['STRATA_SYCL_PREFILL_MOE_CENSUS'] == '0'
        and env['STRATA_SYCL_DECODE_Q6_DPAS_HEAD'] == '2'
        and env['STRATA_SYCL_DECODE_Q6_DPAS_DENSE'] == '1', 'clean immutable phase settings required')
    return env


def invariant(row, plan):
    value = dict(config=configuration(row, plan), environment=environment(row, plan))
    args = value['config']['args']; args[args.index('--pcie-frac')+1] = '<arm>'
    return signature(json.dumps(value, sort_keys=True).replace(plan['root'], '<process>'))


def current(row):
    require(reuse.build.validation.committed(RESULT, (KEY,)) == compact(row)
        and core.pins(row['source_commit']) == row['sources'], 'commit exact comparison inputs/state')
    native, proof, prior, providers = basis(row['native_observer_admission'])
    require(row['binary'] == native['binaries']['strata']
        and row['native_source_commit'] == native['native_source_commit']
        and row['native_validation_probe'] == prior['native_validation_probe']
        and row['model_assets'] == prior['model_assets'] and row['configuration'] == prior['configuration']
        and row['environment_template'] == prior['environment_template'], 'native/configuration/assets differ')
    contract, manifest = runner.model.python_manifest()
    require(contract == row['python_contract'] and manifest['selected'] == row['python_selected']
        and sha(row['template']['path']) == row['template']['sha256']
        and sha(row['runtime_manifest']['path']) == row['runtime_manifest']['sha256']
        and json.loads(Path(row['runtime_manifest']['path']).read_text()) == providers,
        'frontend/provider inputs differ')
    require(build.supervision.artifact_bytes(ROOT) <= row['whole_bytes']
        and len(row['plans']) == len(ARMS), 'campaign size/order differs')
    for i, plan in enumerate(row['plans']):
        root = Path(plan['root'])
        require(plan['index'] == i and plan['kernel'] == ARMS[i] and plan['adapt_swaps'] == 0
            and plan['heads'] == 1 and invariant(row, plan) == row['invariant_signature']
            and json.loads((root / 'config.json').read_text()) == configuration(row, plan)
            and json.loads((root / 'plan.json').read_text()) == dict(binary=row['binary'],
                native_validation_probe=row['native_validation_probe'])
            and all(sha(root / n) == h for n, h in plan['generated'].items())
            and runner.comparison.fingerprint(environment(row, plan)) == plan['environment_sha256'],
            'actual generated arm/configuration/environment differs')
    shared.models.model_pins(row['configuration'], row['model_assets'], full=False)
    return providers, manifest


def copy_records(text, samples=None, *, arm):
    selected = [s for s in text.splitlines() if s.startswith(
        ('strata serve: prompt ', 'strata serve: SYCL_EXPERT_COPY '))]
    require(len(selected) > 0 and len(selected) % 2 == 0, 'complete ordered prompt/copy pairs required')
    require(samples is None or len(selected) == 2*len(samples), 'actual API request/copy coverage differs')
    rows = []
    for i in range(len(selected)//2):
        m = re.fullmatch(r'strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read '
            r'in (\d+) ms \([0-9.]+ tok/s\), (\d+) generated in (\d+) ms \([0-9.]+ tok/s\), '
            r'drafts accepted (\d+) of (\d+), (\d+) checkpoints', selected[2*i])
        require(m is not None, 'prompt footer must precede its copy footer')
        n, reused, read, pp_ms, generated, tg_ms, accepted, offered, checks = map(int, m.groups())
        require(n == read and reused == checks == 0 and generated > 0 and offered >= accepted >= 0,
            'actual fresh request/draft extent differs')
        if samples is not None:
            s = samples[i]
            require((n, reused, read, generated, accepted, offered) == tuple(s[k] for k in
                ('prompt_tokens', 'reused_tokens', 'read_tokens', 'generated_tokens', 'drafts_accepted', 'drafts_offered'))
                and abs(pp_ms-s['prompt_ms']) <= 1 and abs(tg_ms-s['decode_ms']) <= 1,
                'actual API counts/drafts/rounded clocks do not bind the copy footer')
        m = re.fullmatch(r'strata serve: SYCL_EXPERT_COPY submitted_blobs=(\d+) submitted_bytes=(\d+) '
            r'completed_blobs=(\d+) completed_bytes=(\d+) host_view_groups=(\d+) '
            r'gpu_miss_groups=(\d+) cpu_unique_misses=(\d+) cpu_entries=(\d+)', selected[2*i+1])
        require(m is not None, 'ordered copy footer missing')
        submitted, size, completed, retired, views, groups, cpu, entries = map(int, m.groups())
        require(submitted == completed == views == groups and size == retired
            and submitted*1510400 <= size <= submitted*2662400 and entries >= cpu > 0,
            'private host-view submissions/bytes/completions do not conserve')
        require((arm == 'reference' and submitted == size == 0)
            or (arm == 'candidate' and (submitted > 0 or read < 32768)), 'actual offload arm differs')
        rows.append(dict(read_tokens=read, generated_tokens=generated,
            submitted_blobs=submitted, completed_blobs=completed, host_view_groups=views,
            GPU_miss_groups=groups, requested_bytes=size, completed_bytes=retired,
            CPU_unique_misses=cpu, CPU_entries=entries))
    return rows


def clean_observers(text):
    runner.clean_observers(text)
    require(not any(x in text for x in ('SYCL_PCIE_EARLY', 'SYCL_EVENT_QUERY',
        'strata MoE planned job census:', 'strata MoE upload census:', 'SYCL_VERIFY_CALLER')),
        'disabled detailed observer appeared')


def copy_controls():
    sample = dict(prompt_tokens=32769, reused_tokens=0, read_tokens=32769,
        generated_tokens=1024, drafts_accepted=700, drafts_offered=1000,
        prompt_ms=40000.4, decode_ms=100000.4)
    prompt = ('strata serve: prompt 32769 tokens = 0 reused + 32769 read in 40000 ms '
        '(819.2 tok/s), 1024 generated in 100000 ms (10.2 tok/s), drafts accepted 700 of 1000, 0 checkpoints\n')
    copied = ('strata serve: SYCL_EXPERT_COPY submitted_blobs=4 submitted_bytes=6041600 '
        'completed_blobs=4 completed_bytes=6041600 host_view_groups=4 '
        'gpu_miss_groups=4 cpu_unique_misses=9 cpu_entries=10\n')
    zero = ('strata serve: SYCL_EXPERT_COPY submitted_blobs=0 submitted_bytes=0 '
        'completed_blobs=0 completed_bytes=0 host_view_groups=0 '
        'gpu_miss_groups=0 cpu_unique_misses=9 cpu_entries=10\n')
    require(copy_records(prompt+copied, [sample], arm='candidate')[0]['requested_bytes'] == 6041600
        and copy_records(prompt+zero, [sample], arm='reference')[0]['requested_bytes'] == 0,
        'literal positive/zero copy controls')
    cases = [(prompt+copied.replace('host_view_groups=4', 'host_view_groups=0'), [sample], 'candidate'),
        (prompt+copied.replace('completed_bytes=6041600', 'completed_bytes=6041601'), [sample], 'candidate'),
        (prompt+copied, [dict(sample, drafts_accepted=701)], 'candidate'),
        (prompt+copied, [dict(sample, prompt_ms=40002)], 'candidate'),
        (prompt+copied, [dict(sample, generated_tokens=1025)], 'candidate'),
        (copied+prompt, [sample], 'candidate'), (prompt+copied+copied, [sample], 'candidate'),
        (prompt+zero, [sample], 'candidate'), (prompt+copied, [sample], 'reference')]
    for text, samples, arm in cases:
        try: copy_records(text, samples, arm=arm)
        except RuntimeError: pass
        else: raise RuntimeError('unbound/nonconserving/wrong-arm copy stream accepted')
    return dict(passed=True, positive=2, rejected=len(cases), GPU_executed=False)


def route(text, read):
    require(32768 <= read <= 65537, 'unplanned ordinary long input')
    out = prefill.context().request_route(text, dict(read=read, heads=1))
    runner.generation.gdn.native16_route(text, read)
    runner.generation.gdn.gdn_selection(text, 'candidate')
    copies, = copy_records(text, arm=ARMS[INDEX])
    require(copies['read_tokens'] == read, 'long route/copy extent differs')
    out['decode_copy'] = copies
    return out


def official(root, result):
    suite = result['results']['long_generation']; rows = shared.gen.collect_samples(suite)
    runner_context().generation_contract(suite, rows)
    require(result['complete'] and not result['results']['failures']
        and result['params']['runs'] == 1 and result['params']['think'] is True
        and result['params']['long_generation_only'] is True, 'official benchmark incomplete')
    journal, actual = shared.gen.journal_evidence(root)
    require(journal['valid'] and journal['complete'] and
        [{k: v for k, v in r.items() if k not in ('controller_eligible', 'reasoning_tokens')} for r in rows] == actual,
        'official result/journal mismatch')
    path, = (root / 'bench').glob('*/requests.jsonl')
    events = [json.loads(s) for s in path.read_text().splitlines()]
    warm = shared.gen.warmup_reconciliation(events, WORKLOADS)
    identity, binding = events[0], result['comparison_identity']
    from strata_bench import client_fingerprint
    workload = dict(workload_id=WORKLOAD_ID, nonce=WORKLOAD_ID, nonce_policy='fixed-id')
    require(identity['mode'] == binding['mode'] == 'suite' and identity['requested_runs'] == 1
        and identity['thinking'] is binding['thinking'] is True
        and identity['source'] == binding['client'] == client_fingerprint()
        and identity['context'] == binding['context'] == result['machine']['context'] == 262144
        and identity['workload'] == binding['workload'] == result['params']['workload'] == workload
        and all(identity[k] == binding[k] == result['machine'][k] for k in ('model', 'quant')),
        'actual client/context/workload identity differs')
    text = (root / 'engine.stderr').read_text()
    marker = re.findall(r'^strata verify: ordinary-Arena copy enabled mode=0 '
        r'staging_bytes=(\d+) arena_bytes=(\d+)$', text, re.M)
    require(len(marker) == 1 and int(marker[0][0]) == 16*2662400
        and int(marker[0][1]) > int(marker[0][0]), 'actual common private-Arena admission required')
    copies = copy_records(text, [e['sample'] for e in events if e['kind'] == 'result'], arm=ARMS[INDEX])
    return rows, events, dict(warm, valid=True, journal_sha256=sha(path),
        copy_request_binding_checked=len(copies), requested_copy_bytes=sum(r['requested_bytes'] for r in copies),
        physical_bus_traffic_or_saturation_qualified=False)


def decision(processes):
    out = runner_context().original_decision(processes)
    out.update(variable='--pcie-frac 0 versus0.0625;adapt-swaps0 common',
        required_agent_and_dedicated_prefill_pending=True,
        next='Judge the repeated per-workload distribution tradeoff;preserve independent prefill and decode. '
            'Agent/task/physical262144/RESTORE,ordinary teardown,arbitrary-client nonfinite-parser repair '
            'and default adoption remain separate. No hardware saturation or70TG claim from stage shares.')
    return out


def runner_context():
    spec = importlib.util.spec_from_file_location('coherent_decode_clean_context', runner.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx)
    ctx.select_mode('cohort'); ctx.ROOT, ctx.KEY, ctx.WORKLOAD_ID = ROOT, KEY, WORKLOAD_ID
    ctx.ARMS = tuple((a, 0) for a in ARMS)
    ctx.original_decision = ctx.decision
    ctx.current, ctx.environment, ctx.save, ctx.compact = current, environment, save, compact
    ctx.route, ctx.official, ctx.decision, ctx.clean_observers = route, official, decision, clean_observers
    return ctx


def prepare():
    require(not ROOT.exists() and KEY not in json.loads(RESULT.read_text()), 'fresh comparison only')
    failed = reuse.build.validation.committed(RESULT, ('coherent_decode_pcie_speed_v1',))
    require(failed['stage'] == 'failed' and not failed['passed'] and not failed['complete']
        and failed['ownership_closed'] and not failed['active'] and failed['processes'] == []
        and failed['failed_action'] == 'host', 'preserve original V1 host prelaunch failure')
    n, proof, prior, providers = basis()
    contract, manifest = runner.model.python_manifest()
    assets = shared.models.model_pins(prior['configuration'], prior['model_assets'], full=True)
    row = {k: copy.deepcopy(prior[k]) for k in ('binary', 'native_source_commit',
        'native_validation_probe', 'configuration', 'environment_template', 'template', 'boot', 'driver')}
    row.update(stage='prepared', source_commit=shared.gen.git('rev-parse', 'HEAD').decode().strip(),
        native_observer_admission=proof, sources=core.pins(shared.gen.git('rev-parse', 'HEAD').decode().strip()),
        python_contract=contract, python_selected=manifest['selected'], model_assets=assets,
        runtime_manifest=prior['runtime_manifest'], plans=[], processes=[], host_passed=False,
        passed=False, complete=False, active=False, ownership_closed=True, whole_bytes=448 << 20,
        budgets=dict(wall=2700, cpu=16200, rss_cap=104 << 30, text_cap=16 << 20,
            file_cap=32 << 20, total_cap=64 << 20),
        previous_attempt=dict(key='coherent_decode_pcie_speed_v1', source_commit=failed['source_commit'],
            stage='failed', GPU_executed=False,
            correction='V1saved successful copy controls before the inherited host gate rechecked committed state. '
                'V2keeps controls local until host closure;native/configuration/workload/performance criteria unchanged.'),
        plan=dict(question='Does assigning1/16of distinct decode misses to existing GPU copies improve sustained generation?',
            evidence='Corrected finite1/16/EARLY0 same-build32769/82/32769handoff passes. Existing caller/pool/wait '
                'observations do not establish saturation. This changes CPU/GPU/PCIe work distribution,not native math.',
            order='Six fresh processes R,C,C,R,R,C. Four official warmups then regular-code and varied-prose once '
                'each per process. Commit each closed process before the next;no speed-based early stopping.',
            constants='Coherent14508/dcb8,262144logical/20480INT8resident,1256static cache slots,5workers,spec4,'
                '8192throughput BF16/vLLM ordered-exp/bulk1/staging0,largeGU/head2/dense1. ARENA_COPY1,'
                'VERIFY_OVERLAP1,EARLY0 and qualified runtime safety/preload in both arms. Only finite '
                '--pcie-frac0/0.0625 differs. No native build or process environment mutation.',
            observers='Keep quiet parameter validation and the retained-reference preload. No trace,per-job census,'
                'CPU service/work,timing,query or caller observers. Ordinary native request/copy/cache footers remain. '
                'Read counters/providers at paused gates outside request clocks. Requested bytes are not physical traffic.',
            acceptance='All6processes/12long requests must have>=32768fresh input,>=1024generated including thinking,'
                '>=512answer chars,>=256offered drafts,complete SSE/journal,valid clocks/TTFT,matched bodies,'
                'actual GPU-prefill/largeGU/Down/handoff and mode2draft chunks,strict ordered API/prompt/copy count,'
                'draft and rounded-time joins,exact private host-view/retirement conservation. Baseline copies0;'
                'candidate long copies positive with positive CPU remainder. Source/build/12ELFs/provider/model/owner '
                'identities exact,normal0,quiet validation,no freshfault;all15full asset hashes before/after cohort.',
            decision='Keep individual samples and all3adjacent paired TG,prompt and TTFT ratios per workload. '
                'Reproducible TG gain requires all3candidate/referenceTG>1,without a minimum floor. Strong code '
                'requires every acceptance>=.60;weak prose every<=.40. Class failure does not stop collection. '
                'This screen does not adopt:agent,dedicated prefill preservation,task quality,fullphysical262144/'
                'RESTORE and ordinary teardown remain open;fix nonfinite parser before arbitrary-client adoption.',
            stopping='First source/input/provider/owner/clock/route/validation/fault/budget/eligibility failure stops. '
                'Durable exclusive attempt precedes preflight. No retries,hash refresh,exclusion or changed criteria.',
            retention='Existing coherent generation only. Keep compact individual samples,actual copy/cache/route '
                'counts,source/recipe/runtime/configuration and decision. Retire normal raw logs/JIT at verified closure. '
                'Generated config/plan bytes are reconstructible from committed base configuration,arm,root and probe.'))
    ROOT.mkdir(mode=0o700)
    for index, arm in enumerate(ARMS):
        root = ROOT / str(index); root.mkdir(mode=0o700)
        for name in ('tmp', 'cache'): (root / name).mkdir()
        plan = dict(index=index, kernel=arm, adapt_swaps=0, heads=1, root=str(root))
        core.write(root / 'config.json', configuration(row, plan))
        core.write(root / 'plan.json', dict(binary=row['binary'], native_validation_probe=row['native_validation_probe']))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        plan.update(generated={s: sha(root / s) for s in ('config.json', 'plan.json')},
            command=['/bin/sh', str(P / 'bench.sh'), '--config', str(root / 'config.json'), '--port', str(port),
                '--long-generation-only', '--think', '--runs', '1', '--profile-request-gates',
                '--workload-id', WORKLOAD_ID, '--fail-fast-engine-error', '--ready-timeout', '600',
                '--out', str(root / 'bench')])
        plan['environment_sha256'] = runner.comparison.fingerprint(environment(row, plan)); row['plans'].append(plan)
    row['invariant_signature'] = invariant(row, row['plans'][0])
    require(all(invariant(row, p) == row['invariant_signature'] for p in row['plans']), 'undeclared arm difference')
    save(row)
    print('Prepared6fresh official bench.sh processes/12thinking-enabled long generations;commit before host/model.', flush=True)


def execute(action):
    global INDEX
    INDEX = int(action) if action.isdigit() else None
    attempt = ROOT / ('host-attempt.json' if INDEX is None else f'{INDEX}/attempt.json')
    # Claim before any source/provider/journal preflight; a failure is terminal.
    with attempt.open('x') as stream: json.dump(dict(action=action, result_key=KEY), stream)
    row = json.loads((ROOT / 'record.json').read_text())
    try:
        if INDEX is None:
            current(row)
            controls = copy_controls()
            runner.comparison.host(runner_context())
            row = json.loads((ROOT / 'record.json').read_text())
            row['copy_controls'] = controls; save(row)
        else:
            runner_context().run(INDEX)
    except BaseException as error:
        row = json.loads((ROOT / 'record.json').read_text())
        if row['stage'] != 'failed':
            row.update(stage='failed', passed=False, complete=False, error=repr(error), failed_action=action)
            save(row)
        raise


if __name__ == '__main__':
    require(sys.executable == PYTHON, 'Use the qualified interpreter: '+PYTHON)
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'host', *map(str, range(len(ARMS)))), 'prepare|host|0..5')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); build.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else execute(sys.argv[1])
