#!/usr/bin/env python3
"""Compare the qualified sole-packed target/MTP head through bench.sh; Main only.

Reuse the maintained finite cohort supervisor and its current frontend,
body/journal, runtime, route and closure checks. Only the head selector differs.
Main-authored; repository LICENSE applies.
"""
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import q6_head_model_entry as entry
import pool_spin as protocol
import official_model_entry as frontend
import engine_once
from iq3_model_entry import dispatches

P, B, RESULT, build = entry.P, entry.B, entry.RESULT, entry.build
ROOT, KEY = B / 'q6-packed-head-generation-v1', 'q6_packed_head_generation_speed_v1'
ARMS = (('reference', 96), ('candidate', 96), ('candidate', 96),
        ('reference', 96), ('reference', 96), ('candidate', 96))
require, digest, signature = entry.require, entry.sha, entry.signature
SELF = Path(__file__).resolve()


def pins(revision):
    result = protocol.shared.gen.benchmark_pins(revision)
    paths = build.source.git(P, 'ls-tree', '-r', '--name-only', revision, '--', 'strata_control').decode().splitlines()
    for name, raw in build.source.blobs(P, revision, [p for p in paths if p.endswith('.py')]):
        require((P / name).read_bytes() == raw, 'uncommitted shared-control import: ' + name)
        result[name] = hashlib.sha256(raw).hexdigest()
    name = 'data/expert-profile.bin'
    data = build.source.git(P, 'show', revision + ':' + name)
    require(data == (P / name).read_bytes(), 'committed expert profile changed')
    result[name] = hashlib.sha256(data).hexdigest()
    return result


def configuration(reference):
    config = copy.deepcopy(reference['configuration'])
    config.update(exe=str(P / 'tools/sycl/engine_once.py'), family='qwen', backend='sycl', env={})
    args = config['args']
    args[args.index('--adapt-swaps') + 1] = '96'
    if '--adapt-every' not in args: args += ['--adapt-every', '1']
    require(args[args.index('--adapt-every') + 1] == '1' and
        args[args.index('--max-context') + 1] == args[args.index('--kv-resident') + 1] == '131072' and
        args[args.index('--prefill') + 1] == '8192' and args[args.index('--expert-cache') + 1] == '1256',
        'preserve the measured long-generation placement')
    return config


def basis(revision):
    model = json.loads((entry.ROOT / 'record.json').read_text()); entry.verify(model)
    require(model['stage'] == 'closed' and model['passed'] and model['complete'] and
        model['ownership_closed'] and len(model['results']) == 2 and
        all(r['passed'] and r['native_exit_code'] == 0 and r['fresh_faults'] == []
            for r in model['results']), 'both current same-build model entries must close first')
    native = entry.native()
    reference = json.loads(entry.REFERENCE.read_text())
    contract, manifest = frontend.python_manifest()
    require(reference['model_assets'] == model['models'], 'fixed model assets differ')
    config = configuration(reference)
    original_environment = reference['environment_template']
    environment = {k: v.replace(original_environment['STRATA_MEASUREMENT_ROOT'], str(ROOT))
                   for k, v in original_environment.items()}
    environment.update(PATH=str(Path(manifest['selected']['path']).parent) + ':/usr/bin:/bin',
        STRATA_IQ3_PAIR_AUDIT='0', STRATA_SYCL_Q6_ROUTE_AUDIT='0',
        TMPDIR=str(ROOT / 'tmp'), XDG_CACHE_HOME=str(ROOT / 'cache'),
        PYTHONPYCACHEPREFIX=str(ROOT / 'cache/python'), STRATA_MEASUREMENT_ROOT=str(ROOT),
        UR_LOG_VALIDATION='level:warning;flush:warning;output:file,' + str(ROOT / 'validation.log'))
    environment.pop('LD_PRELOAD', None); environment.pop('STRATA_UR_PROBE_EXPECTED_LOADER', None)
    prior = dict(source_commit=model['source_commit'], sources=model['controller']['source_sha256'],
        binary=native['binaries']['strata'], native_source_commit=native['project']['revision'],
        native_validation_probe=model['probe'], configuration=config, template=reference['template'],
        boot=model['boot'], driver=model['driver'], python_contract=contract, python_selected=manifest['selected'],
        environment=environment, model_assets=model['models'])
    providers = model['runtime_files']
    pool = {p: native['project']['source_sha256'][p]
            for p in ('src/kernels/cpu/pool.cpp', 'include/strata/kernels/cpu/pool.hpp')}
    require(pins(revision).get(str(SELF.relative_to(P))) == digest(SELF), 'commit the new cohort controller')
    return prior, providers, manifest, pool


def head_evidence(text, arm):
    base, packed = entry.route_tables(text, arm)
    requests = {k[0] for k in base}
    require(len(requests) == 1 and requests <= {'5', '6'}, 'one actual long request5/6 required')
    heads = [(k, v) for k, v in base.items() if k[2:4] == ('decode', '4')]
    require(len(heads) == 1 and int(heads[0][1]['replay_completed']) > 0, 'positive actual target-head replay required')
    key, fields = heads[0]
    return dict(request=int(key[0]), capture_id=int(key[4]), completed=int(fields['replay_completed']),
        packed_calls=int(packed[key]['calls']) if arm else 0, sole_packed_all_live_head_windows=arm == 2,
        scope='one live capture epoch;completed is cumulative within its identity')


def head_transfer(text, arm):
    pattern = r'^strata prefill heads: target detached bytes=(\d+) source=owned-RAM D2H=(\d+) H2D-on-restore=(\d+)$'
    rows = re.findall(pattern, text, re.M)
    require(type(arm) is int and arm in (0, 2), 'explicit head arm')
    expected = ('521472000', '521472000', '521472000')
    require(rows == [expected], 'actual head retirement/restore extent differs')
    drafts = re.findall(r'^strata prefill heads: draft detached bytes=(\d+) retained-map-bytes=(\d+)$', text, re.M)
    require(drafts == [('0' if arm == 2 else '223227900', '425196')], 'actual MTP subset transfer differs')
    # Preserve the actual text, including the candidate's real0B draft retirement.
    return text, dict(freed=int(expected[0]),
        D2H=int(expected[1]), H2D=int(expected[2]), draft_freed=int(drafts[0][0]), retained_map_bytes=425196)


def head_controls():
    def sample(arm):
        lines = []
        for phase, tokens in (('prompt', 4), ('decode', 1), ('decode', 4)):
            shared = f'request=5 device=0 phase={phase} tokens={tokens} capture_id=2'
            lines.append('strata Q6_ROUTE ' + shared + ' scope=live_capture_cumulative native_half_enabled=0 complete=1 replay_started=3 replay_completed=3')
            if arm: lines.append('strata Q6_PACKED_HEAD ' + shared +
                ' layout=sole-packed extra_weight_bytes=0 calls=1 complete=1')
        return '\n'.join(lines)
    for arm in (0, 2):
        head_evidence(sample(arm), arm)
        line = f'strata prefill heads: target detached bytes=521472000 source=owned-RAM D2H=521472000 H2D-on-restore=521472000'
        line += '\nstrata prefill heads: draft detached bytes=' + ('0' if arm == 2 else '223227900') + ' retained-map-bytes=425196'
        normalized, actual = head_transfer(line, arm)
        require('bytes=521472000 ' in normalized and actual['H2D'] == 521472000, 'head transfer control')
    text = sample(2)
    invalid = [text.replace('layout=sole-packed', 'layout=canonical', 1),
        text.replace('extra_weight_bytes=0', 'extra_weight_bytes=521472000', 1),
        text.replace('calls=1', 'calls=0', 1), text.replace('complete=1', 'complete=0', 1),
        text.replace('replay_completed=3', 'replay_completed=2', 1), text + '\n' + text.splitlines()[0],
        '\n'.join(v for v in text.splitlines() if not v.startswith('strata Q6_PACKED_HEAD request=5 device=0 phase=prompt'))]
    for value in invalid:
        try: head_evidence(value, 2)
        except RuntimeError: continue
        raise RuntimeError('corrupted head policy/join accepted')
    bad_transfer = [line.replace('H2D-on-restore=521472000', 'H2D-on-restore=0'),
        line.replace('draft detached bytes=0', 'draft detached bytes=223227900'),
        line.replace('retained-map-bytes=425196', 'retained-map-bytes=0'), line + '\n' + line]
    for value in bad_transfer:
        try: head_transfer(value, 2)
        except RuntimeError: continue
        raise RuntimeError('corrupted target/MTP transfer accepted')
    # Each fresh imported context owns its parser expectation. The common
    # legacy module stays unchanged; no device log/counter is normalized.
    original = protocol.comparison.DRAFT_BYTES
    for index, expected in ((0, 223227900), (1, 0)):
        ctx = context(index)
        require(ctx.comparison is not protocol.comparison and ctx.comparison.DRAFT_BYTES == expected and
            protocol.comparison.DRAFT_BYTES == original,
            'head-arm parser expectation leaked into the shared comparison module')
    return dict(passed=True, positive_arms=2, invalid_routes=len(invalid), invalid_transfers=len(bad_transfer),
        isolated_route_expectations=2, actual_transfer_bytes_preserved=True)


def context(index=None):
    require(index is None or type(index) is int and 0 <= index < len(ARMS), 'explicit process index')
    spec = importlib.util.spec_from_file_location('strata_head_speed_context', protocol.__file__)
    ctx = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctx); ctx.select_mode('cohort')
    ctx.ROOT, ctx.KEY, ctx.ARMS = ROOT, KEY, ARMS
    ctx.WORKLOAD_ID = 'q6_packed_head_generation_v1_20261109'
    # pool_spin.route looks up these fields in this independently imported
    # module's globals. Supply a private namespace with just the draft-owner
    # expectation changed; shared comparison functions retain their own globals.
    draft_bytes = 0 if index is not None and ARMS[index][0] == 'candidate' else 223227900
    require(protocol.comparison.DRAFT_BYTES == 223227900, 'original reference draft extent changed')
    ctx.comparison = SimpleNamespace(**(vars(ctx.comparison) | {'DRAFT_BYTES': draft_bytes}))
    ctx.basis = basis
    ctx.model = SimpleNamespace(KEY=entry.KEY, pinned_sources=pins, engine_once=engine_once,
                               mtp_route=protocol.model.mtp_route)
    old_environment, old_route, old_checks, old_decision, old_save = (
        ctx.environment, ctx.route, ctx.selfcheck, ctx.decision, ctx.save)
    require(old_route.__globals__['comparison'] is ctx.comparison,
            'ordinary route reader must use the private head-arm expectation')

    def environment(row, plan):
        base = dict(row, environment_template=dict(row['environment_template'],
            STRATA_IQ3_PAIR_AUDIT='0', STRATA_SYCL_Q6_ROUTE_AUDIT='0'))
        env = old_environment(base, plan)
        env.update(STRATA_SYCL_DECODE_Q6_DPAS_HEAD='2' if plan['kernel'] == 'candidate' else '0',
            STRATA_SYCL_DECODE_Q6_COLUMNS='1', STRATA_SYCL_DECODE_Q6_CENTERING='0',
            STRATA_SYCL_DECODE_Q6_NATIVE_HALF='0', STRATA_SYCL_Q6_ROUTE_AUDIT='1',
            STRATA_IQ3_PAIR='1', STRATA_IQ3_PAIR_AUDIT='1', STRATA_CPU_SERVICE='0', STRATA_CPU_WORK_CENSUS='0',
            STRATA_SYCL_ADAPT_SYNC='1')
        require(plan['adapt_swaps'] == 96 and env['STRATA_SYCL_CACHE_SCRATCH_LEASE'] == '1' and
            env['ZES_ENABLE_SYSMAN'] == env['STRATA_SYCL_VERIFY_NATIVE_CAPTURE'] == '0', 'fixed model/safety policy')
        return env

    def invariant(row, plan):
        config = json.loads((Path(plan['root']) / 'config.json').read_text())
        require(config['args'] == row['configuration']['args'], 'same engine arguments required')
        env = environment(row, plan); env['STRATA_SYCL_DECODE_Q6_DPAS_HEAD'] = '<head-arm>'
        return hashlib.sha256(json.dumps(dict(config=config, environment=env), sort_keys=True)
            .replace(plan['root'], '<process>').encode()).hexdigest()

    def route(text, read):
        require(type(index) is int and 0 <= index < len(ARMS), 'actual process index required')
        arm = 2 if ARMS[index][0] == 'candidate' else 0
        observed, transfer = head_transfer(text, arm)
        result = old_route(observed, read)
        result['target_head_transfer'] = transfer
        result['target_head_route'] = head_evidence(text, arm)
        actual = dispatches(text, 1, positive=True)
        require(len(actual) == 1 and actual[0]['request'] == result['target_head_route']['request'], 'IQ3/head request join')
        result['IQ3_dispatch'] = actual[0]
        result['dispatch_proof'] = 'Observed current native16/GDN3 headers,MTP mode2 callbacks,actual IQ3 entry and target-head capture/replay policy. Source/runtime are pinned;no hardware saturation claim.'
        return result

    def checks():
        return dict(common=old_checks(), head=head_controls())

    def decision(processes):
        result = old_decision(processes)
        result['next'] = 'Judge sole-packed target/MTP TG per workload and PP/TTFT independently. Keep individual pairs and unresolved variance;no minimum speedup floor. If full-vocabulary MTP loses,keep target-head packed eligible and evaluate a separately materialized subset. Dedicated length/prefill,agent/task,physical262144/RESTORE and adoption remain separate.'
        return result

    def save(row):
        if row['stage'] == 'prepared':
            row['qualification_reuse'] = dict(native_build_key=build.KEY, device_key=entry.device.KEY,
                model_key=entry.KEY, model_evidence_commit=build.source.git(P, 'rev-parse', 'HEAD').decode().strip(),
                native_source_commit=row['native_source_commit'], engine_sha256=row['binary']['sha256'],
                scope='Exact current four ELF/source/recipe/runtime checks and both bounded real-model head arms. No native rebuild or inherited sustained-speed result.')
            row['plan'].update(question='Does the sole-packed shared target/MTP Q6 head improve sustained model generation?',
                motivation='Owned packed-vs-columns component gain6.55283x is separate from the new7.238970x generic-prompt component. Neither is a model claim. Exact same-build head0/2 fixture and128K model/handoff checks must pass first. Target521472000B restore remains charged;MTP trades223227900B subset weights for425196B subset logits and full-vocabulary arithmetic.',
                order='Six fresh processes head0,2,2,0,0,2;three counterbalanced adjacent pairs. Four standard warmups,then fixed v5 regular-code and varied-prose once per process. Commit each closed result before next.',
                constants='Same current qualified engine/source,84providers,safety/private retained preload,throughput BF16/GDN16/MTP16,heads1,131072INT8resident,1256slots,5workers,spec4/lease1,8192chunk,caches0,20ms spin,adapt96/every1,IQ3pair1 and Q6columns1. Only immutable head0/2 differs,including its explicitly recorded MTP owner/representation.',
                observers='Keep common parameter validation,retained-runtime preload,IQ3 and Q6 capture/replay audits. Disable TRACE,CPU attribution,DECODE_TIMING,API trace and profiler. Paused identity/provider/body checks lie outside client request clocks;observer overhead and physical saturation remain unqualified.',
                acceptance='Every long request>=32768fresh read,>=1024generated including thought,>=512answer characters,>=256offered drafts;normal0,closed owners,quiet validation,no fault. Preserve standard body/journal/clock/provider/route/cache checks. Verify actual sole target521472000B and MTP0/223227900B retirement against the private arm expectation;do not normalize logs or counters. Require completed target C4 and all live packed head joins,including prompt/C1.',
                decision='Per workload,retain3exact paired TG ratios/geomean and arm median/min/max with actual MTP/output lengths. Reproducible gain needs all3ratios>1,no minimum floor. Strong/weak labels require measured>=.60/<=.40 in every arm sample;classification or speed failure does not stop collection. PP/TTFT decisions stay independent.',
                interpretation='Small model qualification is not speed evidence. Head audit counts are live-capture cumulative,not per-request deltas. No exact-output,task-quality,saturation,full-context or default-adoption claim.')
            row['budgets'] = dict(wall=3600, cpu=43200, rss_cap=104 << 30, text_cap=16 << 20,
                                  file_cap=32 << 20, total_cap=64 << 20)
        old_save(row)

    ctx.environment, ctx.invariant, ctx.route = environment, invariant, route
    ctx.selfcheck, ctx.decision, ctx.save = checks, decision, save
    return ctx


if __name__ == '__main__':
    args = sys.argv[1:]
    require(len(args) == 1 and (args[0] in ('prepare', 'host') or args[0].isdigit() and int(args[0]) < len(ARMS)),
            'q6_head_generation_speed.py prepare|host|INDEX0..5')
    index = int(args[0]) if args[0].isdigit() else None
    ctx = context(index)
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); build.guards.no_model_owners([])
        if args[0] == 'prepare': ctx.prepare()
        elif args[0] == 'host': ctx.comparison.host(ctx)
        else: ctx.run(index)
