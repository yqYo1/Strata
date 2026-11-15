#!/usr/bin/env python3
"""Build the measured decode-only IQ3 candidate; repository LICENSE.

Recover the fixed Q6/phase/observer engine from Git, then change only the
qualified CPU component, its fixtures and a checked request-level route record.
Reuse the maintained build supervisor in a private context. Main owns all runs.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import build_cpu_service_model as original
import cpu_service_sources as source

P, B, RESULT, base = original.P, original.B, original.RESULT, original.base
ROOT = B / 'decode-iq3-pair-model-v1'
KEY = 'decode_iq3_pair_model_build_v1'
SELF = Path(__file__).resolve()
COMPONENT_EVIDENCE = 'cf0a6b175261f013099965dcb9ca19d1e840ad2d'
COMPONENT_SOURCE = '1ec110452cdd70c425bbc31edd48bbd0421563ca'
AUDIT_SOURCE = '7969ccb9f5fe3c4900e79df7937e709830e4d7e4'
CPU_FILES = ('include/strata/kernels/cpu/native_expert.hpp',
             'include/strata/kernels/cpu/iq_avx2.hpp',
             'src/kernels/cpu/native_expert.cpp',
             'src/kernels/cpu/iq_single_avx2.cpp',
             'tests/core/iq_single_exact_test.cpp',
             'tests/core/cpu_service_pool_test.cpp')
TARGETS = ('strata', 'cpu_service_pool_test', 'iq_single_exact_test',
           'sycl_submission_test', 'sycl_verifier_retirement_test')
require, digest, write = base.require, base.digest, base.write


def component():
    raw = base.basis.build.git(P, 'show', COMPONENT_EVIDENCE + ':' + str(RESULT.relative_to(P)))
    row = json.loads(raw)['cpu_iq3_pair_dispatch_v3']
    require(row['stage'] == 'closed' and row['passed'] and row['complete'] and
            row['ownership_closed'] and not row['active'] and
            row['source_commit'] == COMPONENT_SOURCE and len(row['processes']) == 6 and
            row['correctness']['passed'] and row['correctness']['prefill_pair_rows'] == 0 and
            len(row['decision']['cells']) == 8 and all(
                c['component_gain_qualified'] and len(c['paired_candidate_over_reference']) == 3 and
                all(x < 1 for x in c['paired_candidate_over_reference']) for c in row['decision']['cells']),
            'closed winning production-entry and real-pool component evidence required')
    return row


def projection():
    native, before = source.projection()
    old = base.committed(RESULT)[original.KEY]
    pins, fingerprint = source.aggregate(before)
    require(old['native_sources'] == dict(files=len(pins), aggregate_sha256=fingerprint) and
            old['source_commit'] == '8e2499937969dbfa1dcd950b95025d9712037a83',
            'recover every original observer source byte from its declared Git construction')
    component()
    values = dict(before)
    for name in CPU_FILES:
        raw = base.basis.build.git(P, 'show', COMPONENT_SOURCE + ':' + name)
        require((P / name).read_bytes() == raw, 'qualified CPU source drift: ' + name)
        values[name] = raw
    merge = base.basis.merge_delta
    values[source.PROGRAM] = merge(before[source.PROGRAM], source.PROGRAM,
                                   AUDIT_SOURCE + '^', AUDIT_SOURCE)
    require(merge(values[source.PROGRAM], source.PROGRAM, AUDIT_SOURCE,
                  AUDIT_SOURCE + '^') == before[source.PROGRAM],
            'request-route evidence changed unrelated generator code')
    require(set(values) == set(before) and len(values) == 563 and
            {n for n in values if values[n] != before[n]} == {*CPU_FILES, source.PROGRAM},
            'only six CPU/fixture files and the route-record patch may change')
    # In particular, all prefill/GPU math, phase ownership, cache placement,
    # scheduler, runtime and graph source bytes remain those already measured.
    return native, values


def recipe(native, root):
    value = source.recipe(native, root)
    value['targets'] = list(TARGETS)
    value['environment'].update(STRATA_IQ_SINGLE_EXACT='0', STRATA_IQ3_PAIR='0',
        STRATA_IQ3_PAIR_AUDIT='1', STRATA_CPU_SERVICE='0')
    value['host_modes'] = [dict(STRATA_IQ3_PAIR=str(arm), STRATA_CPU_SERVICE='0') for arm in (0, 1)]
    value['retention'] = ('Imminent same-binary decode IQ3 on/off comparison. Keep479a fallback;'
        'retire the closed observer engine and closed CPU-only comparison products before building.'
        'New five-output generation needs same-build arithmetic/pool and small traced device gates,'
        'then short actual-model route admission before long generation. No third inference generation.')
    return value


def source_basis():
    native, values = projection()
    old = base.committed(RESULT)[original.KEY]
    overview = base.committed(RESULT)['verify_window_cpu_service_overhead_v1']
    require(overview['passed'] and overview['complete'] and overview['ownership_closed'] and
            not overview['active'], 'close the original six-process observer comparison first')
    return native, dict(root=str(original.ROOT),
        native_source_commit=COMPONENT_SOURCE,
        source_construction_aggregate=source.aggregate(values)[1],
        original_observer_source_commit=old['source_commit'],
        original_observer_source_aggregate=old['native_sources']['aggregate_sha256'],
        original_observer_engine_sha256=old['binaries']['strata']['sha256'],
        component_evidence_commit=COMPONENT_EVIDENCE, component_source_commit=COMPONENT_SOURCE,
        generator_route_patch_commit=AUDIT_SOURCE, changed_files=[*CPU_FILES, source.PROGRAM],
        scope='Exact-source reconstruction and closed component qualification only;all new outputs need their own admission.')


def context():
    spec = importlib.util.spec_from_file_location('strata_iq3_native_build', original.__file__)
    ctx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctx)
    ctx.ROOT, ctx.KEY, ctx.CONTROLLER = ROOT, KEY, SELF
    ctx.source = SimpleNamespace(**{n: getattr(source, n) for n in
        ('NATIVE_ROOT', 'PROGRAM', 'POOL', 'HEADER', 'FIXTURES', 'aggregate')},
        NATIVE=COMPONENT_SOURCE, projection=projection, recipe=recipe)
    ctx.source_basis = ctx.selected_basis = source_basis

    def inputs(row):
        revision = base.basis.build.git(P, 'rev-parse', 'HEAD').decode().strip()
        require(base.controls(row['source_commit'], controller=SELF) == row['controller_inputs'],
                'commit and freeze complete IQ3 build controls')
        native, proof = source_basis()
        _, values = projection()
        require(proof == row['selected_basis'] and source.aggregate(values)[0] == row['native_sources'] and
                recipe(native, ROOT) == row['recipe'], 'IQ3 source/recipe qualification drift')
        require(original.measured.build.monitor_qualification(base.committed(RESULT)) ==
                row['monitor_qualification'], 'qualified EOF/pipe build supervisor changed')
        fallback = json.loads((original.selected.model.ROOT / 'record.json').read_text())
        allowed = original.selected.model.inputs(fallback, reuse_revision=revision,
            controller_changes=original.selected.NATIVE_CONTROLS)
        allowed = {p: h for p, h in allowed.items()
                   if not Path(p).is_relative_to(original.selected.model.ROOT)}
        for name, raw in values.items():
            path = ROOT / 'source' / name
            require(path.is_file() and not path.is_symlink() and path.read_bytes() == raw,
                    'frozen IQ3 native input drift: ' + name)
            allowed[str(path.resolve())] = row['native_sources'][name]
        allowed.update(row['generated_expected'])
        return allowed

    def retention_gate():
        review = base.committed(RESULT)['retained_builds_review_20261103']
        row = review.get('iq3_model_replacement', {})
        require(row.get('completed') and row.get('original_inputs_recovered') and
                row.get('next_root') == str(ROOT) and row.get('old_root') == str(original.ROOT),
                'recover and retire the closed observer outputs before another engine generation')
        require(digest(row['manifest_path']) == row['manifest_sha256'], 'retirement manifest drift')
        retired = json.loads(Path(row['manifest_path']).read_text())
        require(retired['completed'] and retired['files'] and
                all(not Path(x['path']).exists() for x in retired['files']) and
                not (original.ROOT / 'build').exists(), 'old observer generation is still present')
        fallback = json.loads((original.selected.model.ROOT / 'record.json').read_text())
        require(all(Path(v['path']).is_file() and digest(v['path']) == v['sha256']
                    for v in fallback['binaries'].values()), 'protect all validated fallback outputs')
        return review

    ctx.inputs, ctx.retention_gate = inputs, retention_gate
    return ctx


def prepare(ctx):
    ctx.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['host_plan'] = dict(arms=[0, 1], CPU_SERVICE=0, IQ3_PAIR_AUDIT=1,
        SINGLE_EXACT=0, wall_each=60, expected_exact_single=384,
        expected_exact_pair=80, expected_pool_cases=28,
        scope='Four fresh same-build CPU children:exact0,pool0,exact1,pool1. '
            'Require actual decode routing,prefill exclusion,bitwise GGML/canaries '
            'and equal real-pool checksums;not pool/model speed.')
    row['model_comparison_plan'] = dict(
        question='Does the winning exact CPU IQ3 NT1 gate/up path improve sustained model decode?',
        selector='One binary,immutable STRATA_IQ3_PAIR0/1 in separate owned processes;prefill unchanged.',
        qualification='First four same-build host cases,then existing traced submission/phase GPU checks,'
            'then two thinking-on short-output actual32K requests in each selector with complete '
            'IQ3_GU_DELTA and final model hashes. These stages do not authorize an automatic long run.',
        comparison='After entry admission,commit six fresh processes0,1,1,0,0,1 through maintained '
            'bench.sh long-generation workloads with thinking on and >=1K output;actual strong/weak '
            'MTP classes,all individual samples and same-request dispatch;all three pairs faster '
            'per workload qualifies reproducible gain without a minimum percent floor.',
        remaining='Dedicated64K/117K PP/TTFT preservation,agent/task,prompt-verify,physical262144/RESTORE '
            'and ordinary provider teardown remain separate;no component-to-model speed projection.')
    ctx.save(row)


def host(ctx):
    row = ctx.verify()
    require('host_result' not in row and not (ROOT / 'host').exists(), 'one frozen same-build CPU check')
    owner, result = None, dict(passed=False, complete=False, cases=[], GPU_executed=False)
    def persist():
        write(ROOT / 'host-execution.json', dict(active=bool(owner and owner.active),
            commands=owner.commands if owner else []))
    try:
        owner = base.supervision.Owner(ROOT / 'host', persist=persist)
        sums = []
        for arm in (0, 1):
            env = dict(row['recipe']['environment'], STRATA_IQ3_PAIR=str(arm))
            for kind, target, args in [('exact', 'iq_single_exact_test', ['--dispatch']),
                                       ('pool', 'cpu_service_pool_test', ['0'])]:
                entry, out, err = owner.run(kind + '-' + str(arm),
                    [row['binaries'][target]['path'], *args], env, ROOT,
                    wall=60, cpu=60, rss_cap=512 << 20, text_cap=64 << 10,
                    file_cap=64 << 10, total_cap=1 << 20)
                require(base.supervision.closed(entry) and entry['exit_code'] == 0 and not err.stat().st_size,
                        'same-build IQ3 fixture failed: ' + err.read_text(errors='replace')[-2000:])
                text = out.read_text().strip()
                if kind == 'exact':
                    expected = ('IQ single exact PASS: 384 single and 80 paired cases, bitwise GGML parity and row canaries\n'
                        f'IQ3 production dispatch PASS arm={arm} calls=80 nt1_rows=20480 pair_rows={arm * 10240} prefill_pair_rows=0')
                    require(text == expected, 'exact arithmetic/phase/route output incomplete')
                else:
                    match = re.fullmatch(r'cpu_service_pool PASS observer=0 formats=7 nt=1\.\.4 cases=28 checksum_fnv1a64_le=([0-9a-f]{16})', text)
                    require(match is not None, 'real-pool output incomplete')
                    sums.append(match[1])
                result['cases'].append(dict(kind=kind, arm=arm, passed=True, exit_code=0))
        require(len(sums) == 2 and sums[0] == sums[1] == component()['correctness']['pool_checksum'],
                'actual model-library pool checksum differs from the qualified CPU component')
        result.update(passed=True, complete=True, pool_checksum=sums[0])
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        result.update(active=bool(owner and owner.active), ownership_closed=bool(owner and
            owner.active is None and all(base.supervision.closed(e) for e in owner.commands)))
        row['host_result'] = result
        ctx.save(row)
    require(result['passed'] and result['ownership_closed'], 'same-build IQ3 arithmetic/pool gate failed')
    print('Four same-build IQ3 CPU children pass;GPU/model qualification pending.', flush=True)


if __name__ == '__main__':
    require(len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'build', 'host', 'verify'),
            'usage: build_iq3_model.py prepare|build|host|verify')
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.guards.no_model_owners([])
        ctx = context()
        {'prepare': lambda: prepare(ctx), 'build': ctx.build,
         'host': lambda: host(ctx), 'verify': ctx.verify}[sys.argv[1]]()
