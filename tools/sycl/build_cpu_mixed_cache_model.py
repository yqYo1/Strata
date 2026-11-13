#!/usr/bin/env python3
"""Coherent opt-in mixed-NT decode build; Main authored, repository LICENSE.

The successful CPU comparison admits build/plan preparation, not model speed,
quality, context capacity or adoption. No GPU work is launched by this tool.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import build_coherent_prefill as coherent
import cpu_mixed_cache as cpu

P, B, RESULT = coherent.P, coherent.B, coherent.RESULT
ROOT, KEY = B / 'cpu-mixed-cache-native-v2', 'cpu_mixed_cache_native_v2'
CPU_RESULT, CPU_KEY = cpu.RESULT, cpu.KEY
EVIDENCE = '29193d347e5235e192a31d740b897cec24c25c49'
RETIREMENT = 'mixed_nt_prebuild_retirement_v1'
require, sha = coherent.require, coherent.sha
class_prerequisite = coherent.prerequisite
coherent.TARGETS = [*coherent.TARGETS, 'sycl_expert_dispatch_bounds_test']
coherent.base.common.PREFIXES = tuple(dict.fromkeys((*coherent.base.common.PREFIXES,
    'AGENTS.md', 'docs/ARTIFACT_RETENTION.md')))


def fields(value):
    names = ('component_source_commit', 'fixture_and_controller_commit',
        'plan_path', 'plan_sha256', 'origin', 'cost_evidence', 'policy',
        'representative_cases', 'prospective_comparison', 'build',
        'numeric_results', 'numeric_exit_codes', 'numeric_record_sha256',
        'source_commit', 'case_columns', 'cases', 'sample_columns', 'samples',
        'summary', 'setup_columns', 'setup_samples', 'execution_receipts',
        'eligibility', 'gates', 'complete', 'active', 'ownership_closed',
        'performance_qualified', 'component_eligible_for_model_plan',
        'model_trial_allowed', 'adopted')
    return json.dumps({k: value[k] for k in names}, sort_keys=True,
        separators=(',', ':'), allow_nan=False)


def prerequisite():
    proof = class_prerequisite()
    git = coherent.base.common.source.git
    path = str(CPU_RESULT.relative_to(P))
    original = json.loads(git(P, 'show', EVIDENCE + ':' + path))[CPU_KEY]
    committed = json.loads(git(P, 'show', 'HEAD:' + path))
    live = json.loads(CPU_RESULT.read_text())
    require(fields(original) == fields(committed[CPU_KEY]) == fields(live[CPU_KEY]),
            'original mixed-cache qualification fields changed')
    require(original['complete'] and original['ownership_closed'] and not original['active']
        and original['performance_qualified'] and original['component_eligible_for_model_plan']
        and not original['model_trial_allowed'] and not original['adopted'],
        'closed successful component evidence; preserve model qualification hold')
    require(len(original['summary']) == 12 and all(s['all_three_pairs_gain']
        and len(s['paired_reference_over_candidate']) == 3
        and all(x > 1 for x in s['paired_reference_over_candidate'])
        for s in original['summary']), 'all twelve original repeated component gates')
    require({r['down'] for r in original['numeric_results']} == {20, 42}
        and all(r['result'] == cpu.expected_numeric(r['down'])
        for r in original['numeric_results']), 'original actual-route numeric controls')
    plan = json.loads(cpu.PLAN.read_text())
    cpu.verify(plan)
    built = cpu.base.qualified_build(plan)
    checked = json.loads((cpu.ROOT / 'check.json').read_text())
    require(cpu.base.closed(checked, plan) and checked['binaries'] == built['binaries']
        and built['binaries']['pool'] == original['build']['binary']
        and sha(original['build']['original_private_record']) ==
            original['build']['original_private_record_sha256'],
        'same coherent CPU generation, numerics and final executable')
    retired = live[RETIREMENT]
    require(retired == committed[RETIREMENT] and retired['stage'] == 'retired'
        and retired['original_source_recipe_final_inputs_recovered'] is True
        and all(not (B / name / 'build').exists() for name in (
            'cpu-compact-small-batch-pool-v1', 'cpu-native-whole-down42-v2',
            'cpu-joint-census-native-v1')), 'retire ended comparison builds first')
    failed = coherent.base.validation.committed(RESULT, ('cpu_mixed_cache_device_v1',))
    old = coherent.base.validation.committed(RESULT, ('cpu_mixed_cache_native_v1',))
    require(failed['stage'] == 'failed' and failed['ownership_closed'] and not failed['active']
        and len(failed['results']) == 1 and not failed['results'][0]['GPU_executed']
        and old['artifact_retirement']['stage'] == 'retired'
        and old['artifact_retirement']['original_source_recipe_final_inputs_recovered']
        and not (B / 'cpu-mixed-cache-native-v1').exists(),
        'preserve the V1host SIGILL and retire its ended candidate before V2')
    return dict(coherent_class_basis=proof, CPU_evidence_commit=EVIDENCE,
        CPU_source_commit=original['source_commit'], CPU_plan_commit=original['build']['plan_commit'],
        CPU_qualification_sha256=hashlib.sha256(fields(original).encode()).hexdigest(),
        CPU_binary=built['binaries']['pool'], retired_generations=retired['generations'],
        scope='Opt-in coherent build and new model-plan preparation only. Earlier whole-pool '
            'stress/failure evidence supplies no admission. Preserve cache0,SmallJobs,MTP and both '
            'prefill strategies. Fresh native and actual model route/numeric/setup/RSS/handoff, '
            'repeated long/agent/thinking strong/weak,physical262144/tails/RESTORE gates remain.')


coherent.ROOT, coherent.KEY = ROOT, KEY
coherent.SEED = B / 'coherent-prefill-native-v1/record.json'
coherent.SEED_KEY = 'coherent_prefill_native_v1'
coherent.prerequisite = prerequisite


def prepare():
    coherent.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['plan'] = ('Fresh complete compilation of engine and all thirteen executables after explicit '
        'decode MixedNt wiring and ExpertDispatch header change;no native object/archive reuse. '
        'STRATA_CPU_COMPACT_GU_POLICY=small remains default;cache budget0 remains default. '
        'Only a separately committed model plan may choose mixed with an admitted immutable '
        'cache. Mixed selection is decode-only,workers5+host,NT1..4,totalrows<=40;onlyNT1 '
        'uses encoded GU. Quantization,NT2..4,Down and prefill math remain native. Unsupported '
        'shapes/source/format/cache admission fall back. Record actual bindings and native audit '
        'separately from clean speed. Check ordinary teardown and physical262144 qualification '
        'independently. No speed/quality/model qualification is inherited from CPU components.')
    row['retention'] = ('Keep clean14508/dcb8 baseline and this one temporary model candidate. '
        'Ended f59observer,492whole-stress and c0dc/c858CPUfixtures must be retired first with '
        'original-source/recipe/final-input recovery. Keep3d7f CPUfixture only as this exact live '
        'qualification input. Review/retire the extra engine at model comparison closure;legacy '
        'IR/deployment/provenance holds remain separate. Do not retain successful raw output.')
    coherent.context().save(row)


if __name__ == '__main__':
    require(sys.executable == cpu.hot.PYTHON and len(sys.argv) == 2
        and sys.argv[1] in ('prepare', 'configure', 'compile'),
        'qualified Python;prepare|configure|compile')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        coherent.base.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else coherent.context().build(sys.argv[1])
