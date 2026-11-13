#!/usr/bin/env python3
"""Coherent full build for the independently measured Down42 CPU selector.

Main authored; repository LICENSE. No GPU execution or inherited model speed.
"""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import build_coherent_prefill as coherent
import cpu_native_schedule as cpu

P, B, RESULT = coherent.P, coherent.B, coherent.RESULT
ROOT, KEY = B / 'cpu-down42-model-native-v1', 'cpu_down42_model_native_v1'
CPU_RESULT = P / cpu.CPU_RESULT
CPU_KEY = 'native_whole_down42_v2'
PREVIOUS_ROOT = B / 'compact-cpu-model-native-v1'
PREVIOUS_KEY = 'compact_cpu_model_native_v1'
EVIDENCE = None
class_prerequisite = coherent.prerequisite
require, sha = coherent.require, coherent.sha
# Include this adapter and its CPU admission readers in the committed full-build snapshot.
coherent.base.common.PREFIXES = tuple(dict.fromkeys((*coherent.base.common.PREFIXES,
    'tools/sycl/build_cpu_decode_schedule.py', 'tools/sycl/cpu_native_schedule.py',
    'tools/sycl/cpu_compact_pool.py', 'tools/sycl/cpu_compact_numeric.py',
    'tools/sycl/cpu_compact_hot.py')))


def fields(value):
    # Freeze qualification, rather than unrelated result rows or retirement.
    result = {k: value[k] for k in (
        'source_commit', 'plan_commit', 'plan_sha256', 'cases', 'case_columns',
        'samples', 'sample_columns', 'summary', 'order', 'pairs', 'gates',
        'complete', 'active', 'ownership_closed', 'build', 'numerical_results',
        'eligibility', 'execution_receipts')}
    result['binary_sha256'] = value['binary']['sha256']
    return json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)


def prerequisite():
    proof = class_prerequisite()
    require(isinstance(EVIDENCE, str) and re.fullmatch(r'[0-9a-f]{40}', EVIDENCE),
            'pin the original committed CPU measurement evidence')
    name = str(CPU_RESULT.relative_to(P))
    git = coherent.base.common.source.git
    original = json.loads(git(P, 'show', EVIDENCE + ':' + name))[CPU_KEY]
    committed = json.loads(git(P, 'show', 'HEAD:' + name))[CPU_KEY]
    live = json.loads(CPU_RESULT.read_text())[CPU_KEY]
    require(fields(original) == fields(committed) == fields(live),
            'CPU qualification fields/type changed')
    require(original['complete'] and original['ownership_closed']
            and not original['active'] and original['gates']['build']
            and original['gates']['numerical'] and original['gates']['performance']
            and original['gates']['model_trial_allowed']
            and not original['gates']['model_adopted'], 'closed passing CPU gate first')
    expected = {(42, n, 'observed_mix', pattern)
                for n in (24, 36) for pattern in ('reuse', 'stream64')}
    require({tuple(c) for c in original['cases']} == expected
            and len(original['summary']) == 4
            and all(v['all_three_pairs_gain']
                    and all(r > 1 for r in v['paired_reference_over_candidate'])
                    for v in original['summary']), 'all four independent parts must pass')
    plan = json.loads(cpu.PLAN.read_text())
    cpu.verify(plan)
    built = cpu.base.qualified_build(plan)
    checked = json.loads((cpu.ROOT / 'check.json').read_text())
    require(cpu.base.closed(checked, plan)
            and checked['binaries'] == built['binaries']
            and built['binaries']['pool']['sha256'] == original['binary']['sha256'],
            'same fresh CPU numeric/executable generation required')
    previous = coherent.base.validation.committed(RESULT, (PREVIOUS_KEY,))
    retirement = previous.get('artifact_retirement', {})
    require(retirement.get('stage') == 'retired_after_verified_original_source_recovery'
            and retirement.get('original_Git_sources_and_recipe_match_actual_final_inputs') is True
            and not PREVIOUS_ROOT.exists(),
            'retire the ended89be7temporary generation before another full build')
    return dict(coherent_class_basis=proof, CPU_evidence_commit=EVIDENCE,
        CPU_source_commit=original['source_commit'], CPU_plan_commit=original['plan_commit'],
        CPU_qualification_sha256=hashlib.sha256(fields(original).encode()).hexdigest(),
        CPU_binary=built['binaries']['pool'],
        retired_temporary_engine=PREVIOUS_KEY,
        scope='Down42parts only;original broad selector failure stays failed. '
              'Full model coverage,quality,prefill/TTFT preservation and speed remain unqualified.')


coherent.ROOT, coherent.KEY = ROOT, KEY
coherent.SEED = B / 'coherent-prefill-native-v1/record.json'
coherent.SEED_KEY = 'coherent_prefill_native_v1'
coherent.prerequisite = prerequisite


def prepare():
    coherent.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['plan'] = ('Fresh coherent full-engine build of the measured Down42-only CPU selection. '
        'Compile every affected core/generate/CPU/GGML translation unit;no object/archive reuse. '
        'Check complete source/generated/compiler/link inputs and every Gemm header consumer. '
        'Keep independent prefill/default-row and explicit pre-exec experimental decode selection. '
        'After build,commit bounded native/actual-model coverage/state and phase-preservation '
        'checks,then matched official long/agent/thinking long-generation comparisons. '
        'No GPU,model speed,task,ordinary teardown or physical262144/RESTORE qualification here.')
    row['retention'] = ('Keep14508clean GPUreference and this one temporary candidate;the ended '
        '89be7generation must already be retired after exact source/recipe recovery. '
        'Keep49296435CPUauxiliary only for this actual prerequisite until its consumer closes. '
        'LegacyH9/P5/D5reader/IR/deployment holds remain separate. Review actual paths and '
        'retire comparison bytes after validated selection;do not retain routine output.')
    coherent.context().save(row)


if __name__ == '__main__':
    require(sys.executable == cpu.hot.PYTHON and len(sys.argv) in (2, 3)
            and sys.argv[1] in ('prepare', 'configure', 'compile'),
            'qualified Python;prepare CPU_EVIDENCE_SHA|configure|compile')
    action = sys.argv[1]
    if action == 'prepare':
        require(len(sys.argv) == 3, 'prepare needs the committed CPU result SHA')
        EVIDENCE = sys.argv[2]
    else:
        require(len(sys.argv) == 2, 'configured stages use their original pinned evidence')
        EVIDENCE = json.loads((ROOT / 'record.json').read_text())[
            'build_prerequisite']['CPU_evidence_commit']
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        coherent.base.guards.no_model_owners([])
        prepare() if action == 'prepare' else coherent.context().build(action)
