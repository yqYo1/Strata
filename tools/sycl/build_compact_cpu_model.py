#!/usr/bin/env python3
"""Fresh full-engine qualification of the measured CPU decode selection.

Main-authored adapter to this fork's coherent build controller; repository LICENSE.
No project object reuse, GPU execution or inherited model-performance claim.
"""
import fcntl
import hashlib
import json
import sys

sys.dont_write_bytecode = True
import build_coherent_prefill as coherent
import cpu_compact_pool as cpu

P, B = coherent.P, coherent.B
ROOT = B / 'compact-cpu-model-native-v1'
KEY = 'compact_cpu_model_native_v1'
CPU_RESULT = P / 'bench/results/2026-10-15-pr6-recovery-audit/cpu-roofline-current.json'
CPU_KEY = 'compact_small_batch_pool_v1'
CPU_EVIDENCE = '07ea14c4c623c73505d5fc863dbb9231c6e252d5'
CORE_SOURCE = '244c46b97f9edc77208b4b61dc820d4de2de697c'
coherence_prerequisite = coherent.prerequisite


def prerequisite():
    proof = coherence_prerequisite()
    raw = coherent.base.common.source.git(P, 'show', 'HEAD:' + str(CPU_RESULT.relative_to(P)))
    coherent.require(raw == CPU_RESULT.read_bytes(), 'commit exact CPU qualification evidence')
    value = json.loads(raw)[CPU_KEY]
    original = json.loads(coherent.base.common.source.git(P, 'show',
        CPU_EVIDENCE + ':' + str(CPU_RESULT.relative_to(P))))[CPU_KEY]
    coherent.require(value['summary'] == original['summary'] and value['numerical_qualified']
        and value['complete'] and value['ownership_closed'] and not value['active'],
        'preserve original closed CPU samples and failed shapes')
    selected = [r for r in value['summary'] if r['workers'] == 5 and r['host_works']
        and r['jobs_per_call'] in (3, 6)]
    coherent.require(len(selected) == 8 and all(r['cached_eligible']
        and r['all_three_pairs_gain'] and all(x > 1 for x in r['paired_reference_over_candidate'])
        for r in selected), 'all eight selected CPU shapes must pass their original gates')
    plan = json.loads(cpu.PLAN.read_text())
    cpu.verify(plan)
    built = cpu.qualified_build(plan)
    checked = json.loads((cpu.ROOT / 'check.json').read_text())
    coherent.require(cpu.closed(checked, plan) and checked['binaries'] == built['binaries']
        == value['binaries'], 'same coherent CPU numerical generation required')
    name = 'src/core/expert_source.cpp'
    source = coherent.base.common.source.git(P, 'show', CORE_SOURCE + ':' + name)
    coherent.require(hashlib.sha256(source).hexdigest() == coherent.sha(P / name),
        'production selection differs from the committed 3/6-job proposal')
    return dict(coherent_class_basis=proof, CPU_evidence_commit=CPU_EVIDENCE,
        CPU_source_commit=value['source_commit'], CPU_plan_commit=value['plan_commit'],
        CPU_binary=built['binaries'], selected_cells=selected,
        core_proposal_source=CORE_SOURCE, core_sha256=coherent.sha(P / name),
        metadata_frozen=str(CPU_RESULT.relative_to(P)),
        scope='Eight passing CPU shapes only;no blanket <=6 policy or model adoption. '
              'Actual core/generate,allocation,RAM,TTFT,prefill and model checks remain pending.')


coherent.ROOT, coherent.KEY = ROOT, KEY
coherent.SEED = B / 'coherent-prefill-native-v1/record.json'
coherent.SEED_KEY = 'coherent_prefill_native_v1'
coherent.prerequisite = prerequisite


def prepare():
    coherent.prepare()
    row = json.loads((ROOT / 'record.json').read_text())
    row['plan'] = ('Fresh full source build of the actual 3/6-job,5worker+host CPU selection '
        'and allocation repair,including every affected core/generate/CPU/GGML translation unit. '
        'Preserve selected14508/dcb8 GPU fallback and current CPU fixtures. '
        'Check complete compiler/link/generated inputs and Gemm definition census. '
        'No partial archives,GPU execution,model qualification or inherited speed. '
        'After build,commit bounded actual-dispatch/numerical/safety checks before model execution; '
        'then matched32K+thinking long-generation,RAM/TTFT and prefill-preservation evidence. '
        'Freeze CPU_RESULT and native build RESULT through this build.')
    row['retention'] = ('One new full-engine candidate for named compact-CPU model qualification; '
        'keep validated14508/dcb8 engine and c0dc/c858 CPU auxiliary. '
        'No old CPUarchive is consumed. Legacy H9/P5/D5 reader/IR/deployment source-recovery '
        'holds remain separate and are not newly qualified. Select validated engine and retire '
        'ended comparison outputs after model comparison and original-input recovery.')
    coherent.context().save(row)


if __name__ == '__main__':
    coherent.require(sys.executable == cpu.hot.PYTHON and len(sys.argv) == 2
        and sys.argv[1] in ('prepare', 'configure', 'compile'), 'qualified Python;prepare|configure|compile')
    with (B / 'owned-v0141-measurement.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        coherent.base.guards.no_model_owners([])
        prepare() if sys.argv[1] == 'prepare' else coherent.context().build(sys.argv[1])
