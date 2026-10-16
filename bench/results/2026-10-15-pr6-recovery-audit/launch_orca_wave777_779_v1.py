#!/usr/bin/env python3
"""Two eligible reviewed scopes plus one bounded broader memory-layout question."""
import hashlib
import json
from pathlib import Path
import subprocess
A = Path(__file__).resolve().parent
P = A.parents[2]
RUN = 'run_c8f478972ec8'


def main():
    review = A / 'research-scope-review-round776-v1.json'
    assert hashlib.sha256(review.read_bytes()).hexdigest() == '9b9d5dd22847830d2b602a559b921e8c987db57665348b8a060e131421400716'
    scopes = [v for v in json.loads(review.read_text())['next_three_research_scopes'] if v['schedulable_now'] is True]
    assert len(scopes) == 2 and all(v['prerequisite_evidence'] for v in scopes)
    scopes.append(dict(priority=3,purpose_label='knowledge collection',scope_origin='Main finite question, no unmet local measurement prerequisite',
        title='Memory transactions and address-translation limits in irregular resident GPU gathers',
        finite_question='For large irregular gathers, when does changing device tensor layout/tiling outperform adding concurrent workgroups, and how can published memory-transaction and TLB/working-set evidence distinguish the mechanisms? Compare two or more useful approaches (bucketing/tile reuse, coalescing/subgroup redistribution, prefetch or page/working-set reduction) and conditions where they fail. Use a substantive non-LLM graph/scientific/ASIC case as well as Intel or portable SYCL applicability. Do not equate low observed bandwidth with a PCIe bottleneck.',
        breadth_and_counterevidence='One coherent investigation of irregular device-memory access efficiency, rather than a list of generic GPU tips. Distinguish transaction waste, translation misses, dependency latency and cache reuse; separate observations on other GPUs/ASICs from inference for Arc B570. Do not repeat earlier scheduling/liveness/pipeline reports unless the new memory-transaction evidence changes the conclusion.',
        report_budget='About1200-1500 words, at most6 substantive primary/original-author sources, one concrete future experiment with discriminating counters/outcomes.',
        schedulable_now=True,prerequisites='Public-source/library access only, no local process/GPU/profiler inspection.',
        prerequisite_evidence='Sources and prior index available; broader transferable non-LLM GPU/ASIC collection explicitly requested. No local empirical result is a prerequisite to this literature question.',
        prior_report_paths_to_avoid_repeating=['research-nonllm-gpu-round759-v1.txt','research-nonllm-pipeline-residency-round763-v1.txt','research-gpu-liverange-rematerialization-round766-v1.txt','research-knowledge-round773-v1.txt']))
    preamble = (A / 'orca-research-round768-spec-v1.txt').read_text().split('\nScope metadata from reviewer767:\n')[0]
    # Exact evidence/version pins for the decision-support scope. Collection
    # report bodies are never printed or loaded into Main's model context.
    closed = A / 'qsa-subprofile-model-v2.json'
    analysis = A / 'qsa-subprofile-analysis-v2.json'
    build = json.loads((A / 'own-dense-prefill-nll-build-v3.json').read_text())
    pins = dict(historical_closed_receipt=closed.name,historical_closed_sha256=hashlib.sha256(closed.read_bytes()).hexdigest(),
                historical_analysis=analysis.name,historical_analysis_sha256=hashlib.sha256(analysis.read_bytes()).hexdigest(),
                historical_binary_sha256=json.loads(analysis.read_text())['binary_sha256'],
                current_binary_sha256=build['binary_sha256'],current_source_snapshot=build['source_pins'],
                current_build_receipt='own-dense-prefill-nll-build-v3.json',runtime_file_receipt='own-qsa-select-model-a1-library-hashes-v1.json',
                version_limit='Installed runtime/IGC files verified stable before live FP32 control; historical loaded-process proof is not a new map/counter measurement. No local inspection allowed.',
                existing_event_method='Own compat EventRecord submits queue barrier; elapsed uses event profiling command_end timestamps. QSA observer seven section pairs; intervals include markers/host gaps and are not individual kernel counters.')
    for n, scope in zip(range(777,780), scopes):
        spec = A / f'orca-research-round{n}-spec-v1.txt'
        report = A / f'research-round{n}-v1.txt'
        receipt = A / f'orca-research-round{n}-launch-v1.json'
        assert not spec.exists() and not report.exists() and not receipt.exists()
        body = preamble.replace('Purpose: knowledge collection.', 'Purpose: '+scope['purpose_label']+'.')
        body += '\nCurrent FP32-a78444-token NLL diagnostic still LIVE; no new long-quality or speed conclusion. Do not inspect live state directories. Safety flags pinned preexec: UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1,EnableDirectSubmission=0,samequeue1. No reset/env mutation/build/tests.\n'
        body += '\nScope eligibility metadata:\n' + json.dumps(scope,ensure_ascii=False,indent=2)
        if scope['purpose_label'] == 'decision support':
            body += '\nExact source/version/evidence pins:\n' + json.dumps(pins,indent=2)
        body += f'\nOutput only: {report}\n'
        spec.write_text(body)
        p = subprocess.run(['orca','orchestration','worker-start','--spec',str(spec),'--worktree','current',
                            '--agent','codex','--model','gpt-6-luna','--run',RUN,'--json'],cwd=P,capture_output=True,text=True,timeout=90)
        r = dict(cli_exit_code=p.returncode,receipt=json.loads(p.stdout),stderr=p.stderr)
        receipt.write_text(json.dumps(r,indent=2)+'\n')
        assert p.returncode == 0 and r['receipt']['ok'], 'Inspect failed dispatch; do not duplicate a live worker'
        v = r['receipt']['result']
        print(json.dumps(dict(round=n,task=v['taskId'],dispatch=v['dispatchId'],effective=v['launch']['effective'],stage=v['stage'],purpose=scope['purpose_label'])),flush=True)


if __name__ == '__main__':
    main()
