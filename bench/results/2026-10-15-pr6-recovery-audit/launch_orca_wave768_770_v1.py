#!/usr/bin/env python3
"""Dispatch fresh research from reviewer metadata without reading archive bodies."""
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
A = Path(__file__).resolve().parent
P = A.parents[2]
RUN = 'run_c8f478972ec8'


def main():
    review = json.loads((A / 'research-scope-review-round767-v1.json').read_text())
    scopes = review['next_three_research_scopes']
    assert len(scopes) == 3
    # These are assignment metadata, not a Main read of the review/report bodies.
    common = f'''Purpose: knowledge collection. Existing project {P}, primary feature/sycl.
Read AGENTS.md, docs/ARTIFACT_RETENTION.md and {A}/research-index.json first.
Read the relevant previous reports and Main corrections to avoid duplication;
Main does not read collection report bodies on arrival. A native Codex Luna-max
reviewer will assess depth/novelty/redundancy, pin hashes and index retrieval metadata.
This is a fresh bounded read-only session. Do not spawn children. Write only your
named report. No product edits, Git commands, builds, tests, GPU/profiler jobs,
process inspection, cleanup, services or environment changes. Main continues
independent work; do not request steering. No XeStrata source, helpers or archives.
Use primary documentation, papers and original technical blogs; at most six
substantive sources. Search current sources, distinguish measured facts from
inference, and identify device/version/license limits and counterevidence.
Do not repeat generic background. Compare alternatives deeply enough to be useful;
the budget is a working limit, not a demand for filler or a tiny fact lookup.
Return a file report with precise URLs, applicability limits, new findings and
one concrete future experiment. In the final response supply only the report
path, its hash and a <=150-character operational completion notice, without
technical findings. Report completion with your live exact Task/Dispatch via
Orca worker_done, list filesModified, then end. Do not message Main findings.
Current evidence: clean PP349.3 median versus control339.2, not >600/1000 yet.
Actual 78452-row CPU diagnostic: main218.681 CPU s/wall225.874; explicit waits
165.704 CPU s; residual host copy call CPU25.793s, versus hostgroup.19649s.
Route-ID exchange inclusive131.610wall includes previous GPU work and is not
removable pure copy latency. Recent idle 320MiB transfer component: ordinary
H2D6.015GB/s versus HostUSM5.095GB/s; USM faster submission but slower completion.
Blanket pinning rejected; no model end-to-end claim. FP16 residual remains
unqualified for quality/admission. Default-OFF full-vocab NLL observer now built,
host tested and synthetic129-input paired-GPU tested, fault0; this is not model
quality evidence. Preserve the existing FP32 path and physical262144 requirement.
'''
    for n, scope in zip(range(768, 771), scopes):
        spec = A / f'orca-research-round{n}-spec-v1.txt'
        report = A / f'research-knowledge-round{n}-v1.txt'
        receipt = A / f'orca-research-round{n}-launch-v1.json'
        assert not spec.exists() and not report.exists() and not receipt.exists()
        body = common + '\nScope metadata from reviewer767:\n' + json.dumps(scope, ensure_ascii=False, indent=2) + f'\nOutput only: {report}\n'
        spec.write_text(body)
        command = ['orca','orchestration','worker-start','--spec',str(spec),
                   '--worktree','current','--agent','codex','--model','gpt-6-luna',
                   '--run',RUN,'--json']
        run = subprocess.run(command, cwd=P, capture_output=True, text=True, timeout=90)
        result = dict(cli_exit_code=run.returncode, receipt=json.loads(run.stdout), stderr=run.stderr)
        receipt.write_text(json.dumps(result, indent=2) + '\n')
        r = result['receipt']
        assert run.returncode == 0 and r['ok'], f'Inspect failed dispatch {receipt}; do not duplicate'
        v = r['result']
        print(json.dumps(dict(round=n,task=v['taskId'],dispatch=v['dispatchId'],
                              effective=v['launch']['effective'],stage=v['stage'])), flush=True)


if __name__ == '__main__':
    main()
