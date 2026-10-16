#!/usr/bin/env python3
"""Dispatch eligible scope metadata after completed native Luna-max review."""
import hashlib
import json
from pathlib import Path
import subprocess
A = Path(__file__).resolve().parent
P = A.parents[2]
RUN = 'run_c8f478972ec8'


def main():
    review = A / 'research-available-scopes-round772-v1.json'
    assert hashlib.sha256(review.read_bytes()).hexdigest() == '462dca63707a9b27ca340ec7e381b85d630317e8ddd1c3d6979a175ee871b761'
    scopes = json.loads(review.read_text())['next_three_research_scopes']
    assert len(scopes) == 3 and all(v['schedulable_now'] is True and v['prerequisite_evidence'] for v in scopes), 'Do not launch unmet conditional research'
    preamble = (A / 'orca-research-round768-spec-v1.txt').read_text().split('\nScope metadata from reviewer767:\n')[0]
    preamble += '\nScope eligibility and redundancy reviewed by native Codex Luna-max R772. All scopes below are knowledge collection. Main has started a fixed78444-token FP32 NLL diagnostic; no long loss results, FP16 quality/admission or speed adoption yet. Do not read live state directories or inspect processes. Never read /home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/prefill-gemm-only-contract-root-v1/record.json (credential). Use sources/library only and do not infer new measurements.\n'
    for n, scope in zip(range(773,776), scopes):
        spec = A / f'orca-research-round{n}-spec-v1.txt'
        report = A / f'research-knowledge-round{n}-v1.txt'
        receipt = A / f'orca-research-round{n}-launch-v1.json'
        assert not spec.exists() and not report.exists() and not receipt.exists()
        spec.write_text(preamble + '\nScope metadata from reviewer772:\n' + json.dumps(scope,ensure_ascii=False,indent=2) + f'\nOutput only: {report}\n')
        p = subprocess.run(['orca','orchestration','worker-start','--spec',str(spec),'--worktree','current',
                            '--agent','codex','--model','gpt-6-luna','--run',RUN,'--json'], cwd=P,capture_output=True,text=True,timeout=90)
        r = dict(cli_exit_code=p.returncode,receipt=json.loads(p.stdout),stderr=p.stderr)
        receipt.write_text(json.dumps(r,indent=2)+'\n')
        assert p.returncode == 0 and r['receipt']['ok'], 'Inspect failed launch receipt; never duplicate a live dispatch'
        v = r['receipt']['result']
        print(json.dumps(dict(round=n,task=v['taskId'],dispatch=v['dispatchId'],effective=v['launch']['effective'],stage=v['stage'],eligible_before_launch=True)),flush=True)


if __name__ == '__main__':
    main()
