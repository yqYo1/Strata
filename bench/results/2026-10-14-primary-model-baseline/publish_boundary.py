#!/usr/bin/env python3
"""Preserve exact closed receipts and compact diagnostics; no log deletion."""
import fcntl
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path

P = Path('/home/yayoi/orca/workspaces/Strata/sycl')
B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
D = Path(__file__).parent
R = B / 'research-20261009'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    index = []
    roots = [f'main-primary-model-diagnostic-root-v{x}' for x in (1, 2, 4)]
    roots += [f'main-primary-size-t-range-build-root-v{x}' for x in (1, 2)]
    roots += ['main-primary-throw-observer-root-v1']
    previous = json.loads((D / 'receipts-index.json').read_text()) if (D / 'receipts-index.json').exists() else []
    def original(source):
        if source.exists():
            return source.read_bytes()
        matches = [entry for entry in previous if entry['original_path'] == str(source)]
        if not matches:
            return None
        data = gzip.decompress((D / matches[0]['path']).read_bytes())
        assert sha(data) == matches[0]['decoded_sha256']
        return data
    for optional in ['main-primary-size-t-range-gpu-root-v1', 'main-primary-model-diagnostic-root-v5',
                     'main-primary-post-deadline-gpu-root-v1', 'main-primary-model-diagnostic-root-v6',
                     'main-primary-accepted-mmq-gpu-root-v1', 'main-primary-model-accepted-diagnostic-root-v7',
                     'main-primary-stock-bench-accepted-root-v1', 'main-primary-post-abort-gpu-root-v1',
                     'main-primary-host-abort-observer-root-v1', 'main-primary-host-abort-observer-root-v2',
                     'main-primary-cold-jit-short-root-v8', 'main-primary-cold-jit-short-root-v9',
                     'main-primary-post-trace-cap-gpu-root-v1']:
        if original(B / optional / 'record.json') is not None:
            roots.append(optional)
    with (B / 'owned-v0141-measurement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for root in roots:
            output = B / root
            record = json.loads(original(output / 'record.json'))
            assert not record['active'], root
            for command in record['commands']:
                assert (command.get('exit_code') is not None
                    and command.get('direct_child_reaped') and command.get('session_empty')
                    and not command.get('survivors')), (root, command['label'])
                # Failed deadlines keep observation_complete=false. Publishing
                # their original evidence is not a successful lifecycle proof.
            for name in ['record.json', 'result.json', 'engine-exit.json', 'engine-start.json', 'control-action.json', 'config.json', 'plan.json', 'sampler-summary.json', 'stderr-relay-summary.json']:
                source = output / name
                data = original(source)
                if data is None:
                    continue
                target = D / 'receipts' / root / (name + '.gz')
                target.parent.mkdir(parents=True, exist_ok=True)
                compressed = gzip.compress(data, mtime=0)
                assert gzip.decompress(compressed) == data
                if target.exists():
                    assert gzip.decompress(target.read_bytes()) == data
                else:
                    target.write_bytes(compressed)
                index.append(dict(original_path=str(source), path=str(target.relative_to(D)),
                    decoded_bytes=len(data), decoded_sha256=sha(data), archive_sha256=sha(target.read_bytes())))
        prior_research = json.loads((D / 'research-index.json').read_text()) if (D / 'research-index.json').exists() else {}
        reports = []
        known_reports = [
            (459, 'research-primary-profiler-attribution-round459-v1.txt'),
            (460, 'research-low-vram-moe-scheduling-round460-v1.txt'),
            (461, 'research-primary-origin-reaudit-round461-v1.txt'),
            (462, 'research-primary-component-reaudit-round462-v1.txt'),
            (463, 'research-primary-large-range-contract-round463-v1.txt'),
            (464, 'research-primary-prior-cpu-rejections-round464-v1.txt'),
            (465, 'research-primary-native-function-provenance-round465-v1.txt'),
        ]
        known_rounds = {number for number, _ in known_reports}
        known_reports += [(entry['round'], Path(entry['path']).name) for entry in prior_research.get('reports', []) if entry['round'] not in known_rounds]
        for round_number, name in known_reports:
            target = D / 'research' / name
            data = (R / name).read_bytes() if (R / name).exists() else target.read_bytes()
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert target.read_bytes() == data
            else:
                target.write_bytes(data)
            reports.append(dict(round=round_number, path=str(target.relative_to(D)),
                original_path=str(R / name), sha256=sha(data), model='gpt-6-luna', read_only=True))
        prior_research.update(prior_indexes=['../2026-10-14-primary-native-runtime/research-index.json',
                           '../2026-10-14-niko-main-service-port/research-index.json'], reports=reports,
                           no_intervention_until_final=True)
        (D / 'research-index.json').write_text(json.dumps(prior_research, indent=2, ensure_ascii=False) + '\n')
        windows = json.loads((D / 'model-diagnostic-windows.json').read_text()) if (D / 'model-diagnostic-windows.json').exists() else []
        for version in (1, 2, 4, 5, 6, 7, 8, 9):
            root_name = f'main-primary-model-diagnostic-root-v{version}' if version != 7 else 'main-primary-model-accepted-diagnostic-root-v7'
            if version in (8, 9):
                root_name = f'main-primary-cold-jit-short-root-v{version}'
            source = B / root_name / 'owned/engine.stderr'
            if not source.exists():
                continue
            windows = [item for item in windows if item['version'] != version]
            lines = source.read_text().splitlines()
            indices = set()
            for i, line in enumerate(lines):
                if any(marker in line for marker in ['strata trace: request', 'strata trace: prompt start',
                    'strata trace: prompt chunk', 'strata serve: prompt ', 'strata prefill timing:',
                    'strata decode timing:', 'free after', 'free with everything', 'window up to', 'draft head over',
                    'capture 1 tokens', 'capture 4 tokens', 'no progress for', 'stall report', 'expert pool threads',
                    'memory:', 'released the verify', 'stack smashing', 'needs --spec', 'unknown argument', 'unknown option',
                    'terminate called', 'what():', 'strata diagnostic SYCL throw:']):
                    indices.update(range(max(0, i-1), min(len(lines), i+6)))
            # A remaining throw needs its whole backtrace, not just its signal.
            for i, line in enumerate(lines):
                if line.startswith('strata diagnostic SYCL throw:'):
                    indices.update(range(i, min(len(lines), i+55)))
            data = source.read_bytes()
            windows.append(dict(version=version, original_path=str(source), original_bytes=len(data),
                original_sha256=sha(data), original_lines=len(lines),
                selected_lines=[dict(line=i+1, text=lines[i]) for i in sorted(indices)],
                reason='distinct CLI/startup/range mechanism and relevant request/phase evidence'))
        (D / 'model-diagnostic-windows.json').write_text(json.dumps(windows, indent=2) + '\n')
        gpu = B / 'main-primary-size-t-range-gpu-root-v1'
        summaries = json.loads((D / 'gpu-diagnostic-summary.json').read_text()) if (D / 'gpu-diagnostic-summary.json').exists() else []
        for gpu in [gpu, B / 'main-primary-post-deadline-gpu-root-v1', B / 'main-primary-accepted-mmq-gpu-root-v1', B / 'main-primary-post-abort-gpu-root-v1', B / 'main-primary-post-trace-cap-gpu-root-v1']:
            for source in sorted((gpu / 'owned').glob('*debug.stderr')):
                summaries = [item for item in summaries if item['path'] != str(source)]
                data = source.read_bytes()
                text = data.decode()
                known = '<VALIDATION>[ERROR]: There are no valid references to handle 0'
                summaries.append(dict(path=str(source), bytes=len(data), sha256=sha(data),
                    UR_success_mentions=text.count('UR_RESULT_SUCCESS'),
                    ZE_success_mentions=text.count('ZE_RESULT_SUCCESS'),
                    UR_errors=sorted(set(re.findall(r'UR_RESULT_ERROR_[A-Z_]+', text))),
                    optional_null_pool_validation_count=text.count(known),
                    other_validation=[line for line in text.splitlines() if '<VALIDATION>[ERROR]' in line and line.strip() != known],
                    query_false_count=text.count('ERROR (2013265955) in zeCommandListIsGraphCaptureEnabledExt'),
                    phase_markers=[line for line in text.splitlines() if line.startswith(('runtime:', 'compat:'))]))
        (D / 'gpu-diagnostic-summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
        prior = subprocess.check_output(['git', '-C', str(P), 'show',
            'dc2ca4eee42c5870e11a1339c0fccc9fd9e856f6:bench/results/2026-10-04-sycl-cpu-single-pair/run.json'])
        prior_record = json.loads(prior)
        (D / 'prior-cpu-decision.json').write_text(json.dumps(dict(
            original_commit='dc2ca4eee42c5870e11a1339c0fccc9fd9e856f6',
            original_path='bench/results/2026-10-04-sycl-cpu-single-pair/run.json',
            original_sha256=sha(prior), original_status=prior_record['status'],
            workload=prior_record['workload'], hardware=prior_record['hardware'],
            IQ3_S_original_summary={key:value for key,value in prior_record['summary'].items() if key.startswith('21/')},
            current_decision='keep existing unpaired route; no new A/B merely from narrower synthetic mean; revisit only with documented compiler/code/route exposure change',
            current_toolchain_equivalence_to_old_run_proven=False), indent=2) + '\n')
        (D / 'receipts-index.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps(dict(closed_roots=len(roots), receipts=len(index), research=len(reports), raw_deleted=False)), flush=True)


if __name__ == '__main__':
    main()
