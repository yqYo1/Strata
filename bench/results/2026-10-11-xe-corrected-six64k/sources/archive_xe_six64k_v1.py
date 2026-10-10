import datetime, fcntl, hashlib, json, shutil, statistics, subprocess
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
R = B / 'research-20261009'
WS = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
WX = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-b570-prefill-publication-20261011')
A = WS / 'bench/results/2026-10-11-xe-corrected-six64k'
P = WS / 'bench/results/2026-10-11-parallel-round89'
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def run(args):
    q = subprocess.run(args, cwd=WS, env=ENV, capture_output=True, text=True, timeout=120)
    if q.returncode:
        raise RuntimeError(str(args[:4]) + ': ' + q.stderr[:500])
    return q.stdout.strip()

def require(value, message):
    if not value:
        raise RuntimeError(message)

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not run(['/usr/bin/git', 'status', '--porcelain']), 'archive worktree clean')
    require(not A.exists() and not P.exists(), 'immutable new archive paths')
    A.mkdir(); P.mkdir()
    copies = []; deletions = []; pending = []

    def copy(src, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        require(sha(src) == sha(dst), 'byte-identical archival copy')
        copies.append(dict(path=str(src), retained=str(dst.relative_to(WS)), bytes=src.stat().st_size, sha256=sha(src)))

    names = ['xe-prefill-long64k-serial-root-v1'] + [f'xe-corrected-clean64k-{a}-r{i}' for a in ['baseline', 'xe-corrected'] for i in range(1, 4)]
    records = {}
    for name in names:
        p = B / name / 'record.json'; d = json.loads(p.read_text())
        require(d['active'] is False and d['passed'] is True and d['complete'] is True, 'closed passed ' + name)
        require(d['model_result'] == dict(exit_code=0, owned_closed=True), 'closed model ' + name)
        require(all(c.get('session_empty') is True and c.get('direct_child_reaped') is True for c in d['commands']), 'closed children ' + name)
        require(d['health_after_passed'] and d['kernel_gate_passed'] and d['source_and_executable_stable'] and not d['new_fault_messages'], 'source/fault gate ' + name)
        copy(p, A / name / 'record.json'); records[name] = d
        evidence = []
        for c in d['commands']:
            label = c['label']
            for stream in ['stdout', 'stderr']:
                q = B / name / 'commands' / (label + '.' + stream)
                if not q.exists():
                    continue
                raw = q.read_bytes(); txt = raw.decode('utf-8')
                if label == 'model' and stream == 'stderr' and name in ['xe-prefill-long64k-serial-root-v1', 'xe-corrected-clean64k-baseline-r1', 'xe-corrected-clean64k-xe-corrected-r1']:
                    copy(q, A / name / 'model.stderr.txt')
                    replacement = str((A / name / 'model.stderr.txt').relative_to(WS))
                else:
                    if label.startswith('kernel-'):
                        excerpt = json.dumps(d.get('kernel_device_entries', d.get('kernel_preflight_relevant_entries', [])), ensure_ascii=False)
                    elif label.startswith('health'):
                        excerpt = '\n'.join(v for v in txt.splitlines() if v.startswith(('STATUS ', 'QUEUE ', 'H2D ', 'KERNEL ', 'D2H ', 'PASS ')) or any(x in v for x in ['[ERROR]', '[FATAL]', 'UR_RESULT_ERROR_', 'ZE_RESULT_ERROR_']))
                    elif label == 'model' and stream == 'stdout':
                        # Full T/LP/PP/DONE, startup and validation are retained in record.json.
                        excerpt = '\n'.join(v for v in txt.splitlines() if not v.startswith(('T ', 'LP ', 'PP ', 'DONE ', 'RESUME ', 'REUSED ', 'READY', 'INFO ', 'HELLO ', 'VOCAB ')))
                        require(len(excerpt) < 16384, 'bounded extra protocol evidence')
                    elif label == 'model':
                        excerpt = '\n'.join(v for v in txt.splitlines() if any(x in v.lower() for x in ['error', 'warning', 'failed', 'cache', 'free', 'ready']))
                    else:
                        excerpt = txt[:16384]
                    evidence.append(dict(label=label, stream=stream, bytes=len(raw), sha256=sha(q), excerpt=excerpt))
                    replacement = str((A / name / 'command-evidence.json').relative_to(WS))
                pending.append((q, replacement, 'Closed successful stream; individual result, recipe, ownership, protocol and relevant diagnostics verified in committed archival evidence.'))
        (A / name / 'command-evidence.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + '\n')
        token_file = B / name / 'tokens.txt'
        if token_file.exists():
            require(sha(token_file) == d['input']['sha256'], 'original measured input hash')
            pending.append((token_file, str((A / name / 'record.json').relative_to(WS)), 'Duplicate run-local comma token serialization; original fixture and generating controller retained as current consumer.'))

    def protocol(d):
        return [s for s in d['request_protocol'] if s.startswith(('T ', 'LP '))]

    summary = dict(execution_order=['baseline-r1', 'xe-corrected-r1', 'xe-corrected-r2', 'baseline-r2', 'baseline-r3', 'xe-corrected-r3'], arms={}, paired_ratios=[], adopted=False, math_equivalence=False, quality_qualified=False, full262144_qualified=False, fork_pristine=False)
    for arm in ['baseline', 'xe-corrected']:
        ds = [records[f'xe-corrected-clean64k-{arm}-r{i}'] for i in range(1, 4)]
        require(all(protocol(d) == protocol(ds[0]) for d in ds), 'within-arm 64 token IDs and LP5 identical')
        result = dict(samples=[d['sample'] for d in ds], within_arm_token_and_lp5_identical=True)
        for phase in ['prefill', 'decode']:
            vals = [d['sample'][phase + '_tps'] for d in ds]
            result[phase] = dict(median_tps=statistics.median(vals), mean_tps=statistics.mean(vals), min_tps=min(vals), max_tps=max(vals), sample_stddev_tps=statistics.stdev(vals))
        summary['arms'][arm] = result
    for i in range(1, 4):
        left = records[f'xe-corrected-clean64k-baseline-r{i}']['sample']; right = records[f'xe-corrected-clean64k-xe-corrected-r{i}']['sample']
        summary['paired_ratios'].append(dict(repetition=i, prefill_xe_over_baseline=right['prefill_tps']/left['prefill_tps'], decode_xe_over_baseline=right['decode_tps']/left['decode_tps']))
    for phase in ['prefill', 'decode']:
        summary[phase + '_median_ratio'] = summary['arms']['xe-corrected'][phase]['median_tps'] / summary['arms']['baseline'][phase]['median_tps']
    baseline = protocol(records['xe-corrected-clean64k-baseline-r1']); xe = protocol(records['xe-corrected-clean64k-xe-corrected-r1'])
    bi = [v for v in baseline if v.startswith('T ')]; xi = [v for v in xe if v.startswith('T ')]
    prefix = next((i for i, (x,y) in enumerate(zip(bi,xi)) if x != y), 64)
    summary.update(cross_arm_common_greedy_prefix=prefix, comparison_scope='Whole configured engines: baseline slots128/ring8/pageable versus corrected Xe slots144/ring16/host-USM/issuer OFF. 65536 actual input, 4096 chunks, 64 outputs, no reuse, same fixture and requested settings. Different generated paths and MTP acceptance confound causal decode attribution.', old_8k_403tps_scope='Historical one-sample baseline used 811614 fixture and 8K chunk; excluded from this new 5284 fixture /4K comparison.', newer_xe_tip='e32b8b0594095352365db0e140a68d5bf56d8b0f', newer_xe_pending='40 commits/140 files, prefill stock SHA unchanged; do not execute known faulting pristine source. Latest doorbell volatile/fence contract requires new capability/path qualification before GPU.', official_next='v0.1.42 tag and main both 61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406; after latest Xe assessment. All server features mandatory, prefill/decode inference adoption independent.')
    (A / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    for name in ['xe-corrected-clean64k-comparison-v1', 'xe-prefill-long64k-serial-v1']:
        for q in (B / name).iterdir():
            if q.is_file() and q.suffix in ['.py', '.json']:
                copy(q, A / 'sources' / name / q.name)
    copy(B / 'xestrata-clean-64k-comparison-v4/direct_owner.py', A / 'sources/direct_owner-imported.py')
    copy(Path(__file__), A / 'sources/archive_xe_six64k_v1.py')
    additions = []
    for i in range(329, 344):
        matches = list(R.glob(f'round{i}-*.txt')); require(len(matches) == 1, 'unique report R' + str(i))
        q = matches[0]; copy(q, P / q.name)
        additions.append(dict(path=str(q), report=str(q), sha256=sha(q), bytes=q.stat().st_size, model='gpt-6-luna', read_only=True, scope=q.name))
    registry = json.loads((R / 'report-registry-v111.json').read_text())
    known = {v['sha256'] for v in registry['reports']}; require(all(v['sha256'] not in known for v in additions), 'no repeated registered reports')
    registry['reports'] += additions
    registry.update(registry_version=112, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), research_completed=len(registry['reports']), new_reports=additions, previous_committed_registry=dict(path='bench/results/2026-10-11-parallel-round88/report-registry-v111.json', commit='8f3e44f8'), live_agent_snapshot=dict(running_read_only=[], completed=['R342 all doorbell paths', 'R343 official server tests']))
    registry['root_review_corrections_round89'] = {
        'R330': '403.54 historical baseline used OLD 811614 fixture, not NEW 5284. Actual quiet comparison uses 4K chunks both arms; corrected 8K not qualified and raw8K free50MiB rejected.',
        'R331': 'Current engine WT HEAD7d0105; docs archive HEAD8f3e44. Official release tag/main both61b3fb5, not postrelease delta cdbf458. Optional replicas/keepawake runtime availability remains mandatory integration.',
        'R332': 'Main DOES contain SYCL; stale official mirror caused false absence claim. R333 full v0.1.41-to-v0.1.42 delta supersedes narrow latest CUDA commit-only conclusion.',
        'R336': 'GEN1 does NOT append its predicted output to live consumed tokens. R337 validated source counterexample: successive external reference token prefixes can reuse full live input without batched reread. LP20 still not full-vocabulary KL or ability proof; actual vocab248320.',
        'R335': 'Existing R293 actual IQ2_S GU22 is largest observed mixed-pair caller; IQ3_S-only MT1 is not a new all-model solution. Six measurements do not assign bottleneck/causal speedup.',
        'R340': 'Atomic host allocation and system scope capabilities not yet queried on this B570. New volatile+fence is no portable atomic/happens-before proof; all native/MTP/session paths need contract audit. No attribution of historical GPU fault.',
        'R341': 'Full latest source/dependency rebuild required; prior prefill-TU-only v7 not latest qualification. Free DPC++ unavailable and original full suite missing CMake; exact available tests must be recorded.',
        'R343': 'Report repeats stale8f3e44 engine HEAD; actual engine WT7d0105 and archive8f3e44. Proposed upstream test filenames and fixture CPU-only status require direct verification after fetching pinned official61. Do not claim any test execution.'}
    (R / 'report-registry-v112.json').write_text(json.dumps(registry, indent=2, ensure_ascii=False) + '\n')
    copy(R / 'report-registry-v112.json', P / 'report-registry-v112.json')
    (P / 'RESULTS_JA.md').write_text('読み取り専用Luna R329〜343の15件を原文保存。registry v112は454件。R330の入力fixture混同、R331/R343のversion/HEAD、R332のSYCL不在、R336のGEN1 live-prefix判断は別項目で訂正し、原レポートは変更しない。R342は全doorbell経路の直列fallback、R343は公式版サーバーのGPU不要統合試験の提案。直接source/fixture確認と実行資格はメイン担当。\n')
    (A / 'RESULTS_JA.md').write_text('旧39bd由来Xeにpublication/lifetime対策を入れた実64K入力・4Kchunk・64出力のquiet比較を各3回完了。実行順B1/X1/X2/B2/B3/X3。全6件exit0、直前直後の16384整数一致、new fault/reset/dumpなし、source/binary安定。各arm内64tokenとLP5は3回同一。\n\n中央値はbaseline prefill201.33 / decode8.86 tok/s、修正Xe prefill205.20 / decode10.73 tok/s。prefill paired差は+0.28%/+5.32%/-1.55%で一貫しない。decodeは各pair+31.08%/+20.97%/+18.93%で有利な信号。ただしcache/ring/host USM等が異なり、arm間の出力は5token以降分岐、MTP37/78対40/72。実装単位の速度差であり、特定同期やCPU最適化の寄与、能力同等、採用を証明しない。全個別値と範囲はsummary.jsonに保存。\n\n長入力診断1件はLevel Zero/UR loggingありのため速度比較不適格。256Kは要求容量だけでfull262144の使用/restore証明ではなく、全候補未採用。従来403.54 tok/sのbaselineは別入力fixture/8Kchunkの1件で今回から除外。\n\n最新Xe e32b8b05は40commit更新。stock prefillはfaultした39bdと同一なので未対策再実行を拒否し、対策を移植して全source/dependencyを新規buildする。新GPU/host doorbellのvolatile+fence同期は、実B570 capabilityと全経路fallbackを確認・対策後にGPU実行。公式v0.1.42はその後、未変更版と統合版を別測定。サーバー全機能を取り込み、推論変更のprefill/decode採否は個別に判断する。\n')
    # Exact captured source/report whitespace is evidence, not new project formatting.
    for directory in [A, P]:
        (directory / '.gitattributes').write_text('sources/** -text -whitespace\nround*.txt -text -whitespace\n')
        q = subprocess.run([str(WX / '.lint/bin/gitleaks'), 'detect', '--no-git', '--redact', '--no-banner', '--source', str(directory)], cwd=WS, env=ENV, capture_output=True, text=True, timeout=60)
        require(q.returncode == 0, 'secret scan ' + directory.name)
    for q, replacement, reason in pending:
        require((WS / replacement).exists(), 'verified retained evidence')
        st = q.stat()
        deletions.append(dict(path=str(q), bytes=st.st_size, allocated_bytes=st.st_blocks*512, sha256=sha(q), retained_replacement=replacement, reason=reason))
    (A / 'retention-manifest.json').write_text(json.dumps(dict(original_receipts_unchanged=True, copies=copies, deletions=deletions, removed_logical_bytes=sum(v['bytes'] for v in deletions), removed_file_allocation_bytes=sum(v['allocated_bytes'] for v in deletions), actual_available_space_reclaimed_not_measured=True, canonical_fixtures_retained=['clean64k-task-fixture-root-r3/coding-review-65536-tokens.txt'], raw_tensor_capture_created=False), indent=2, ensure_ascii=False) + '\n')
    run(['/usr/bin/git', 'add', '-f', '--', str(A.relative_to(WS)), str(P.relative_to(WS))])
    run(['/usr/bin/git', 'diff', '--cached', '--check'])
    for q, _, _ in pending:
        q.unlink()
    run(['/usr/bin/git', 'commit', '-m', 'docs: record six matched 64K Xe samples and updated fork review'])
    run(['/usr/bin/git', 'push'])
    print(json.dumps(dict(commit=run(['/usr/bin/git', 'rev-parse', 'HEAD']), reports=len(registry['reports']), removed_logical_bytes=sum(v['bytes'] for v in deletions), summary=summary), ensure_ascii=False))
