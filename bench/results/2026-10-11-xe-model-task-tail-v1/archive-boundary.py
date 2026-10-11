"""Archive closed qualification and source-only next candidate, without GPU work."""
import ast
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
X = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
OWNER = B / 'xe-llvm-v711-source-v5/direct_owner.py'
s = importlib.util.spec_from_file_location('owner', OWNER)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
m.require(m.sha(OWNER) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'owner identity')

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    d = W / 'bench/results/2026-10-11-xe-model-task-tail-v1'
    q = W / 'bench/results/2026-10-11-parallel-round98'
    m.require(not d.exists() and not q.exists(), 'fresh archive')
    d.mkdir()
    q.mkdir()
    copies = []

    def copy(src, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        m.require(m.sha(src) == m.sha(dest), 'exact archive ' + str(dest))
        copies.append(dict(source=str(src), archive=str(dest.relative_to(W)), sha256=m.sha(src), bytes=src.stat().st_size))

    def put(dest, value):
        dest.write_text(json.dumps(value, indent=2) + '\n')

    receipts = [
        ('xe-source-llvm-model8k-root-v3/record.json', 'model-v3-failed.json', False),
        ('xe-source-llvm-model8k-cpu-root-v4/record.json', 'cpu-v4.json', True),
        ('xe-source-llvm-model8k-cpu-root-v5/record.json', 'cpu-v5.json', True),
        ('xe-e32-lints-baseline-compare-root-v5/record.json', 'lint-v5.json', True),
    ]
    for src, name, passed in receipts:
        r = json.loads((B / src).read_text())
        m.require(not r['active'] and r['passed'] is passed, 'closed original status ' + src)
        m.require(all(c.get('direct_child_reaped') and c.get('session_empty') and not c.get('survivors') for c in r['commands']), 'closed owners ' + src)
        copy(B / src, d / name)
    for v in (3, 4):
        for p in (B / ('xe-source-llvm-model8k-v' + str(v))).glob('*.py'):
            copy(p, d / ('controller-v' + str(v)) / p.name)
    copy(B / 'xe-source-llvm-model8k-cpu-v5/run_root.py', d / 'cpu-controller-v5.py')
    copy(B / 'xe-source-llvm-model8k-root-v3/commands/model.stderr', d / 'model-v3-progress-stderr.txt')
    copy(B / 'xe-serialized-serve-source-commit-v1.json', d / 'source-commit.json')
    v3 = json.loads((d / 'model-v3-failed.json').read_text())
    health = json.loads((B / 'xe-source-llvm-small-health-root-v3/record.json').read_text())
    mapping = [dict(path=x['path'], expected_sha256=x['sha256'], actual_sha256=v3['observed_libraries'].get(x['path'], {}).get('sha256'), matched=v3['observed_libraries'].get(x['path'], {}).get('sha256') == x['sha256']) for x in health['loaded_libraries']]
    m.require(all(x['matched'] for x in mapping), 'v3 actual mapped runtime equality')
    put(d / 'analysis.json', dict(created_utc=m.utc(), source_revision=v3['candidate_revision'], original_model_status='FAILED; observer demanded MTP coverage before QUIT; preserved SIGTERM -15', actual_progress=v3['validation'], mapped_runtime_comparison=mapping, mapping_scope='Independently checked captured v3 maps against prior qualified health identities; no post-model health probe occurred in v3.', kernel_gate_passed=v3['kernel_gate_passed'], new_faults=v3['new_fault_messages'], selected_profile_fill=dict(allocated=144, filled=29, residents=29, first_lookup_hits=4, first_lookup_total=480), input_issue='First8192 prefix omits the independently verified49-token task/chat suffix. A possible cause of early EOS, not yet experimentally established.', next_test='Unchanged51436a0 source, exactly8192 tokens from first8143+last49; close QUIT normally before judging MTP coverage. v4 not executed at this archive boundary.', performance_eligible=False, full_model_quality_qualified=False, full262144_qualified=False, remote_refresh=dict(repository='MistVVK/XeStrata', method='git fetch --quiet origin', observed_utc='2026-10-11T03:36:00+00:00', exit_code=0, remote_HEAD='e32b8b0594095352365db0e140a68d5bf56d8b0f', commits_after_current=0)))
    stage = B / 'xe-per-layer-profile-fill-source-v1'
    report = B / 'research-20261009/implementation-xe-per-layer-profile-fill-v1.txt'
    expected = {'per-layer-profile-fill.patch':'73750b009ad0edd1970d1c7cb4b91cd43d2adea9b82762921d8223d64d978431', 'manifest.json':'33a9fbcdfc907cd46a050af682fa428fe98ba2c1044c359630459bf5d44907a5'}
    for name, digest in expected.items():
        m.require(m.sha(stage / name) == digest, 'returned source artifact')
        copy(stage / name, q / name)
    m.require(m.sha(report) == '4ec3aafac53ef6a4074981d1467a7ad3bc1b359ba9a49fe2af5cbfff39e3b254', 'returned report')
    copy(report, q / report.name)
    previous = W / 'bench/results/2026-10-11-parallel-round97/report-registry-v120.json'
    registry = dict(registry_version=121, created_utc=m.utc(), registry_format='incremental; follow previous_committed_registry for earlier reports', research_completed=490, previous_committed_registry=dict(path=str(previous.relative_to(W)), sha256=m.sha(previous), reports=489), reports=[dict(path=str(report), archive=str((q/report.name).relative_to(W)), sha256=m.sha(report), bytes=report.stat().st_size, model='gpt-6.1-sol', read_only=False, qualification='Full report and patch read by Main; independent source review pending; no build/lint/model qualification or Main application')], new_reports=[report.name], recurring_research_policy='Repeated distinct read-only Luna scopes with shared past reports; no mid-task intervention. Actual service quota error prevents Luna until Oct14 2026 12:33PM; no model substitution.', live_agent_snapshot=dict(observed_utc=m.utc(), implementation_agent='completed source-only', research_agents='0 active; actual quota error remains', Main='Admission correction source514 qualified build/host-ASan/lint; modelv3 reachedREADY then earlyEOS and originalFAILED. CPUv5 passed; unchanged-source complete-tail v4 pending.'), root_review_corrections=dict(report_input_causality='Original implementer calls the missing ending the reason for replacement. Main treats it as a hypothesis only: no v4 result yet.', current_candidate='Per-layer fill may change CPU/GPU accumulation routing; no numerical equivalence or performance qualification.'))
    put(q / 'report-registry-v121.json', registry)
    (d/'RESULTS_JA.md').write_text('# 受付修正後のモデル実行と入力末尾の確認\n\nsource51436a0の受付修正は両モードのbuild・host ASan/UBSan・lint比較を通過した。lintは既存の83件を各モードで維持し、新規findingは0。元run.shの非zero exitはreceiptに保存した。\n\n[前回の実行](model-v3-failed.json)はREADYに到達し、4096と8191位置までの2チャンクを完了したが、停止トークン1個、MTP draft0で終了した。監督側のcoverage判定がQUIT前に失敗し、所有プロセスをSIGTERMで回収した。元FAILEDとexit-15は変更しない。[独立解析](analysis.json)は実行時library8個の一致、新規GPU障害0、144枠中29枠のfillを確認した。8K診断は長文性能・品質・262144全域の検証ではない。\n\n[次のcontroller](controller-v4/run_root.py)は依頼文と会話の49-token末尾を残して8192入力を作り、完了済みのリクエストをQUITで正常終了させてからMTP coverageを判定する。[CPU検査](cpu-v5.json)はPASS。モデルv4はこの時点で未実行で、末尾欠落とEOSの因果関係は未確定。\n\n[次のキャッシュ修正](../2026-10-11-parallel-round98/RESULTS_JA.md)は別実装agentから戻ったsource-only候補。現行sourceのv4が閉じてからMainが統合・検証する。元の失敗と今回の修正を混同しない。\n')
    (q/'RESULTS_JA.md').write_text('# per-layer profile fillの修正候補\n\n[報告](implementation-xe-per-layer-profile-fill-v1.txt)と[patch](per-layer-profile-fill.patch)をMainで全文確認した。1レイヤーの満杯で全fillを打ち切る処理を、有限の全rank走査とレイヤー別quotaへのfillに変更する候補である。globalのprefix順序は維持し、重複profileは明示的に拒否する。\n\n[索引](report-registry-v121.json)には共有した過去reportと今回のsource identity、実際のagent状態を残した。実装agentは完了し、Lunaはusage limitによりactive0。報告の末尾欠落によるEOSの説明は仮説として扱う。Mainの独立レビュー・統合・build/lint/ASan・実occupancy・モデル品質・性能は未検証。\n')
    attrs=W/'.gitattributes'
    attrs.write_text(attrs.read_text()+'bench/results/2026-10-11-parallel-round98/*.patch -text -whitespace\n')
    copy(Path(__file__), d/'archive-boundary.py')
    for root in (d,q):
        for p in root.rglob('*.json'): json.loads(p.read_text())
        for p in root.rglob('*.py'): ast.parse(p.read_text())
    validation=dict(created_utc=m.utc(), active=True, passed=False, complete=False, commands=[], copied_files=copies, json_and_python_ast_checked=True, status_preserved=True)
    def save(): (d/'archive-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    owner=m.Owner(B/'xe-model-task-tail-boundary-commands-v1',save)
    validation['commands']=owner.commands
    save()
    for i,root in enumerate((d,q)):
        e,_,_=owner.run('gitleaks-'+str(i),[str(X/'.lint/bin/gitleaks'),'dir',str(root),'--no-banner','--redact','--exit-code','1'],dict(PATH='/usr/bin:/bin',HOME='/home/yayoi',LANG='C.UTF-8'),W,wall=30)
        m.require(m.closed(e) and e['exit_code']==0,'archive gitleaks')
    validation.update(active=False,passed=True,complete=True,finished_utc=m.utc())
    save()
    print(json.dumps(dict(archived=True,files=len(copies),model_v4_executed=False,profile_fill_applied=False)))
