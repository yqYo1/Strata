"""Root archive at the CPU-build/first-GPU phase boundary; no cleanup."""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, shutil, subprocess

B = Path(__file__).parent
R = B / 'research-20261009'
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
A = W / 'bench/results/2026-10-10-xestrata-pristine-build'
P = W / 'bench/results/2026-10-10-parallel-round83'

def ident(p):
    with p.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=digest)

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert subprocess.check_output(['git', 'status', '--porcelain'], cwd=W, text=True) == ''
    assert not A.exists() and not P.exists()
    build = B / 'xestrata-pristine-icpx-build-root-v1'
    contract = B / 'xestrata-diagnostic-contract-root-v3'
    receipt = json.loads((build / 'record.json').read_text())
    assert receipt['passed'] and not receipt['active']
    assert json.loads((contract / 'record.json').read_text())['passed']
    A.mkdir(); P.mkdir()
    copied = []
    def copy(src, dst):
        shutil.copyfile(src, dst)
        before, after = ident(src), ident(dst)
        assert before['bytes'] == after['bytes'] and before['sha256'] == after['sha256']
        copied.append(dict(source=before, archive=after, byte_identical=True))
    for name in ['record.json', 'configure.stdout', 'configure.stderr', 'build.stdout', 'build.stderr', 'build_owner.py']:
        copy(build / name, A / ('build-' + name))
    for name in ['compile_commands', 'cache_identity', 'build_json']:
        copy(Path(receipt[name]['path']), A / Path(receipt[name]['path']).name)
    for name in ['build_xestrata_pristine_icpx_v1.py', 'prepare_xestrata_diagnostic_contract_v2.py', 'prepare_xestrata_diagnostic_contract_v3.py']:
        copy(B / name, A / name)
    for name in ['record.json', 'build-receipt-adapter.json', 'run_first_diagnostic_v3.py']:
        copy(contract / name, A / ('contract-' + name))
    failed = B / 'xestrata-diagnostic-contract-root-v2'
    for name in ['failed-preparation.json', 'run_first_diagnostic_v2.py', 'build-receipt-adapter.json']:
        copy(failed / name, A / ('failed-v2-' + name))
    for name in ['run_first_diagnostic.py', 'root_recipe.txt']:
        copy(B / 'xestrata-owned-comparison-v1' / name, A / ('source-only-v1-' + name))
    copy(R / 'implementation-xestrata-owned-comparison-v1.txt', A / 'implementation-xestrata-owned-comparison-v1.txt')
    previous = W / 'bench/results/2026-10-10-parallel-round82/report-registry-v105.json'
    registry = json.loads(previous.read_text())
    reports = [
        ('round286-nativecpu-capacity-source-metric-production-call-role.txt', '/root/research_bottleneck_evidence_audit_v203'),
        ('round287-moe-next-shared-scratch-ceiling.txt', '/root/research_native_iq4nl_esimd_k640_v202'),
        ('round288-xestrata-comparative-source-audit.txt', '/root/research_bottleneck_evidence_audit_v203'),
        ('round289-intermediate-storage-precision-quality-evidence.txt', '/root/research_native_iq4nl_esimd_k640_v202'),
        ('round290-xestrata-arc-b570-xmx-grouped-gemm-eligibility.txt', '/root/research_bottleneck_evidence_audit_v203'),
        ('round288-correction-after-r290.txt', '/root/research_bottleneck_evidence_audit_v203'),
        ('round291-residual-storage-local-quality-experiment.txt', '/root/research_native_iq4nl_esimd_k640_v202'),
        ('round292-pristine-xestrata-64k-memory-formats-eligibility.txt', '/root/research_bottleneck_evidence_audit_v203'),
    ]
    # Earlier task scopes choose their original report filename. Resolve by round
    # only when exactly one immutable report matches that round.
    new_reports = []
    for name, agent in reports:
        source = R / name
        if not source.exists():
            round_id = name.split('-', 1)[0]
            matches = [x for x in R.glob(round_id + '-*.txt') if 'correction' not in x.name]
            assert len(matches) == 1, (name, matches)
            source = matches[0]
        copy(source, P / source.name)
        new_reports.append(dict(**ident(source), archive_path=str(P / source.name), model='gpt-6-luna',
                                agent=agent, status='completed-read-only', root_full_report_reviewed=True,
                                scope=source.stem))
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    running = [dict(agent='/root/research_bottleneck_evidence_audit_v203', model='gpt-6-luna', status='running-read-only-R293'),
               dict(agent='/root/research_native_iq4nl_esimd_k640_v202', model='gpt-6-luna', status='running-read-only-R294'),
               dict(agent='/root/implement_prefill_phase_timer_validity', model='gpt-6.1-sol', status='running-source-only-grouped-GEMM-qualifier')]
    registry.update(registry_version=106, created_utc=now, reports=registry['reports'] + new_reports,
                    research_completed=len(registry['reports']) + len(new_reports), new_reports=new_reports,
                    previous_committed_registry=dict(**ident(previous), storage_commit='71b9ef02'),
                    scope='Pristine Xe build and first diagnostic CPU contract;64K accepted goal condition',
                    live_agent_snapshot=running,
                    current_root_decisions=dict(adopted=False, model_executed=False, gpu_executed=False,
                                               closed_build_passed=True, contract_cases=20, diagnostics_not_timing=True,
                                               actual_prefill_goal_input=65536, fresh_no_reuse=True,
                                               minimum_comparison_input=32768, full_physical_context=262144,
                                               next='Small owned first GPU check and narrow grouped kernel qualifier; pristine fork32K admission then same-length64K comparisons'),
                    root_review_corrections_round83=[
                        'Discard R288 NVIDIA/RTX B570/TensorCore eligibility conclusions and full-K GU/H sizing for compact D; preserve original plus separate R290 erratum.',
                        'R292 default-ring description is not actual selected ring: pinned fork clamps STRATA_PREFILL_RING=8 to16 for large chunks. Record actual effective ring difference from current baseline8.',
                        'R291 LOGPOS quality coverage requires source-path confirmation; R294 assigned before adopting it for batched candidate quality.',
                        'Source-only first controller had wrong PP total32767 and syntax error; preserved v1/v2, rootv3 fixed total32768 and missing parenthesis and passed20CPU cases; compiler full2026.1.1 recorded.',
                        'FP16/BF16/FP8 intended storage changes require ability metrics; existing strict numerical failures remain unchanged and cannot alone establish ability degradation.',
                    ])
    (P / 'report-registry-v106.json').write_text(json.dumps(registry, ensure_ascii=False, indent=2) + '\n')
    (R / 'report-registry-v106.json').write_text((P / 'report-registry-v106.json').read_text())
    (A / 'RESULTS_JA.md').write_text('''# XeStrataの無改変ビルドと測定条件

ghqで独立して取得したXeStrataの39bdadcc9e2b89b1e3c8be7bb2a603b04fa0e197
（xe0.1.40.2.1、fork元0.1.40.2/e8ca9af）をworktreeで無改変ビルドした。
IntelLLVM2026.1.1、contrib-icpx、Release、GGML3cf03257を使用した。
configure/buildは正常終了し、所有した全プロセスを回収した。モデルとGPUの実行はこの時点では未実施。
これはビルド成功であり、推論・品質・速度の検証結果ではない。

## 目標と比較条件

ユーザーの2026-10-10の指定により、prefill1000 tok/sの達成判定は約64Kの実入力
（予定65536 tokens）でもよい。32Kを固定目標長にしない。長い入力で初期読み込み等の
固定費を償却する条件を採用できる。双方で同じtoken列・入力長を使い、比較は最低32768 tokens。
入力fixtureのハッシュ、実際に読み込んだ全token数、再利用数0を確認する。
計時範囲を示し、起動・ロードを含む時間とrequestのprefill時間を分けて記録する。
1000 tok/sで65536 tokensを処理するrequest時間の予算は65.536秒。
prefix cacheで読み込まなかったtokenをfresh-prefillの分子に含めない。

decode70 tok/sは別に評価し、複数回の個別サンプルと分布を保存する。
prefillの利益だけでdecodeの変更を採用しない。採用前にはphysical262144位置を
実際に消費するcontext境界・状態復元・容量の検証を別途行う。

初回は32768入力の診断で安全性とprotocolを確認し、別のcleanプロセスで64K比較へ進む。
同じ設定値でもforkのring override8はsourceで16にclampされる。現在実装のeffective8との
差を明記し、設定文字列だけを見て同一のメモリ配置とは扱わない。

## CPU側の確認

source-onlyの初版controllerにはPP総数を32767と判定する誤りと、fault-filterの
閉じ括弧欠落があった。実装sourceでは総数32768、batch最終位置32767である。
失敗したCPU準備と原文を保存し、rootのv3で修正した。現行のcanonical32768入力に
対する受理、欠けたDONE/PP/LP、再利用、非有限値、範囲外token等、20ケースを確認した。
ビルドreceipt、binary、source、compiler、compile inputs、GGMLのidentityも照合した。
初版・失敗v2を推論に使っていない。

forkにはこちらの独立した全head/live-state dumpがない。初回のID/LP比較だけを
数学的同値や品質同等と呼ばない。grouped GEMMの小さなqualifierを別途準備している。

並行調査は[registry106](../2026-10-10-parallel-round83/report-registry-v106.json)に保存した。
R288のGPU種別の誤りは原文と別の訂正を保存した。Qwen開発元の残差FP8保存の記述は
低精度保存の候補を検討する根拠だが、この実装での劣化率はまだ測定していない。
''')
    guide = W / 'bench/results/2026-10-10-hardware-architecture-decisions/CAPACITY_AND_ARCHITECTURE_JA.md'
    text = guide.read_text()
    anchor = '## 順番で改善できる範囲と、転送量を変える必要がある範囲\n\n'
    assert text.count(anchor) == 1
    note = ('prefill1000 tok/sの目標は、ユーザーの指定により約64Kの実入力でも評価する。'
            '65536 tokensなら時間予算は65.536秒。以下の32Kの算術は、測定済みの仕事量を'
            'その入力長で照合したものであり、目標を32Kに限定するものではない。'
            '64Kの仕事量や固定費の償却を実測し、双方で同じ入力長・計時範囲を使う。'
            '[比較条件](../2026-10-10-xestrata-pristine-build/RESULTS_JA.md)。\n\n')
    guide.write_text(text.replace(anchor, anchor + note))
    copy(Path(__file__), A / Path(__file__).name)
    (A / 'archive-identity.json').write_text(json.dumps(dict(created_utc=now, copies=copied, cleanup_performed=False), indent=2) + '\n')
    secrets = [v.encode() for k, v in os.environ.items() if any(t in k.upper() for t in ['TOKEN', 'PASSWORD', 'SECRET', 'CREDENTIAL', 'API_KEY']) and len(v) >= 12]
    for d in [A, P]:
        for p in d.rglob('*'):
            if p.is_file(): assert not any(v in p.read_bytes() for v in secrets), 'credential content in new archive'
    print(json.dumps(dict(archived=True, reports_added=len(new_reports), registry_reports=registry['research_completed'], build_passed=True, contract_passed=True, gpu_executed=False)))
