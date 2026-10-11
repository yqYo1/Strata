import ast
import fcntl
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path

B = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007')
W = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/docs-sycl-storage-retention-20261009')
X = Path('/home/yayoi/ghq/github.com/MistVVK/XeStrata/.worktree/fix-xestrata-e32-host-handoff-20261011')
D = W / 'bench/results/2026-10-11-xe-native-dequant-group/optimized-ir'
A = B / 'xe-native-dequant-group-optimized-ir-analysis-v1'
R = B / 'xe-native-dequant-group-optimized-ir-root-v2'
O = B / 'xe-llvm-v711-source-v5/direct_owner.py'
sp = importlib.util.spec_from_file_location('owner', O)
m = importlib.util.module_from_spec(sp)
sp.loader.exec_module(m)
m.require(m.sha(O) == 'a29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f', 'qualified owner')
ENV = dict(PATH='/usr/bin:/bin', HOME='/home/yayoi', LANG='C.UTF-8', LC_ALL='C.UTF-8')
if 'SSH_AUTH_SOCK' in os.environ:
    ENV['SSH_AUTH_SOCK'] = os.environ['SSH_AUTH_SOCK']

with (B / 'owned-v0141-measurement.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = B / 'xe-native-dequant-group-optimized-ir-boundary-root-v1'
    m.require(not out.exists(), 'new immutable boundary')
    out.mkdir()
    r = dict(active=True, complete=False, passed=False, commands=[], started_utc=m.utc(),
             controller_sha256=m.sha(__file__), gpu_executed=False, model_executed=False, deleted=[])

    def save():
        (out / 'record.json').write_text(json.dumps(r, indent=2) + '\n')

    owner = m.Owner(out / 'commands', save)
    r['commands'] = owner.commands
    save()

    def run(label, args):
        e, so, se = owner.run(label, args, ENV, W, wall=60, cpu=120,
                              text_cap=4 << 20, total_cap=16 << 20, rss_cap=2 << 30)
        m.require(m.closed(e) and e['exit_code'] == 0, 'normal closed ' + label)
        return so.read_text().strip()

    try:
        m.require(not run('report-status', ['/usr/bin/git', 'status', '--porcelain']), 'clean report branch')
        m.require(not D.exists(), 'new IR archive')
        D.mkdir()
        copies, plans, receipts = {}, [], {}

        def copy(src, name):
            dst = D / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            m.require(m.sha(dst) == m.sha(src), 'exact IR archive')
            copies[str(dst.relative_to(W))] = dict(original_path=str(src), bytes=src.stat().st_size,
                                                   sha256=m.sha(src))
            return dst

        for name, root in [('original-missing-quote', 'xe-native-dequant-group-optimized-ir-root-v1'),
                           ('both-mode-optimized-ir', 'xe-native-dequant-group-optimized-ir-root-v2')]:
            src = B / root / 'record.json'
            v = json.loads(src.read_text())
            m.require(not v['active'] and all(m.closed(c) for c in v['commands']), 'closed IR processes')
            receipts[name] = dict(root=root, sha256=m.sha(src))
            copy(src, name + '/record.json')
            if name == 'original-missing-quote':
                m.require(not v['passed'] and not v['complete'] and v['commands'][-1]['exit_code'] == 1,
                          'original FAIL stays FAIL')
                copy(B / root / 'commands/free-v1-optimized-device-ir.stderr', name + '/representative-error.stderr')
            else:
                m.require(v['passed'] and v['complete'], 'new IR PASS')
                m.require(all(m.sha(x['path']) == x['sha256'] and m.sha(x['source']['path']) == x['source']['sha256']
                              for x in v['optimized_ir']), 'exact complete IR')
                for version in ['v1', 'v2']:
                    filename = 'free-' + version + '-driver-recipe.stderr'
                    copy(B / root / 'commands' / filename, name + '/' + filename)
            for c in v['commands']:
                for i in c['owners']:
                    live = m.identity(i['pid'])
                    m.require(not live or live['start_ticks'] != i['start_ticks'], 'no live IR owner')
            for p in (B / root / 'commands').glob('*'):
                if p.suffix in ['.stdout', '.stderr']:
                    st = p.stat()
                    plans.append(dict(path=str(p), bytes=st.st_size, allocated_bytes=st.st_blocks * 512,
                                      sha256=m.sha(p), retained_receipt=str((D / name / 'record.json').relative_to(W)),
                                      reason='Closed compiler/help/driver traces. Exact commands/env/source/header/IR identities, original failure cause/status, representative compiler error and one actual driver recipe per candidate retained. No unresolved full log consumer.'))

        texts = {k: (R / ('free-' + k + '.ll')).read_text() for k in ['v1', 'v2']}
        defs = {k: {re.search(r'@([^ (]+)\(', f.group(0).splitlines()[0]).group(1): f.group(0)
                    for f in re.finditer(r'^define[^\n]*\{\n.*?^\}', text, re.M | re.S)}
                for k, text in texts.items()}
        norm = lambda t: re.sub(r'!\d+', '!NODE_ID', t)
        sha_text = lambda t: hashlib.sha256(t.encode()).hexdigest()
        m.require(defs['v1'].keys() == defs['v2'].keys() and len(defs['v1']) == 200, 'same200 complete definitions')
        identities = json.loads((A / 'defined-function-identities.json').read_text())
        for item in identities:
            key = item['symbol']
            m.require(all(item['raw_sha256'][k] == sha_text(defs[k][key]) and
                          item['metadata_id_normalized_sha256'][k] == sha_text(norm(defs[k][key]))
                          for k in defs), 'complete function identities')
            m.require(item['metadata_id_normalized_equal'] == (norm(defs['v1'][key]) == norm(defs['v2'][key])),
                      'exact unchanged/changed classification')
        m.require(len(identities) == 200 and sum(not x['metadata_id_normalized_equal'] for x in identities) == 2,
                  'only2 group functions differ')
        analysis = json.loads((A / 'analysis.json').read_text())
        m.require(analysis['compile_receipt']['sha256'] == m.sha(R / 'record.json') and len(analysis['selections']) == 10,
                  'Main analysis/selection pins')
        for item in analysis['selections']:
            body = defs[item['version']][item['mangled_symbol']]
            p = Path(item['snippet'])
            m.require(p.read_text() == body + '\n' and m.sha(p) == item['snippet_sha256'], 'full function snippets')
        for k in texts:
            duplicate = R / ('contrib-llvm-' + k + '.ll')
            m.require(duplicate.read_bytes() == (R / ('free-' + k + '.ll')).read_bytes(), 'actual mode IR byte equality')
            st = duplicate.stat()
            plans.append(dict(path=str(duplicate), bytes=st.st_size, allocated_bytes=st.st_blocks * 512,
                              sha256=m.sha(duplicate), retained_receipt=str((D / 'both-mode-optimized-ir/record.json').relative_to(W)),
                              reason='Actual independently compiled contrib-llvm IR is byte-identical to corresponding current free IR. Per-mode command/source/receipt and exact equality retained; one full copy per source version suffices for named final-lowering consumer.'))
        consumer = json.loads((A / 'current-large-consumers.json').read_text())
        m.require(sum(x['bytes'] for x in consumer['files']) <= consumer['byte_budget'], 'named full IR consumer5MiB')
        for x in consumer['files']:
            m.require(m.sha(x['path']) == x['sha256'], 'current IR fixture')
        for p in sorted(A.iterdir()):
            if p.is_file():
                copy(p, 'analysis/' + p.name)
        for p in [B / 'xe-native-dequant-group-optimized-ir-v1.py',
                  B / 'xe-native-dequant-group-optimized-ir-v2.py', Path(__file__)]:
            ast.parse(p.read_text())
            copy(p, 'recipes/' + p.name)
        registry = W / 'bench/results/2026-10-11-xe-native-dequant-group/report-registry-v128.json'
        m.require(json.loads(registry.read_text())['research_completed'] == 506, 'current research history')
        (D / 'previous-registry.json').write_text(json.dumps(dict(path=str(registry.relative_to(W)),
                                                               sha256=m.sha(registry), reports=506,
                                                               new_agent_report=False), indent=2) + '\n')
        (D / 'RESULTS_JA.md').write_text('''# Group pointer選択の最適化済みdevice IR確認

Mainがproduction compile DBのO3・数値設定・Werrorを使い、v1と定数添字APIv2の完全translation unitをdevice-only IRとして両モードで実際にcompileした。候補を別directoryから読むため、元sourceのローカルquoted header検索を同じ-iquoteで復元した。最初の試行はcuda_intrinsics.hppを探せずFAIL。原statusとerrorを保持し、修正controllerの4compileはPASS。kernel/math flagの緩和は行っていない。

|確認箇所|v1|APIv2|
|---|---|---|
|GU kernelのprivate alloca / memcpy|各1、776 byte copy|ともに0|
|Down kernelのprivate alloca / memcpy|各1、776 byte copy|ともに0|
|pointer選択|private captureへの動的GEP/load|引数の定数offset loadとswitch/phi|
|kernel引数のbyval構造体|776 byte payload|残る|

各候補のIRは、free/contrib-llvmの独立compileでSHA256まで一致。200の完全define bodyを比較し、変更は2つのgroup kernelだけだった。その他198はmetadata nodeの参照番号だけを正規化するとinstruction textが同一。metadata内容全体の意味的同一性を証明するものではない。decoderと旧個別関数のsource byte不変は別のMain source reviewで確認済み。

生成IRからprivate capture copyを取り除けたことは確認できたが、最終SPIR-V/IGC命令のcopy・scratch・spillやGPU時間への寄与は未測定。kernel ABIのbyval payloadも残る。APIv2のGPU改善、モデル品質や速度はまだ主張しない。v1のcomponent回帰と不採用判断は変更しない。production callerは未適用。

次は適用したv2で両モードのproduction release/host sanitizer/lintを確認し、新しいsource/binaryで18形式の全byte・guard・refusal・healthと同一componentを測る。有益なcomponent結果が出た場合にcaller lifetime・32K厳密出力・速度を扱う。full262144/RESTOREと本家単体の最適設定探索は未完了。

完全IRは最終lowering比較を行うMainの現在のconsumer用としてfreeの2つだけを外部stateに保持する。合計5MiB上限で、v2 component採用・拒否かlowering比較の終了時に再検討する。今回の実行ではGPU/サービス操作やsource編集は行っていない。新規agent報告はなく、索引v128の506件を参照する。
''')
        (D / 'archive-manifest.json').write_text(json.dumps(dict(byte_identical_copies=copies,
                                                               original_statuses_preserved=True,
                                                               source_unchanged=True, gpu_executed=False), indent=2) + '\n')
        m.require(sum(p.stat().st_size for p in D.rglob('*') if p.is_file()) < 4 << 20, 'compact IR evidence4MiB')
        run('archive-secret-check', [str(X / '.lint/bin/gitleaks'), 'detect', '--no-git', '--redact', '--no-banner',
                                    '--exit-code', '1', '--source', str(D)])
        run('diff-check', ['/usr/bin/git', 'diff', '--check'])
        run('stage-evidence', ['/usr/bin/git', 'add', '--', str(D.relative_to(W))])
        run('commit-evidence', ['/usr/bin/git', 'commit', '-m', 'docs(bench): compare grouped dequant capture lowering'])
        r['evidence_revision'] = run('evidence-head', ['/usr/bin/git', 'rev-parse', 'HEAD'])
        retention = D / 'retention-v1.json'
        retention.write_text(json.dumps(dict(retained_commit=r['evidence_revision'], deletions=plans,
                                             logical_bytes=sum(x['bytes'] for x in plans),
                                             allocated_file_bytes=sum(x['allocated_bytes'] for x in plans),
                                             available_space_not_measured=True, current_large_consumers=consumer,
                                             original_statuses_unchanged=True), indent=2) + '\n')
        run('retention-secret-check', [str(X / '.lint/bin/gitleaks'), 'detect', '--no-git', '--redact', '--no-banner',
                                      '--exit-code', '1', '--source', str(retention)])
        for item in plans:
            p = Path(item['path'])
            m.require(m.sha(p) == item['sha256'] and p.stat().st_size == item['bytes'], 'exact closed retired IR path')
            p.unlink()
            r['deleted'].append(item['path'])
        m.require(all(m.sha(B / item['root'] / 'record.json') == item['sha256'] for item in receipts.values()),
                  'original compiler receipts unchanged')
        run('stage-retention', ['/usr/bin/git', 'add', '--', str(retention.relative_to(W))])
        run('commit-retention', ['/usr/bin/git', 'commit', '-m', 'docs(bench): retire duplicate IR and closed compiler traces'])
        r['committed_revision'] = run('final-head', ['/usr/bin/git', 'rev-parse', 'HEAD'])
        run('publish-own-fork', ['/usr/bin/git', 'push', 'origin', 'HEAD'])
        m.require(not run('final-status', ['/usr/bin/git', 'status', '--porcelain']), 'clean report tree')
        r['complete'] = True
    except BaseException as exc:
        r['error'] = repr(exc)
    finally:
        r['active'] = owner.active is not None
        r['passed'] = bool(r['complete'] and not r['active'] and not r.get('error'))
        r['finished_utc'] = m.utc()
        save()
    print(json.dumps({k: r.get(k) for k in ['passed', 'error', 'evidence_revision', 'committed_revision']}))
    raise SystemExit(0 if r['passed'] else 1)
