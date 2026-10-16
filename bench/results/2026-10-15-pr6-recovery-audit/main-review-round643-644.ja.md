# PCIe修正の検証準備と集計validatorの修正

2026-10-15。前の継続turnは、コピーAPIの戻り値無視を確認し、隔離修正proposalと
エラー注入fixtureを保存、calibration用途の差を特定したprogress。今回も同じexec16956を
直接pollしてrunningを確認した。run2 varied-proseのthinkingは2398から4772まで進んだ。
初回code3346/9.8 tok/s、prose8631/6.2 tok/s以降の新しい完了標本はまだない。
新CPU tests/build/GPU/cleanupは実行せず、root product/client/b1 binaryは固定している。

## Mainの次の実行手順

`qualify_checked_pcie_fetch_host_v1.py`を追加した。source-onlyで未実行。測定lockを取得し、
現full OFFの終端receipt、各commandの終了時刻/reap/session空、PID/start消失を確認する。
閉じた旧runが失敗していても、それを書き換えずCPU helperだけを検証できる。
base/artifact/fixture/helper/compiler/Ownerをhashで固定し、実production helperを使うfixtureを
normalとASan/UBSanでcompile/runする。`-UNDEBUG`を必須とし、正常exitと静かなstdout/stderr、
全ownerの終了を確認する。wall/CPU120秒、RSS2 GiB、text1 MiB、job128 MiB、file32 MiB。
これはfake enqueue/drain APIを使うhelper検証であり、Verifier/実queue/driver/モデルではない。

`build_checked_pcie_fetch_v1.py`もsource-onlyで追加した。実host PASSと凍結native buildからだけ
進み、proposal includeを先頭にしてverify.cppとgenerate.cppを同じclass layoutで再compileする。
現在のcompile_commandsでこの2TUを確認した。-MMD/-MFはprivate rootへ出し、両depfileが候補
verify.hppとhelperを含み、旧verify.hppを含まないことを確認する。旧生成objectをlink command
から差し替え、link mapで旧verify/generate objectの混入を拒否する。元archiveとb1 binaryは
上書きしない。全compile/linkのOwnerとhash、source/linked inputsを別receiptへ記録する。
1 commandあたりwall900秒、RSS32 GiB、text8 MiB、job2 GiB、file512 MiB。GPUを起動しない。

compile_commandsは歴史的STRATA_VERSION=0.1.38を保持する。新buildでも同じ文字列macroを
正しくargvへ渡すが、このversion表示を取り込み済み上流機能の証拠にはしない。実sourceと
resolver/機能auditが基準である。setupやserver等の未完了port義務はそのまま残る。

閉じた境界での予定command（すべて未実行）:

```sh
python3 -B bench/results/2026-10-15-pr6-recovery-audit/qualify_checked_pcie_fetch_host_v1.py
python3 -B bench/results/2026-10-15-pr6-recovery-audit/build_checked_pcie_fetch_v1.py
```

現OFF終了後のnonce回復をclient変更より先に行う順序、matched IQ ONのgateは変えない。
新candidateを性能pairのbinaryへ混ぜない。実GPU PCIe診断はこのbuildだけで許可済みにせず、
normal/fault/actual positive PCIe routes/numerical/qualityをMainが別に確認する。

## R643の独立レビューを反映

FINAL後にMainがreportと現proposalを照合した。copy APIエラーの直後に返るbranchにも、
既知の非success work_status_を別codeとして併記するよう修正した。このbranchより後の
event queryの結果まで取得したという意味ではなく、publish/launch等で既に観測したcodeを保つ。
R643のevent-query位置の説明はこの区別が必要である。

helperのoperation名もcopy_enqueue/callback_enqueueからcopy_api/callback_apiへ変更した。
compatのAPI入口は既存sticky async errorでも失敗するので、戻り値だけからenqueue固有の
原因とは断定しない。実fixtureにこのoperation名のassertを加え、manifestとdiffを更新した。
元proposalはcommit87a5f1d5から復元できる。first error/partial drain/consumer拒否の意味は
同じで、変更後も未build・未testである。copy completion未知時のhelper reset拒否は、実GPU
復旧や失敗したVerifierの再利用を許可する証拠ではない。既存teardownのunchecked APIも
正常なdriver/queue回復の証明にはしない。

## R644の独立レビューを反映

q2/q4の算術とissuer hookには新しいblockerを見つけなかった。generic Python validatorでは
bool/floatとintの等値比較が通り得たため、versionとwave_schema_version、layer/chunks/tokens/
GU/Down callsのexact int型と範囲を必須にした。独立hand fixtureの値をbool/floatへ置換する
第4 unittestを追加し、Main runnerも期待4 testsへ更新した。元model用v1 validatorを先に
呼ぶ経路は維持した。製品census headerは変更していない。

buffered flush問題はv1から継承した性質で、wave unionの算術バグとは分ける。
Main serializer fixtureに、十分大きいstdio bufferを付けた/dev/fullへ実Writerを使って
書き、fwrite成功後にfflushで初めて失敗するcaseと、complete()がoutput_io_incompleteを
内部に残すcaseを追加した。このsourceも未実行。外部へ既に渡ったcompletion文字列の
取り消しやdurable acknowledgementを修正したとは言わない。
推論return成功だけで診断出力を合格にせず、collectorの全record/geometry/conservationと
完全なpayload/owner/fault gateを必要とする。1 MiB直前の成功出力や実モデルの最大密度は
まだ未検証で、今後のmodel censusで実byte数とcap適合を確認する。

## 継続範囲

今ある候補の検証を優先し、今回の反復調査は新しい最適化案を増やさず独立source auditに
分けた。Luna2名のFINALを待つ間、Mainは上のhost/build手順を準備した。
1000 PP/70 TG、PR#6速度回復、物理262144、RESTOREとqualityはいずれも未達のまま。
