# XeStrataの時間計測の修正候補

[報告](implementation-xe-profiling-interval-guard-v1.txt)と[patch](profiling-interval-guard.patch)を全文確認し、[Mainレビュー](main-review.json)を経てe188の作業ブランチに適用した。失敗した区間をprefillの集計から黙って省かず、タイムスタンプ逆転は生の端点を残して拒否する。replay診断の既存bool APIにも失敗を伝える。演算・キャッシュ・context・MTPの方針は変更していない。

元の実装基点514とMain基点e188は同一とは扱わない。対象3ファイルの旧bytesが一致することを確認し、e188のキャッシュ修正を保った。[元のmanifest](source-manifest.json)は変更していない。両source-LLVM modeのrelease/CPU確認は通過。host-ASan、必須lint、実GPUの時間計測確認は未完了で、性能の採用判断はない。

[索引](report-registry-v122.json)に実際のagent状態と、返却後に割り当てた別の実装調査を残した。実装agentは小さな実GPU計測用sourceだけを作成し、Mainがすべての実行を直列に管理する。Luna調査はサービスのusage limitでactive0、他モデルへの代替はしていない。旧stage/hardware timingの失敗省略と時計domain/coverageの未検証は残り、時間の根拠としては使わない。

Main追記: 両modeのhost-ASan/UBSanも12 PASS・1 native ISA skipで完了した。34件のsource pinがreleaseと一致する。旧GPU確認のうち4実行ファイルはbyte-identicalで再利用可能だが、denseの2実行ファイルは変化したため新規GPU確認が必要。lintと実timestamp確認は未完了。
