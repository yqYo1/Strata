# Wave集計のCPU検証手順

2026-10-15。前の継続作業は比較入力のprovenance手順、wave census提案のレビュー、
研究報告を準備したprogress。直前の進捗回答では現物の測定ログとGit状態を読んだ。
今回も同じexec session16956をpollし、runningであることを確認した。別の測定は起動していない。
full OFFの約32Kは200.5/199.8/199.5 tok/s、最初のcode長文は3346 tokens/9.8 tok/s。
varied-proseのthinkingは1257から2333へ進んだ。これは途中経過で、正式完了や採用ではない。

Mainは次の三ファイルを追加した。すべてsource-only、未build・未test・未実行である。

- `qualify_wave_census_host_v2.py`: 同じmeasurement lockをnonblocking取得し、現在のfull OFFの
  終端receipt、子processのreap、session空、PID/start identity消失を確認する。提案の全pin、
  Mainのfixture/validator、Owner/helper、compilerのhashを保存する。既存の独立CPU fixtureと
  新しいserializer fixtureをnormalとASan/UBSanでそれぞれbuild/runし、Pythonの実出力照合と
  提案validatorの3 testsも実行する。`-UNDEBUG`を必須にし、assertionを無効にしない。
- `wave_census_serialization_fixture_v1.cpp`: 提案headerそのものをincludeし、5つのsynthetic
  geometryを48 layerずつ処理する。独立のGU/Down発行順を使い、実sizeofとserializerの出力を
  stdoutへ返す。n/pos/chunkは(1,0,1)、(13,17,3)、(32769,0,8192)、(262144,0,8192)、
  (262127,17,8192)。最大絶対位置とnonzero position/tail、q2/q4 full/partialを含む。
- `check_wave_census_serialization_v1.py`: 実際のC++ JSONにgeneric wave validatorを適用した上で、
  Pythonのset/Counter参照から旧exact-ne histogram、新wave summary全欄を独立に再計算して
  全48 layerを照合する。normal/sanitizer双方の結果が同じであることもMain runnerが確認する。

一commandあたりwall/CPU180秒（Python単体testsは60秒）、RSS2 GiB、text4 MiB、
job256 MiB、file32 MiBを上限にする。Ownerのnormal exit/reap/session終了を必要とし、
失敗receiptはそのまま保存する。出力lineは旧1 MiB cap、ownerは旧4 MiB capを維持する。

閉じた境界での実行予定:

```sh
python3 -B bench/results/2026-10-15-pr6-recovery-audit/qualify_wave_census_host_v2.py
```

実行時だけ`wave-census-host-v2.json`と外部stateのfresh rootを作る。現在はPASS receiptを
置いていない。元proposal、実行中client/native/header、modelに変更はない。実行中のfull OFF
とinput provenance回復・matched ONの優先順は前のMain reviewどおり。CPU検証が通れば、
別のdiagnostic binaryを作って32Kのwave unionを観測する。比較用b1cc9cb4…は上書きしない。

このfixtureはsynthetic route集計の256K geometryであり、物理256K入力、実tensor領域の最大量、
GPUの安全性、品質、RESTORE、速度の証拠ではない。wave機会数は省けるexpert展開回数の候補で、
節約時間や実転送帯域ではない。PR#6の645.85 tok/s再現、1000/70、物理256K/RESTOREの未完は続く。

別スコープのLuna調査R639（top-k順を維持したwave Dm削減）とR640（適応MTPの実装可能性）を
再開した。共有先はこのauditディレクトリの過去全report。途中介入せずFINALを待ち、Mainは
この検証手順を並行で準備した。実装・テストの採用判断はMainが行う。

後続source reviewで、旧full OFF producerはreceipt直下に`finished_utc`を書かず、
各commandにだけ終了時刻を書くと確認した。未実行のMain qualifierを修正し、各commandの
終了時刻/reap/session空とPID/start消失を確認するようにした。旧receiptは改変しない。
この訂正も未実行であり、測定終了やCPU test PASSを意味しない。
