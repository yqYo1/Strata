# QSA fallback layout候補の主エージェントレビュー

2026-10-15 JST。Solの実装を受領し、production diffとfixture全体を読んだ。
現在のmultiply/FMA/reduction順、KV format dispatch、empty/masked partial、merge、
scratch span/geometry検査を保持している。自然順APIはlayout0へ固定し、
SYCL prefill fallbackのexact flag1だけがlayout1を使う。matrix admissionは変更していない。
local storeの`(dim%8)*32+dim/8`とloadの`lane+j*32`の対応を旧ソースと照合した。
scopeは旧query layoutのみで、旧engine/merge/演算順の移植ではない。

主エージェントによる追加変更:

- `--host-only`はruntime生成前にgeometry/layout/overflow検査を終了する。
- clean microbenchは全active cellがresidentとなる小さな物理poolを使う。
  元案の大部分nonresidentな入力では通常のscore負荷をほとんど測れないため変更した。
  非identity mapとlogical cell262143は残した。40 samplesの順序を0/1/1/0とする。
- 明示layout flagのある実モデルrunは、成功時にmatrix/fallbackのhost call/query数を出す。
  device時間、GPU成功launch数、因果的critical pathの測定ではない。追加device操作はない。
  これにより候補が実際に選ばれたかを確認してからprefill速度を解釈できる。

fixtureの全format0..3、自然順APIとlayout0/1のoutput/partial bitwise比較、
独立double reference、guard、metadata再利用、mask、logical full-boundaryを読んだ。
これらを追加しても、実モデル256K満杯やRESTOREの証明にはならない。
旧nativeとのnatural-layoutモデル比較用に18.2MBの実binaryを一時保持する。
consumer、128MiB budget、次reviewをbuild receiptに明示し、単独の再開backupにしない。

この時点ではbuild、CPU実行、GPU検査、microbench、実モデル速度とも未実行・未合格。
agent原文のsource hashとbenchmark形状は受領時のものとして保持する。
主エージェントの最終source pinsと結果は以後のbuild/diagnostic receiptを参照する。
