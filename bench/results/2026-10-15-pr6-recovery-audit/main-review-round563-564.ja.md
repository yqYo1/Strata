# GEMM差分とmatrix attention計算量のレビュー

2026-10-15 JST。R563/R564を受領して読み、現在のGEMM wrapperとmatrix kernelの
admission・local accessor・dispatchを主エージェントで確認した。

R563の主要結論は限定して採用する。現在のcompact FP16のexpertごとのGU/Down、
DQ=2と出力offsetは旧実装と対応する。旧column-majorと現在row-majorは同じ出力bytesの
`X W^T`を表現するが、oneMKL内で実際に違うカーネルになるかは未確認。
61.5秒のDown phase bucketはexclusive GEMM時間ではない。
次は同じbinary/library/queue/入力で両記述を実行する診断toolを作る。
最初のscreenはsynthetic hot FP16 operandsと明記し、実7format rotationや全model速度の
測定へ見せかけない。sourceやproductionのorientationを先に変えない。

R564には計算量とmemoryの誤りがあるため、以下を訂正する。原文は保存する。

- 12 query headsはKV headあたりで、モデル全体は2 KV heads×12=24 query heads。
  width2051、dim256でscoreまたはvalueの各productは24×2051×256 = **12,601,344 MAC/query**。
  両product合計25,202,688 MAC/query、2 FLOP/MACなら50,405,376 FLOP/query。
  4096 queriesでは103,230,210,048 MAC、**206.460420096 GFLOP**であり、103 TFLOPではない。
  hi/loの実matrix issue、padding、softmaxなどはこのuseful arithmeticに含めない。
- current sourceのadmission式をそのまま計算すると、mode0/1/2/3は
  **53956 / 60880 / 70112 / 69600 bytes**。mode2/3を69088 bytesとした原文は誤り。
- current accessorを合計すると、mode1・Direct=falseは**60880 bytes**。
  `final`を1 floatにするDirect=trueなら44500 bytesだが、現在のadmissionは同じ保守的条件。
  原文の57808 bytesは正しくない。CH64を同じ構造へ拡大すると少なくとも
  mode1・Direct=falseの共有メモリは増えるため、実deviceの容量と新しい正確なadmissionが必要。
- `STRATA_PROMPT_ATTN_OLD`はmatrix関数内にはないが、root prefill callerにはあり、
  存在するとmatrix呼び出しをskipする。アプリから強制fallbackを選べないという意味ではない。

matrixの49.455秒区間だけでどのkernelが走ったかは判定しない。
新しいlayout比較用のhost route countsで実matrix/fallback利用を確認する。
prefillのcopy/stager/GEMM/QSAとdecodeのCPU/MTPは区別して調べる。
数学的なshape計算をハードウェアの実測上限や達成throughputとして報告しない。
