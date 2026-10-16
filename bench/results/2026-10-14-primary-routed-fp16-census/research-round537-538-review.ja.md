# R537/R538のMain確認

2026-10-14。原研究reportはbytes/hashを保持する。Mainのsource照合であり、fixture/buildや
新しいGPU診断を実行していない。v4 full suiteのnative/client/recipeは変更していない。

R537のMTP coverageはcurrent generate.cpp:5944..5985に一致する。completed window W、
実verified rows R、offered D、accepted A、committed C、emitted Eを分け、正常時の
`D=R-W`、`C=W+A`、discarded rows `D-A`とper-layerのW/Rを照合する。
terminal cap/EOSではcommit済みでemitしなかったrowsがあり、EとCを同一視しない。
最終wait_commitの失敗はcompleteではない。per-expert job ntはdispatch Tとは別。
このsourceに限定した未実装のcoverage設計であり、実行receiptではない。

R538が示す「unique expertsがworkerより少ないのでtaskがunderfillし得る」は、current
pool.cpp:535..570のflattened row分割を確認せずに利用しない。
run_split_multi_nativeは`mtasks=3*(workers+host)`、GU `mrows=nb*640`、Down `mrows=nb*2560`を
`g0=mrows*i/mtasks`、`g1=mrows*(i+1)/mtasks`で分割する。canonical5workers+hostでは18tasks。
nb=1でも640/2560 rowsがあるため、expert数が6未満という理由でempty taskにはならない。
job ntによるwork/reduction差、scheduler overheadや実load imbalanceは別に測定する。

Q2_0 Down(type42)はnative_down_rowsのGGML fallbackだけを読むと誤る。
pool.cpp:559..564から`q2_rows_any`へ進み、expert_layout.cpp:111..116はAVX512がなければ
**q2_0_gguf_rows_multi_avx2**を呼ぶ。Ryzen5600Xでscalar-onlyだという候補根拠はない。
既存AVX2 routeの有無を先に照合し、既にあるkernel/grouping/activation reuseを重複実装しない。
external参考は大きなbatchや別format/ISAを含むので、このPCの改善測定には数えない。

この後の順序は、thinking有効benchmark検証→PR#6/Codex元公開ログ・実source/設定の
性能と安全性回復→必要な追加observer/CPU候補の直列検証。新しいsource適用やGPU診断は
live full suiteと同時に行わない。
