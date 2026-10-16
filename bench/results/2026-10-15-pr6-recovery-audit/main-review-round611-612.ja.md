# Main review: compiler alternative and independent head-split audit

2026-10-15。両LunaのFINAL後にMainが照合した。build/test/GPU実行なし。
現在のDirect0/1比較を中断したりcompiler/runtimeを切り替える操作はしていない。

## R611: compilerを変える余地はあるが3.1倍は転用しない

Mainもpinned llama.cpp公式docをwebで確認した。oneAPI2026.1と2025.3 releaseのB570比較の
記述はあるが、model、prompt長、詳細なbuild/sample条件がない。Strataの改善率には使わない。
<https://github.com/ggml-org/llama.cpp/blob/a0e6dc7384edb8f5d927155e1d26c10b4a310e6d/docs/backend/SYCL.md>

現行primary CMakeCacheと保存済みcompiler-version stdoutをMainが確認した。
compilerはcustom DPC++ **7.1.0 pre-release / Clang22.1.0**、LLVM504366f4...で、
SYCL device AOT指定は空、oneMKLは既に**2026.1**である。保存場所のv711というpath名を
actual compiler versionの代わりにしない。host CPU fixtureのc++13.3もSYCL compilerとは別。

R611が確認した別のinstalled icpx2026.1は、source/oneMKLを固定して比較する候補になる。
ただしSYCL/UR ABIを無理に混在させず、同じruntimeを使えない場合はcompiler+runtime stackの
比較と明記する。oneDNNの上流library差と、StrataのoneMKL/独自joint_matrix codegenを分ける。
まず別build directoryで小さな互換・数値gate、次に同一34K実モデル、改善が見えた候補だけ
公式benchの反復へ進める。現行測定binaryとbuild treeを上書きしない。
hardware専用AOT指定も別の変数として測り、compiler比較へ同時に混ぜない。

## R612: source監査と実context試験を区別する

8+4 head mapping、外部24head stride、padのstore guard、subgroup一様QK分岐、
全WG barrier、共通12head Q scaleとCH32演算順について、独立監査もsource欠陥を見つけなかった。
これは実行合格ではない。CH64はchunk変更による別の数値gateを保ち、bitwise同一としない。

追加のcoverage gapは妥当である。現probeのcap262144 / high-ID262143はactive width65で、
最終full-count chunkには達していない。最小の次fixtureはnq1または3、width262143/262144、
大半のIDをmissingとしterminal IDだけを小physical poolへmapするものになる。
これなら巨大なphysical KVを割り当てず末尾count/indexを確認できるが、**full physical256Kや
実modelのcontext消費/RESTOREの代用にはならない**。最初の28case検証とこの追加scopeを分け、
full-context認定前に双方と実physical試験の証拠を要求する。

kernelの初回実行前にMain owner adapter、実kernel選択、compile admissionを確認する。
既存v2 controllerのkillpg/post-reap group検査を使わないMain判断は変更しない。
