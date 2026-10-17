# window10: 約78K入力の最初の実モデル比較

新規full engine339c1ea17e297ee291461e41f6aa6014985a0cdb2a3ef33783a72660a2ebb406を
用い、同じ公式bench.sh要求、thinking有効、入力78453/batched78452、context131072、
INT8 KV、chunk8192、全BF16 residualをGPUに置いてwindow8/10を順に測った。
Ryzen5600X、128GiB RAM、ArcB57010GiB、xe1.17.39395+14。

|経路|PP(tok/s)|MoE windows|expert重み展開|jobs|
|---|---:|---:|---:|---:|
|mode8/最大65536行|601.3|96|31665|34224|
|mode10/今回78452行|615.2|48|18394|21418|

単回の診断値で約2.31%改善。mode10は入力全体を一つのMoE windowへまとめた。
job/operatorの上限8192行、BF16 GU/Down出力、FP16 matrix input、expert順FP32 FMAは同じ。
同一要求hash、再利用0、24thinking tokenの完了、全48layer、正常終了、所有process消滅、
新規GPU fault/dumpなしを確認した。出力の完全一致を合否条件にはしていない。

admission実測は共有arena1180698624 B、要求1022996480 B、mixed401674240 B。
開始時free4257439744 Bから既存予約と512MiB headroomを含む3668013984 Bを差し引いても
mixedが収まった。端数78452行のalignmentを含む必要量を個別に再計算し、fallbackなしを確認した。
これは今回の構成でのfitであり、全context構成の一般保証ではない。

非排他的MoE queue intervalは24.2404→20.8916秒、QSAは58.1787→58.1553秒、
GDNは18.0917→18.0913秒。unclassified thread CPUは20.9432→17.9861秒だった。
これらをGPUbusy、CPU律速、JIT時間と読み替えない。
次は同一engineの4つのfresh要求をログなしで測り、最初の長文要求と後続3件を分ける。
この比較では一般品質、持続decode、物理262144消費、RESTOREは未検証。

コンパイル初期コストについても調査した。現在のbuildは、vLLM GEMMのBMG G21向け
AOT objectを分離し、その他のengineにはJIT経路を残す構成である。
STRATA_SYCL_DEVICE_ARCHが空であることを、全てのkernelがJITという根拠にしない。
現在のCMakeはvLLM native16有効時のglobal AOT flagを明示的に拒否する。
そのflagだけを追加して使えるという提案は採用せず、first/subsequent測定を先に行う。
