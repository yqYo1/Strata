# MoE窓のC++行振り分け

2026-10-17。Mainによる新規MIT実装。XeStrataのソース・テスト・helperは使用していない。
対象はGPU常駐の連続2-pieceを最大16384行にまとめ、attentionのoperatorとexpertのjobを
最大8192行のままにする長大prefill専用経路。実エンジンへの接続とGPU検証は次の境界。

## 実装済みの範囲

`moe_window_routes.hpp`は、readback完了後の実router ID配列を入力にする。
512 expert、top-k=10、1〜16384行を検証し、token内の重複expertも拒否する。
expert別counts/offsets、安定したsource行一覧、元token-slot順のslot map、最大8192行のjobを作る。
`ne==W`は許可し、たとえば16384行が同expertに集中しても8192行ずつ二回で処理できる。
jobのglobal_rowはhost sourceのoffsetとDownの出力行であり、device sourceのoffsetではない。
GPU sourceへの各uploadは先頭に行う。combineは元のtop-k slot順を維持できる。

公開後のplanは`shared_ptr<const MoeWindowRoutes>`で保持し、copy/move/resize/mutationを公開しない。
ID inputはbuild後に再利用できるが、source uploadのplan ownerはchecked drainまで保持する。
この型だけでGPU drainを強制するわけではなく、実際のenqueue/失敗時の寿命は接続側の責務。

## 実際に行った検証

Clang21.1.8、C++17、O1、AddressSanitizer/UBSan、警告をエラー扱いとしてコンパイル。
行数1/2/511/8191/8192/8193/16383/16384と、10hot/balanced/2hot/前後異なるexpert/
未ソートslot順の5パターン、計40通りを実行した。参照側はexpertごとに元ID列を走査して
source順を独立に作る。各jobの8K source領域を模擬し、expert IDとtoken行の整数witnessを
Downへ置き、元token-slotへ正しく一対一に戻ることを2,892,850 slotで確認した。
geometry、不正ID、token内の重複、job index範囲外など14拒否条件もPASS。
実行は3.346秒、コンパイルは6.890秒、最大観測session RSSは約149MiB。
これはnice19のCPU限定correctness検査であり、性能比較には使わない。

初回は実行ファイルのsymlinkをresolveした結果`nice`のargv[0]が`coreutils`になり、version確認で
終了コード1となった。C++コンパイル・テストは未実行。`host-result-v1.json`と元runnerを保持した。
v2はsymlinkの起動名を保ち、元の失敗receipt hashを記録する。成功で失敗を上書きしていない。
全子プロセスは所有sessionの終了・reapを確認し、GPU runtimeはロードしていない。
既存のforeign CPU測定は変更していない。重いSYCL build/GPU計測との競合は依然避ける。

## 次のエンジン接続

共有helperの境界を次の名前/責務にする。新しいhelperへ既存処理を移し、tokensのsliceとrow=0で
run_implを再呼出しする方法は使わない。現在のエンジンソースとpending profileのhashは維持した。

- `run_layer_major_attention_piece`: 元request tokens、request全長、request相対row/count、pos0、layer、
  GDN/QSA ordinal、明示したGPU residual viewを受け取る。PLE/HC-attention/GDNまたはQSA/HC-writeを
  元の8K順で一度だけ処理し、GPU bankにpost-attentionを書き戻す。PLE履歴の責務をここに限定する。
- `prepare_moe_window_piece`: 2回のattention終了後、各GPU bank viewからHC norm/mix/router/sharedを
  8Kずつ生成し、窓mixed/IDs/weights/shared/sg/injection viewへ保存する。m.embのPLE用途は終了済み。
- `run_moe_window_experts`: 窓IDのchecked readback後、今回のconst planを作る。expertを一度だけ展開し、
  job_sourceをdevice_src先頭へuploadしてgather/GU/swiglu/Downを8K以下で投入する。
  Down global_row、全W行input span、context/device/実allocation extentを投入前に検証する。
  combine後の両HC-writeとchecked drainまでplan、mixed/BO、injectionを生かす。
- `complete_layer_major_window`: 中間層はGPU bankを再コピーせず内部cursorのみ進める。最終層は両pieceの
  writeが完了してから、bankの8K sliceをFP32 m.Rへ展開し既存callback/MTPを順に呼ぶ。

新経路を周期checkpoint consumerに接続する前に同prefixの全層+MTP stateとRESTOREを実証する。
既存未検証のcheckpoint proposalをこの変更で検証済みとは扱わない。

GPU kernel/tensor numerics、GEMM、queue/DMA、実VRAM fit、full physical262144 consume、RESTORE、
速度は未検証。短入力/decode配置の経路や閾値は変更していない。速度採用なし。
