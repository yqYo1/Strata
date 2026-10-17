# GPU常駐2-piece MoE窓: エンジン接続候補

2026-10-17、Mainの新規MIT実装。baseはfeature/syclの6c923e05。
基準測定の一次worktreeを固定するため、検証用verify/moe-windowで実装した。未完了の検証ブランチであり、
GPU検証後にfeature/syclへ統合し削除する。既存推論の採用や速度更新ではない。

## 実装と元処理の保存

起動時環境`STRATA_SYCL_PREFILL_MOE_WINDOW=2`で選択、既定は0。
native16 BF16・8192operator・scratch alias・単一queue・native layer cacheが前提。
既存run_implのtokens全体/request相対c0/絶対p0と、PLE/GDN/QSA本体をそのまま使用する。
共有attention本体をコピーして別の演算を作る代わりに、そのhalfループを窓pieceではattention一回に限定し、
完了後に既存GPU row bankへpost-attentionを保存する。host token/step/PLEの再利用前にchecked drainする。
連続する二pieceがともにGPU bankにある場合だけペア化。GPU/RAM境界、RAM側、単独末尾は既存8K経路。
これにより短入力/decode配置の閾値も変更しない。

両attentionが終了したら`run_moe_window`がFFNだけを処理する。8KずつHC norm/mix、router/sharedを生成し、
窓のmixed/IDs/weights/shared/scalar gate/injectionに保存。router IDだけをchecked readbackして
R970でレビューされたconst C++ planを作る。全窓source mapとslot mapを各一回H2D。
各active expertを一度FP16展開し、最大8192行のjobでgather/GU/swiglu/Downを実行する。
combineが最後のmixed読み出しより後に同queueでBOへ用途を変える。両HC writeのchecked drainまで
全host map/typed viewを保持し、それからroute ownerを解放する。
中間層はcursorのみ進めてbankの自己コピーを避け、最終層はbank sliceを8KずつFP32 m.Rへ展開し、
既存callback/MTPへ渡す。全windowの両writeが完了するまでcallbackを呼ばない。

## 所有者・境界・診断

`MoeWindowContext`は外側のchecked leaseより前に宣言したrequest owner。
途中失敗で早期returnしてもID/readback/route vectorsとbank ownerはleaseのdrain完了まで生き、
drain失敗なら既存leaseがownerを保持したままabortする。session状態のrollbackの証明とは別。
新しいviewは元のm.*ポインタを変更せず局所に作り、実際のregion_bytesと登録された全span/context/deviceを
検証する。source全W×10 view、mixed全W行、jobごとのsource/Down spanを投入前に確認する。
region使用998443520 B、m.embのmixed/BOは既存83886080 B以内。別GPU allocationを増やさない。
host admissionにはID/map/job用2MiBを追加する。guard256 Bは予約のみでcanary検査とは呼ばない。

coarse/QSA profileは各attention pieceのchecked drainで既存eventを退役し、MoE窓の区間を別sampleに記録する。
coarseのMoE sample数は窓とfallback pieceの数であり、attentionのpiece数と同一とは扱わない。
旧shape censusはchunkごとのcountsを仮定するため、この候補と同時指定は明示拒否する。
実window/dequant/job/source H2D bytesの合計を出力する。これらは未実行であり現時点の実測値ではない。

`requires_prefix_checkpoints`は既定true。serverではprompt_cache>0かつprompt_cache_every>0ならtrueにし、
そのconsumerには窓を接続しない。CLIのresidual/MTP/NLL callbackではfalse。
このgateは既存layer-majorの周期checkpoint問題を修復・検証したという意味ではなく、
未検証の新窓経路をそこへ適用しないためのもの。全層+MTP同prefix保存/RESTOREは依然未検証。

## 実行した検査と残る検査

`run_moe_window_syntax_v1.py`で、既存buildのcompile_commandsにあるIntel LLVM/oneMKL/SYCL/ggml設定を使い、
実際のprefill.cppとgenerate.cppをnice19、O0、fsyntax-only、Wall/Wextra/Werrorで検査し両方PASS。
LLVM依存は既存のXeStrata-path配下にあるが、使用したのはcompiler/runtime依存のみ。
XeStrataのプロジェクトコード、helper、testは参照・使用していない。
所要時間は18.891秒/16.442秒、最大観測session RSSは約411MiB。所有子processの終了/reapを確認した。
これはcodegen/link/実GPU/実fit/演算/queue寿命/262144 consume/RESTORE/速度の検証ではない。
C++ route headerは40case/2892850slot/14拒否のASan+UBSan検査済み候補とbyte同一である。

foreign CPU llama-bench PID3431768/start52544839は実装開始時にもliveで、変更していない。
重いbuild、GPU診断と速度測定は競合させない。一次worktreeの基準binary/sourceも固定した。
次は独立source audit、基準profileの完了後のcandidate build、小さいtraced GPU検査、
biased ne==W/tail/nonzero pos0/PLE/MTP/cancel/fault、実数値とstate、full physical262144+RESTORE。
その後、公式bench.shの32K以上/thinking有効・複数回で同一binaryの0/2を比較する。
PP600/1000、TG70は未達。新しい速度値や採用判断なし。
