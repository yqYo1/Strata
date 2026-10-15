# vLLM Xe2 native16 componentのMainレビュー

R810の公開ヘッダー、adapter CPP、派生core、全fixture、CMake、README、provenanceをMainが通読。固定vLLM源の実coreと変換点も確認した。実装はsource-only、まだビルド/GPU/ISA/数値/速度/モデル品質を検証しておらず、製品への組み込みもしていない。

実coreは vLLM XPU6b30c9b9 の `MoE::xe_gemm` から残した block2D loads/prefetch→fragment reorder→float DPAS accumulation→16bit store。Intel BSD3 noticeを保持し、scale/bias/FP8とgrouped/Torchを除去、初期prefetchを実K tile数で制限する。無印policyはコピーせず、公的APIによる独自の2geometryを使う。TLA87f68506は公式ghq cloneの固定detached worktree。207依存headerの個別ライセンス/notice/hash inventoryがあり、192BSD3+15Apache LLVM exception。XeStrata実装は一切の参照/移植元にしない。

4IO形式（FP16/BF16 input、独立FP16/BF16 output）×2tile。localFP32蓄積、globalF32出力/pack/kernel/CPU演算fallbackなしというソース構造。これはemitted ISAや実性能の証明ではない。実Runtime/queueを借りる明示的なcaller lifetime契約で、global/default queue・device0・環境切り替えを呼ばない。供給元Runtimeがqueueのasync handlerを持つことはcaller責任。統合時にもこの条件を保持する。

N%64/K%32、64B base、16B pitch、実logical M surface、登録extent/context/overlap、int座標とsurface上限をenqueue前に確認。split barrierは全lane同じK loopで通り、M端数は実surface boundsに任せる。幅/高さ/pitchの実GPU処理、短K、混合outputとreorderは未検証。完全rawbuffer rectangleを検証してもハードウェアの端数動作を実測したことにはならない。

fixtureは独立FP64/dyadic closed-form oracle、RNE/specials/subnormal、guards、入力不変、同一queue BF16 chain。全host/USM ownerより後にCheckedDrainを生成し、例外時も完了確認前に解放しない。全suiteは未実行。strict denormal gateが失敗しても旧68testは変更せず、別の数値契約として実値/影響を調べて判断する。

R816にMain専用controllerとpaired operator sourceの準備を依頼。既存F32、F32+cast、遅いcustom16、vLLM2tileを同じデータ/queueで比較し、各shape/inputにつき3独立processと各route3timed submitを区別する。Mainだけがビルド/実行を行う。将来QSA native16統合（R818 source-only）と、物理262144/RESTORE/モデル能力の検証も別途必要。default FP32は維持。
