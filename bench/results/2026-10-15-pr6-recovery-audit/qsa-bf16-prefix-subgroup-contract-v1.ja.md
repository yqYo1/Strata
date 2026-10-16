# Row-owned SIMD16 FP32 softmax: prospective private candidate

Mainが独立作成する。現行dense16/VNNI BF16配置はそのまま、各subgroupがhead=sとs+8の2回を処理する。laneはcell=laneとlane+16を担当し、全16laneがmax・invalid flag・FP32 sum collectiveを呼ぶ。NaN/Inf scoreはmax・Pから除外して既存diagnosticへORし、missing/tailはP=0。old normalizer=0のalpha=0を保ちempty−inf同士の減算を回避する。lane0のみFP32 max/alpha/normalizerを保存し、Pの全値はBF16化して従来と同じ配列へ置く。既存のQK後とPV前のbarrierを利用し、WG barrier・global scratch・CPU仕事・転送を追加しない。subgroup collective自体の命令/同期コストは増える。

精度の変更点はFP32 normalizerの加算順だけで、sequential32からpair+subgroup treeになる。BF16に丸めたPをsumしない。max/exp/normalizer/accumulatorを16bit化しない。SGLang877be2974aa151a984e52822c028276c2b1e9303のqsa/sparse_attn.py通常CUDA prefill/no-prefix・prefixのtl.max/tl.sumとBF16 operand/FP32 state境界を数値設計参照とする。実装コード・sampleはコピーしない。SYCL2020 group algorithmの全lane収束参加とoperation/parameter一致、指定SIMD16を守る。Mainはreview955後、具体的実装のためR954の15–34のみ取得。R934は45–60以外、R952本文は未読のまま。

事前gateは現行packedモデル6process比較の閉鎖。Mainだけがbuild→traced smoke→physical262144を含むcomponent→same-binary matched時間→必要なモデル検証を直列実行する。独立host tensor oracleと現行1BF16 ULP+2e-6 budget、finite/NaN/empty/mask/alias/extent/guard/lifetime/faultを保つ。経路間FNV一致を合否にしない（tree roundingの相違を記録）；各経路内の同一入力repeat bitsは引き続き要求する。保存値・丸め境界に関わる別の変更は混ぜない。単体physical262144はモデル全消費/RESTOREではない。

共有row stateはlane0だけが読み書きし、old maximumの値をgroup_broadcastで渡す。subgroup reductionをlocal memoryのfenceと仮定しない。normalizerのreadとalpha expはlane0だけで行い、16lane重複計算を避ける。既存WG barrierがQK score→softmax、softmax state/P→PV、PV終了→次tile/final出力の可視性を担う。追加broadcastの命令コストも候補に含める。
