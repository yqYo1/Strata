# 現行row-major QSAのsubgroup softmax候補

Mainの実モデルwindow0/2診断では、MoE窓を有効にしてもQSAの非排他区間は
24.74/24.72秒、うちattentionは約18.8秒で残った。この区間をGPU利用率や
排他的kernel時間とは解釈しない。まずclean MoE比較を閉じ、その後の単体screenで
QSAの現在のserial softmaxを改善できるか調べる。

以前のqsa-bf16-prefix-subgroup-contract-v1.ja.mdのレビュー済み数値設計を、
現在のscore33・row-majorのQ/K/V/P配置へ適用する。packed/VNNIへの変更は混ぜない。
元source0733dbdc8687c032c2510bc3f350aed0279233a7c0ebb867171c2b2f93ac452cから
softmax blockだけを置き換え、private export/template名を分ける。Mainの独立実装。
XeStrataのproject codeは一切使用しない。compiler/runtime dependencyの既存pathだけを使う。

WG128/SIMD16の8subgroupが各2head行を担当し、各laneがcell lとl+16を計算する。
全laneが同じ順にinvalid/max/FP32 probability sum collectiveとold max broadcastへ参加。
lane0だけがpersistent maximum/normalizer/alpha/invalid flagを更新する。
既存WG barrierとlocal allocationを維持し、queue waitやhost演算を追加しない。
FP32確率のsumを計算してから個々の確率をBF16へ保存する。FP32 state/accumulator、
Q/K/V/P/outputのBF16境界、NaN/missing/tail/mask/empty挙動を保つ。
sumの加算順は変わる。既存SGLang/vLLM数値契約を基準にし、経路間bit一致は要求しない。

buildはmoe-window-clean-cohort-v3.jsonが6processとも正常終了した後だけ許可する。
現在の完全engine buildのsource/headerとarchiveを固定し、private correctness fixtureと
same-binary timing fixtureを生成する。baselineは現在のoptimized score33。
build自体はGPUを使用しない。Mainだけがexclusive lock下で実行する。

次は別processのcombined UR tracing+parameter validation付きsmoke、続けてcomponent。
未変更の独立host oracle（1BF16 ULP+2e-6）、全prefix/guard/input不変、NaN診断、
extent/domain/alias拒否、同経路repeat bits、physical262144の単体caseを要求する。
不合格は保存し、後から閾値を緩めて上書きしない。
合格後に診断を外した同じ入力・scratch・packコストのtimingを行う。
単体合格は実モデル品質・全context消費・RESTORE・速度採用の証拠にはしない。
本候補はまだcompile/GPU実行/採用前であり、production sourceは変更していない。
