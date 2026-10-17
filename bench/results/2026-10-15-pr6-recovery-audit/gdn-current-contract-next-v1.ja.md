# 現在のGDNを最適化するための比較契約

R1052／native R1053の短いレビューと、実際のGGUFメタデータ・呼び出しを
基にした次の実装境界。実装・GPU測定・品質検証の完了報告ではない。

対象は長大prefillの`prefill.cpp`から呼ぶ`prefill::gdn_recurrence`。
現モデルはkey group16、value head48、状態幅128、conv4。
現在はpost-convolution Q/K/V、gate、beta、z、gamma、状態がFP32で、
`head % 16`を使う。状態の要素は`[(r*48+head)*128+c]`。
各トークンで状態を更新し、Qとの積を得て、RMSNormとsigmoid(z)を適用し、
FP32出力とFP16出力を生成する。長大prefillラッパーは`single=true`で
最後の状態を保存する。短入力・decodeの既存入口は維持する。

上流のsource-grouped value headを`s=3*k+j`、現在のtiled順を
`l=16*j+k`とすると、`l=16*(s%3)+s/3`、逆は`s=3*(l%16)+l/16`。
状態はSGLang側の`[s,v,r]`と現在の`[r,l,v]`の対応も必要。
実際のGGUFヘッダーは16/48という寸法を証明するが、元のconverter履歴や
すべてのweight軸の並べ替えまでは証明しない。

その履歴がなくても、現在のロード済み入力順を維持したrecurrenceの比較は
進められる。最初のfixtureは、同じpost-convolution Q/K/V、g、beta、z、
gamma、非ゼロ初期状態を、現在のローカル実装と候補に与える。
head/row/columnを区別できる入力を使い、最終状態と全出力、複数チャンクの
連続実行を確認する。weight/head順の変更は、この比較と分けたtag付き
loader fixtureで検証する。

次の候補は、64トークン単位の行列演算でrecurrenceを処理する方式。
数値基準は、固定したSGLang Qwen4Expの実際のchunk経路とする。
BF16のQ/K/V、FP32 gate/beta/carry、chunk内scratchのBF16丸めを区別する。
FP32のローカル累算があることから、すべての中間状態がFP32だと仮定しない。
固定されたauthor configはsigmoidを指定するが、configのない一般的な
SGLang RMSNormGatedの既定swishとは区別する。
既存のFP32実装は診断に残し、丸め差だけでSGLang契約の候補を拒否しない。

最低限の長さは63/64/65、127/128/129、実際の8192入力チャンク。
非ゼロhistory、複数回のcarry更新、極端なgate・beta、出力ガード、入力不変、
alias・extent・domain、終了時の状態を含める。物理262144の全消費・RESTOREは
別の実モデル検証として残す。LLM全体の同一token出力は普遍的な採用条件にしない。

16-bit化では、projection／convのproducerからrecurrenceと次のprojectionの
consumerまで、実際のdtypeを合意する。現在のFP16出力をBF16として解釈しない。
各演算の前で全量FP32へ戻す方式だけで完了にしない。必要なFP32の局所累算、
gate、persistent carryと、保存・転送するactivationの型は分けて記録する。
FP32 conv historyを共有decodeへ渡す場合も、phase境界での表現を検証する。

現プロファイルの約8秒というGDN recurrenceのdevice集計はwhole-process値。
prefill／decode／warmupの内訳と全体への効果は新しい測定で確認し、
その値だけをprefillの短縮可能時間やハードウェア使用率にはしない。
コードの移植前にはSGLang内のFLA由来ファイルの出自・licenseも確認する。
この計画は独立実装・比較契約の選定で、外部コードの移植ではない。

根拠は[実GGUFメタデータ](gdn-target-gguf-metadata-v1.json)、
[R1053 index proposal](research-round1053-index-proposal-v1.json)、
`src/kernels/sycl/fused_gdn.cpp` SHA256
`70abf4fd1359477a2e170b54f2172ba148a2ae21306e8145e070ebd0db7d995f`。
Mainは元のR1052レポート本文を自動読み込みせず、レビューと必要なソースから
この境界を選定した。
