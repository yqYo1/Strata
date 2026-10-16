# QSA indexer batch proposal: Main source review

2026-10-15 JST。Sol artifact、R553/R554、baseline native indexer、実prefill callerを読んだ。
Mainはproduct sourceを変更せず、fixture、apply check、build、GPUを実行していない。
patch `466e3a0cdef281de6cffab17a9bad7c674d91e0243e60fed7f4fb11a2c1850aa` は未採用。
現在の長い生成suiteのsource/client/recipeとnativeは其のまま稼働している。

実際のplannerはend=p0+n、completed=[floor(p0/4),floor(end/4))であり、其のblock末尾は
必ずinput内にある。最初のunaligned blockのみ、p0以前のrowを旧tailから読む。
row3は旧tailにならず、全raw rowはchunk-relative indexへ変換する。
bulkが旧tailを読み終わる前にtailを更新しないよう、metadataを同じin-order queueの後へ置く。
各bulk groupは一つのpooled rowだけを書く。spareを途中で書くと次groupと競合するため、
完成blockの直後の一rowだけをmetadataで最後に書く設計になっている。

R553の「position zeroがinitial spareをwrite」という表現は、row1へのstoreとは解釈しない。
baselineもproposalもtoken0で書くのは**deadとprovisional pooled row0**で、pooled row1は
未変更。n1..3ではcompletionがないのでmetadataもspareやblock_posを更新しない。
初めてposition3に達した時に完成row0とspare row1を書き、以後同じ規則が続く。
symbolic fixtureのserial_cellは此の正しいbranchを表現している。

tail slot0..2はendより前の最後の同じremainderのrowへ更新し、inputに其のrowが無ければ
旧valueを保つ。aligned endでも最後の完成blockの最初の三rowが残る。
複数chunk、prefix再開、最終cell262143で同じ規則が成立する。
full input末尾で完成row65535、spare65536を書き、追加physical row65537は未変更となる。
此のsource証明は256Kモデル入力を実行した証拠ではない。

validated capacity<=INT32_MAX、base+capacity<=INT32_MAXにより、block末尾の4*b+3と
base+4*bはint範囲内にある。plannerはint64_tで計算する。対象PCの64-bit size_tを前提とする。
validateのraw/gamma/state/table/contextのspan・非alias条件を保持する。
Angles/planとdevice pointer値はkernel lambdaへcopyされ、host handlerのstack参照は其の
synchronous recording中に消費される。raw regionを後から再利用するkernelは同じqueueの後。
registered tablesのownershipやfault時のteardown保証を此の変更で新たに認定しない。

normalize_rotateのF16入力、four-value sum、RMS reduction、gamma、RoPE arithmeticは
baselineから共有helperへ移された。全256 lanesが二つのlocal barrierとsubgroup reductionに
参加し、その後のhigh-lane returnだけがhelperのreturn valueを持つ形になった。
有効blockのgroup内control flowは一様。sourceから意図した式の変更は見つからなかった。

ただし**shared helper化はflag OFFのserial batchとdecode single appendにも影響する**。
「新batch分岐がOFF」だけで既存decodeの数値／速度が不変とは判断しない。
original untouched build、proposal OFF、proposal ONの三者を区別し、既存single/captured appendの
regression gateとprefillのbatch gateを分ける。prefill gainでdecode退行を正当化しない。

fixtureはactual planning headerを使い、独立したsymbolic serial stateを全pooled/tail/dead/
block_posと比較する。FP conversion、normalization、RoPEやdevice executionは検証しない。
其のfixture自体も未実行。Mainの次のgateは以下となる。

- 独立source identity/apply checkとCPU symbolic fixture、実SYCL build。
- original/OFF/ONの全state words、guard canaries、各chunk後のsingle-token／次batch continuation。
- F16 rounding境界、zero/subnormal、cancellation、subgroup境界のimpulse、非一様gamma/epsilon。
  finite入力のpassにNaN一致を混ぜない。非finiteのdiagnosticは別statusとして記録する。
- none/linear/YaRN、非zero aligned base、table内外、MRoPEのdistinct coordinates。
- 実score/IDs/attentionとfirst logits、snapshot/restore、max-contextの実入力と正常終了／fault。
- 原source/OFF/ONのprefillとdecodeを別々に測り、個別sampleを保持する。

max3 submissionsはsource countで、driver内部のcommand数や速度向上ではない。
既存benchmarkの検証を先に閉じ、user指定のPR #6と公開Codexログの高速実装を監査・回復する。
其の既存batch処理で此の候補が不要ならconsumerを閉じ、重複適用を避ける。

Mainが照合したbase:

| File | SHA-256 |
| --- | --- |
| `src/kernels/sycl/native_qsa_indexer.cpp` | `d87abcf8aece04bc1566f4310e0dee546b7c80e0b711d75d118551b606d38685` |
| `include/strata/kernels/native_qsa_indexer.hpp` | `d040ccbef7943f16c7ce99d8070979b555a8c740950a6f53945b7199226d3fc3` |
| proposed source | `db03c74b3552eab8338c6d20b01bf6ec503581c32146568fa209c0dfb8dd4915` |
| proposed planner | `14580bdb1fb2bfbbbd9a2dbc5de841f3f2211a3394bf522cae1966145b0ecf5d` |

new product plannerはまだ存在せず、artifactのみ。原reportとprovenanceのbytes/hashを保存する。
