# Expert-major FP32集約部品の検証

2026-10-17。Main独自実装。XeStrataの実装は参照していない。
既存の16K window APIを保持し、最大65,536行を扱う別factoryと逆引きを追加した。
新しいGPU部品は登録済み実allocationの範囲、入力と出力の重複、in-order queue、
validated host planを検査する。expert内のtoken重複をhostで拒否し、
expert jobsを同一queueで直列化するためFP32浮動小数atomicsは不要。
GPU側でsource/inverseの不一致をstatusへ記録し、callerはHC出力確定前に検査する。

保存する行列入力はFP16、router/shared scalar gateのproducerはBF16。
Gemm::f16_out16の入力は固定FP16で、boolはBF16出力選択である。
shared_gate領域はwindow全行のFP32 scalar分に修正し、status領域を追加した。
65,536行ではshared region 888,410,624 bytes、独立FP16 mixedは335,544,320 bytes。
この変更だけでは既存R/embedding領域を借用せず、推論経路にもまだ接続しない。

CPU容量検査9ケース・拒否34ケース、routing検査12ケースと拒否検査が、
GCC 13.3のReleaseとUBSanでPASS。
strict SYCL buildと以下の全10 GPUケースがPASSした。

| GPUケース | 行数 | kernel数 | 結果 |
|---|---:|---:|---|
| smoke | 3 | 63 | 全出力・guard PASS |
| numeric | 37 | 743 | expert順FP32 fma/BF16 oracleと全要素一致 |
| short-weights | 1 | 0 | 実allocationの不足をenqueue前に拒否 |
| short-down | 1 | 0 | 実allocationの不足をenqueue前に拒否 |
| legacy-plan | 1 | 0 | 逆引きのない旧factoryを拒否 |
| negative-source | 17 | 342 | status=2、出力未確定・guard PASS |
| negative-slot | 17 | 342 | status=1、出力未確定・guard PASS |
| unwind | 17 | 3 | owner破棄前にchecked drain、数値検証はUNRUN |
| hot65536 | 65536 | 177 | 80 jobs、全出力・guard PASS |
| balanced65536 | 65536 | 1041 | 512 jobs、全出力・guard PASS |

clang 22.1、-O3 -fno-fast-math -ffp-contract=off。
B570 / driver 1.17.39395+14。全ケースでUR API tracingとparameter validationを使い、
提出kernel数と期待値が一致した。各processの実ELF・環境・runtimeをgateで確認し、
全owner終了、boot不変、fresh xe faultなし、devcoredumpなしを確認した。
fixture binary SHA-256: 333d78d16678b88726e050fb27c73de54f0f93cba5672f2787685a64fac719c7。

numericケースでは従来route-slot順に対して30/94,720個のBF16出力が異なる。
expert順のCPU fma oracleとは全て一致する。これはモデル品質同等の証明ではない。
このfixtureはsynthetic Down出力を使い、実GEMM、モデル、性能、256K context、
RESTOREは検証していない。部品のみの採用で、production/performance採用はしない。

Mainの個別receipt、source/compiler/runtime/binary pins、コマンドと実環境は
moe-accum-device-{build,smoke,numeric,short-weights,short-down,legacy-plan,
negative-source,negative-slot,unwind,hot65536,balanced65536}-v1.jsonと
moe-accum-device-cohort-v1.jsonへ保存する。host receiptはmoe-accum-plan-host-v2.json。
次は同じ部品を最大64K windowのエンジン経路へ接続し、旧window2を残して
短いtraced model検査、長入力と品質・状態検査を行う。
