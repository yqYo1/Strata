# 長文prefillの転送提出とMoE host境界

2026-10-16 JST。Mainが独立にCPU観測v3を実装した。既存v2の3区間に、route公開+既存queue wait、
48層のupload提出、host/device residual入力/保存の計6区分を足した。default OFF、GPU演算、
同期、配置、runtime環境の順を変更していない。Host区間の明示wait重複は差分として分離する。
source diff d190fba0fa1b591c1bc95c7a9c42a0d1342b40a5f09f6c1e69c73feb9b417223。
base c8a682103b43b02e8cec458ffac29ec696e85023の実blob membershipをMain ownerで検証した。

19 host modes PASS。新しいall-kindsは全9区分とroute内部waitのCPU/wall partitionを実headerで検証し、
all-kinds-offはCPU clock calls0を確認した。従来の異常時計/不正nesting/exception/errno検証も維持。
同じcompiler/runtimeと既存buildでnative/runtime/compat/GEMM probeをbuild。GPU小検証はMainのlock下で
flushed UR trace/parameter validationを使い正常終了。16 GEMMsは全output analytic exact/guard/input不変性
PASS、verboseが全GPU0を報告した。実モデル全callのCPU fallback除外はしていない。
native SHA256 8fc084391da50ece4ad0fb0b05d0b3b1d4d8ce1853b582171d70ebfcda037ff3。

## 実78K結果

公式bench.sh prefill-only、実API78453/batched78452、C8192、FP32、GPU32768行、native/canonical bank1、
Direct1/専用radix1/no-token-graph、thinking有効。TRACE/validation/clock診断でありclean性能標本ではない。
48層10chunks、route/group480、upload48、residual入力/保存はhost各282/device各188回。
正常server/native exit0、全owned identities退去、boot不変、新規fault0、devcoredump無し。

| 区分 | CPU秒 | 経過秒 |
| --- | ---: | ---: |
| 全体 | 218.680931845 | 225.873994809 |
| 明示的GPU wait | 165.704323602 | 166.348116325 |
| source slab準備 | 5.321485532 | 5.356134028 |
| host groupingとmap提出 | 0.196490251 | 0.199851155 |
| GEMM call | 7.366303896 | 7.921806303 |
| route公開+既存queue wait（重複区間） | 131.072101375 | 131.610017977 |
| layer weight upload提出 | 5.718998912 | 5.870030946 |
| host residual入力提出 | 9.430725341 | 9.590991498 |
| host residual保存提出 | 16.362739061 | 16.565061812 |
| device residual入力提出 | 0.004997142 | 0.005601817 |
| device residual保存提出 | 0.013337220 | 0.014085252 |
| unclassified | 8.552161108 | 13.991468966 |

routeのinclusive131.610秒は明示wait166.348秒にも含まれるので、足し合わせない。
route自身のwait差引後はCPU0.009369780秒/wall0.010846707秒。
そのqueue waitは先行attention/GDN/共有expert等のGPU演算もretireする。131.610秒が
「ID転送にかかった時間」「GPU groupingで消せる時間」ではない。GPU側のmap取得完了だけも未測定。
MoEのID+slot/srcは約0.452GB論理payload/480境界。mapped hostUSMでもホスト往復が残り、
旧0.165CPU秒だけを根拠に往復改善を低優先とした判断は修正した。own-moe-host-roundtrip-audit-v1.ja.md参照。

host residualの転送提出の中にCPU25.793464402秒（wall26.156053310秒）がある。
これはCPU math25.79秒を証明せず、copy API内部のruntime/driver処理や暗黙wait/staging等を未分離で含む。
queue.memcpy/Asyncという名前だけで「CPUは転送指示だけをしてすぐ帰る」とは言えない。
upload提出にもCPU5.718998912秒があり、unclassifiedを今回8.552161108秒まで分割できた。
別runのv2 unclassified37.348秒から単純差し引きで改善量を出さない。今回もsource/GEMM数は記録したが
CPU秒の削減をprefill latency削減と同一視しない。CPUの飽和はGPU math fallbackの証拠にならない。

次は同じcompat copy、queue、サイズで通常host対hostUSMのH2D/D2Hを、allocation/setup、提出CPU/wall、
waitCPU/wall、全体に分ける。hostUSM化だけでmodelが速いとは未判定。FP16 residual全GPU案は
転送そのものを減らす別候補であり、quality、容量admission、実context境界の検証を省かない。

## 検査コードの失敗と保存状態

run_own_prefill_host_profile_model_v3.pyにはactual version3の検査後に古いversion2判定を二箇所残した。
Mainの生成時の見落としであり、original receiptはpassed=false/complete=falseのまま保持している。
測定native/serverとGPU検証が失敗したわけではない。元recipe/hash/失敗箇所を変更せず、
assess_own_prefill_host_profile_model_v3.pyで同じ生出力、元receipt、source/実入力、48progress、
全9区分count、正常ownership/fault、CPU/wall partitionを再検証した。別assessmentはPASS。
これは新しいGPU測定や原FAILのPASS書換えではない。旧recipeは当該失敗の再現証拠として保存する。

raw閉鎖root2個はMain lock下でcompact化後に退役した。86 files/1461700 logical bytes。
original FAIL/個別測定値/hashを維持し、closed-own-prefill-host-profile-compact-v3.jsonへprogress、CPU lines、
wrapper、build stdout、GPU/host検証evidenceを抽出した。物理空き容量の増加とは主張しない。

## Orca調査と継続事項

Orca run_c8f478972ec8でLuna R761–763の3件を開始・完了し、全reportをMainが全文読んだ。
stable GPU grouping、mapped kernel対明示copy、非LLM GPU pipelineからの知識をファイルで受け取った。
調査開始後のsteering無し。Mainはこの実装/実GPU測定を並行した。全Taskのworker_doneとsucceededを
照合しrelease要求、Orcaはuser_takeoverによるretainedを返した。未処理reclaimable workersは0。
原文とMain訂正/採用判断を分け、次の短い独立scopeでもOrcaを使用する方針をAGENTS.mdへ記録した。

新たな推論最適化/format変更の性能採用はまだ行っていない。clean PPの既存radix1中央値349.3対control339.2
（各3標本）は以前の別比較であり、今回の診断と混ぜない。反復600超/1000 PPは未達、decodeの新測定も無い。
heads8/NLL案は未適用/未build/未GPU検証のまま。physical262144、RESTORE、FP16 task qualityも未完了。
Niko非推論機能portとdefault反映も未完了で、次の実upstream checkは前回から約48時間の予定を維持する。
