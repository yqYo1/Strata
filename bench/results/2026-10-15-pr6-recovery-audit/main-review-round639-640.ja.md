# WaveメモリとMTP長の次の判断

2026-10-15。LunaのR639とR640をFINAL後にMainが読んだ。調査中はMainがCPU検証用sourceを
準備し、同じexec16956のrunningと生成進行を確認した。GPU/CPU tests、build、cleanupは
追加していない。現在の正式full OFFはまだ長文生成の途中。新しい採用・速度改善はない。
最後の確認ではvaried-prose初回が4500 tokensのthinkingからansweringへ移り、4549 tokensまで
進んだ。まだrequest完了ではないため、decode速度・MTP分類・本文条件の確定値にはしない。

## Waveの出力tileは無条件の解決にならない

R639 v1は存在しないsource path、別のNの容量例、提案waveを現行実装と呼ぶ記述があった。
MainはFINAL後に訂正を依頼し、v2を受領した。v1は訂正対象として保持し、次の判断はv2を使う。
実rootは`src/prefill/prefill.cpp`のchunk単位Dmと`src/prefill/sycl/postops.cpp`のcombine。
実N=2560、K=10なのでFP32 Dmは102400 B/token、chunk4096は400 MiB、8192は800 MiB。

expert-outerで一度の展開を使い回すなら未combineの後続tile出力を保持する必要がある。
tile-outerで出力を退役させるなら、expertの再展開またはdecoded weightsの別cacheが必要。
全512 expertのFP16 GU/Down pairは5033164800 Bであり、既存DQ=2 pairの19660800 Bとは違う。
単なるtile化がwave全体のDm保持と再展開の両方を消すという前提は採らない。

slot-major FMAやSCCによる依存順処理は候補に残るが、現在のtop-k順FMAに必要なbufferと
展開回数のtradeoffが未確定。SCCを取る新observerは今は追加しない。準備済みq2/q4 censusを
先にCPU検証し、その実モデルのunion/row分布を見てwaveを続けるか判断する。

## MTPは既存設定を先に試せる

R640を受けてMainは`generate.cpp:2820-2831,5332-5334,6016-6035,6103-6131`、
`mtp.cpp:438-470,1140-1191`、`draft_policy.cpp`を再確認した。

- loaded MTPのmax_tは`mtp.load(..., o.spec, ...)`で決まる。`--spec 4`を固定し、
  `--mtp-max-t 2/3/4`だけを変える比較では、loaded max_tとVerifierの最大容量は4のまま。
  S_mtpと`set_max_drafts(S_mtp-1)`が実行するverify/draft深さを制限する。
  suffixは0、min-pは0、coupledはoffを揃えれば、最大buffer容量を変えず既存設定で試せる。
  `--spec`自体を2/3へ変える比較と混ぜない。lazy captureされる実graph集合は同一とは限らない。
- expert detachの`max_t_==4`とwrapperの`o.spec==4`は上の比較では維持される。
  `--mtp-max-t 2`だけでこのgateが不適合になるとは解釈しない。
- verify Tはtarget row込み。T=4は最大3 draftで、2/3は最大1/2 draft。
  全armのserve first-windowはT=1。`set_max_drafts(0)`でもdraftの先頭1回は実行されるため、
  T=1 capをMTP-free baselineにしない。真のno-MTPは別の`--spec 0`設定になる。
- 現行min-pはdrafter自身の確率による可変T。DraftPolicyはMTPのaccepted+1とround costを
  EMAで保持するが、MTPのTを選ぶ機能ではなくsuffix候補との比較に使う。

まず同一の長文input/body、source/binary、max-context、KV、expert-cache、CPU/PCIe、
MTP4 loaded容量、thinking、samplerで固定幅2/3/4を比較する。現在のfull OFF/ON比較とは
別のexperimentとする。新T graphの初回診断はログ付き、速度は別のclean processで測る。
固定幅で差がある証拠が出るまでは、online controllerや追加の学習counterを実装しない。

Mainも[vLLMのadaptive verification文書](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/adaptive_verification.md)、
[SGLangのadaptive speculative decoding文書](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/adaptive_speculative_decoding.mdx)、
[vLLM v0.31.0のMoE reduction source](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/layers/fused_moe/topk_weight_and_reduce.py)
をwebで開いた。適応長と出力側reduceの参考であり、Arcでの速度・同じFMA順・状態復元を
証明するものではない。mainを指す二文書はmoving referenceで、実装採用時には改めてpinする。

既存の境界`p+T>max_context`はvariable T/最後の残りcellにも関係する。小さいTの試行だけで
全context消費やRESTOREが安全と結論しない。物理262144、quality、RESTOREと1000/70は未達。

## 次の順序

現在のfull OFFを同じhandleで観測し、正常/異常の元statusと全individual sampleを保存する。
閉じた境界で元nonce回復と入力toolのCPU検証を行い、実入力が揃うOFF/ON比較へ進む。
wave CPU検証のsourceは`main-wave-host-plan-v2.ja.md`に保存済みで、同じlockの下で直列に行う。
その後、bounded model census、既存chunk4096/8192比較、MTP固定幅比較を、得た結果に応じて
優先する。未実行の提案を採用済みと扱わず、失敗した旧suiteや未解決のdriver resetも保持する。
