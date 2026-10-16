# 実モデルne分布に基づく次の候補

2026-10-15。R627/R628とIQ3_XXS subgroup v2のsource handoffを読み、次の実行順を整理した。
現在のclean full OFF比較が閉じるまでbuild/test/GPU/cleanupを増やさない。

32,088の実prefill prefixでは、ne<=16がGU/Down各46,003 calls(55.51%)で、
各210,579 routed rows(1.37%)。ne>512はcall8.76%、論理FLOPs76.02%。
小neの起動・展開費と大neの行列計算費を区別する材料になるが、どちらがwall支配的かは
このcensusから決まらない。FLOP比だけで小neの改善を低優先にする判断もまだできない。

R627の形式名に訂正: Down type42はこのforkのcustom Q2_0で、IQ4_XSではない。
R628末尾の「GU18が最大」も訂正: GU18は36.66%、GU22は41.46%で後者が多い。
元レポートの個別countは変更しない。

次の部品候補は、既存5形式のclean実モデル比較の後に評価する。

1. IQ3_XXS subgroup v2は現行の選択的5役割版への直接差分へ更新された。
   public validation後にtype18 FP16だけをSG16/SG32へdispatchし、他5役割のdispatcherは
   byte同一で維持する。collective参加、src0 broadcast、cluster別select、各32値groupの
   aux/db、乗算順、unsupported事前fallbackをsource reviewした。全GPU gateは未実行。
   部品正しさ、実lowering、独立process timing、モデルの順にMainが実行する。
2. packed IQ XMX v2は本体未適用・未build。現行M<=1280は全形状向けの代表範囲ではないが、
   小neだけを選ぶ実装候補としては明示fallbackとともに維持できる。
   幅を4096へ拡張すれば速くなるという証拠はない。先に小neと代表的大neの境界で
   FP16/FP32の全配列誤差、queue寿命、actual device admissionを確認する。
   shared-USMの一件cold wallはclean device-resident速度比較に転用しない。

既存native MMVQ/MMQ grouped案はQ8 activationへの変更を伴う別候補で、FP16 oneMKLと
同じ数値処理とは扱わない。現在のlayer-major MMQ拒否を単に解除して実行しない。
全prefix MoE順序変更と同様、task能力・logit誤差・所有寿命の検証が必要。

R625は過去のCPU predecode fixtureがlossless byte+original scaleで、FP16 cacheの
測定ではなかったことを明確化した。8GiB request-coldはfill費を回収できず未採用のまま。
20/48GiBのhit率は未測定。R626の1.55GiBも過去のPrefill scratch会計値で、free VRAMや
decode cache解放量ではない。両者を根拠に未確認の大容量割当は行わない。

進行中の正式比較は`selective-iq-clean-bench-v1/check_selective_iq_bench_v1.py full 0`。
生きている同じ所有jobを観測し、終わるまで再起動しない。成功すれば同じrecipeのfull1、
失敗すれば元receiptと原因を保存する。clean比較の完了前に性能採用・default有効化はしない。
