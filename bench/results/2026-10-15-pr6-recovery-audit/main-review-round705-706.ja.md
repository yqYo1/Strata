# R705–706のMain確認と次の実装

2026-10-15。Luna二体のread-only報告を受領。過去index/reportの場所を共有し、Mainはその間に
最後のQSA clean測定を管理、各3processの比較を完了し、NativeDense caller-connected build recipeを作成した。
両researcherはsource/build/test/GPUを変更・実行していない。以下は実装やGPU合格ではない。

## 一層だけのQSA KV配置

R705の実source参照をMainも`src/core/layer.cpp:496–670`とprefill/verifyの該当箇所で確認した。
mode1には各層の全KV pinned-host copyが既にあり、prefixのstageをchunkごとに繰り返す。
新規promptのactive QSA全KV bankを最初からGPUで育てれば、過去prefixの反復H2Dを避けられる。
appendのRAM mirrorを保つ構成では、終了時に改めて全KVをD2Hする必要もない。
decode用のslot/page mapと全indexer ownerを安全に保持・復帰する境界は別に必要。
R705がNativeDense v2をexpert/native bankとまとめた表現は正確でない。v2が移すのはnative denseで、
expert cacheは既存layer-majorの独立owner、canonical全層は残る。

R705の容量段落は古いmetadata-v2のworkspaceを用い、さらにseparate-bankの異なるchunk例を混ぜた。
最新`prefill-capacity-alias-delta-v1.json`の4K/aliasON/512MiB仮定では、
全256K hidden＋一層QSA後はcombined-bank K8V8 1581165808 B (1.4726 GiB)、
separate-max-bankは1516154224 B (1.4120 GiB)。両者の差は65011584 Bで、約1.2 GiBではない。
前turnの容量回答は最新receiptを使用しており、その数値は変更しない。

最初の実装で全12層indexerを保持するなら、厳密なactive-QSA-only仮定より11層分多い。
production geometryでは一層のtail1536 + dead512 + pooled33555456 + block_pos16 =33557520 B。
11層の追加は369132720 B (約0.344 GiB)。page tables・map・step等の残存量も別途加算する。
この追加だけなら4Kの5GiB hidden構想は成立する余地があるが、現在の大きなsession arenaから
inactive KV poolを解放できる実owner分割がまだ無い。現NativeDense v2だけで達成したとは言わない。

## 16bit hiddenの評価

R706は既存FP16/selected-logit/NLL案の再利用を推奨し、新しい同種callbackを増やす提案ではない。
Mainも一次資料を再確認した。Qwen architecture report §2.2 Inference EfficiencyはFP8 residual保存を
評価したと記すが、当forkのIQ3 checkpoint・FP16/BF16・長文contextの許容劣化率は示していない。
https://arxiv.org/html/2608.30320v1
IntelのBF16説明のexponent/fraction比較も確認した。
https://www.intel.com/content/www/us/en/developer/articles/technical/intel-deep-learning-boost-new-instruction-bfloat16.html
trainingのBF16報告を当該inference品質の証拠にしない。
https://arxiv.org/abs/1905.12322

MainはFP32 referenceを維持し、同一token列・同一配置でまず既存のselected logitsによる
FP16再screenを行う方針。出力token完全一致を採用条件にしない。既存NLL案の全row評価と
独立したtask screenを後続で使い、prompt内の相関token数を独立task標本数と数えない。
候補が少標本screenを通ることと、全品質・256K・RESTOREの合格を区別する。

次のMain作業は、NativeDense v2をdefaultOFFで実rootへ適用し、全ABI callerと
実GGUF payload/restore・allocation故障fixtureを別buildへcompileすること。
ユーザーのcombined一層weight仮定とKV owner変更はまだ未実装。物理context producer v1の
full-resident-only制限は実際のmode1検証を覆わないため、別Solへv2修正を依頼済み。
