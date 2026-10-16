# R741/R742のMain確認

2026-10-16 JST。原報告のbyteとSHAは変更しない。両者は読み取り専用調査で、
コンパイル・GPU・品質・性能の実証ではない。

R741: 現行FP16 residualは非最終layer境界の保存だけをFP16にし、次の計算はFP32に戻す。
actual78453入力のbatched78452行を全GPUに要求する場合はR_GPU=78452を明示し、
実planのhost0/staging0とadmissionを別途確認する。1,606,696,960Bは要求byteの計算で、
実測した空き容量ではない。matched placementのdtype比較と、全GPU化の配置比較は分ける。

補足: 原報告が挙げた`check_selected_prefill_logits.py long`は歴史的34603入力のrecipe。
現在はそのbuild receiptのsource pinが30 fileでstaleなため、変更なしでは実行可能としない。
現在入力に適合する新runner・source/binary/runtime/recipeのpinと所有監督が必要。
19 selected位置はdense品質の代用にしない。dense NLL v2は独立source案を受領した段階で、
Main全diff review・適用・build・host/GPU検証は未実施。

R742: CMakeのbmg_g21 AOT設定はsourceに存在するが、実backend resolver・image選択・
実行正しさ・速度は未検証。実modelがmapしたIGCは`/usr/local/lib`の2.41.5であり、
dpkgの2.40.13は実使用版の証明にならない。cold compile時間をphase差分から推定しない。

補正: 提案target `strata_prompt_k_cellmajor_test`は存在せず、実targetは
`sycl_prompt_k_cellmajor_test`。`sycl_prompt_attention_test`はlegacy環境変更fixtureのため
reviewせず再利用しない。公式compiler install外の全toolを禁止する条件はユーザー指示ではない。
公式system ocloc等はresolver/path/hash/loaded librariesを固定すれば検討可能。
禁止対象はXeStrataの実装sourceであり、同じdirectory配下に設置された公式toolchainそのもの
とは区別する。新しいAOT/JITの結果や上流確認をこの読取りから主張しない。
