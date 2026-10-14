# Prefill admissionのcount/carve整合案

2026-10-14。Mainの未適用・未コンパイル・未実行・未採用案。live suiteのengine/client/recipeは
変更していない。R540のsource計算を手がかりに、current bytes_needed/carveを照合した。
この案は推論精度、kernel、queue、copy、配置を変更しない。

carveは`xn`をgr_unfused時だけ確保し、`grs`は両方のpathで確保する。
bytes_neededのprefixはxnを常に数え、grsを数えていない。R540の「余分なxn」の説明には
後者の不足も加える必要がある。Alloc::takeは各extentに256byteの余白を足して256byteへ丸める。
元の8MiB alignment slackを温存し、prefixを実際の確保条件/順序に合わせる。

| chunk | fused時の見積り減少 | unfused時の見積り増加 | 同じchunkの実確保の変化 |
| --- | ---: | ---: | ---: |
| 4096 | 167706624 B（159.9375 MiB） | 65792 B | 0 |
| 8192 | 335413248 B（319.875 MiB） | 131328 B | 0 |

単体でmemoryをfreeする変更ではない。callerのauto chunk/borrow admissionが過大な見積りで
より大きいchunkを拒否している場合に、実際に必要な量で判断するための案。
bytes_neededはgenerate.cppのordinary serve admission、stage borrow、context/chunk planningで
使われる。currentはprefill4096固定であり、この案を速度改善や8192がfitしたという結果にしない。

region=max(GDN,QSA,MoE)、固定scratch、KV stage、ring、session-state、allocator余白は変えない。
全bytes_neededがcarveと完全一致したという主張でもない。fused_layoutのsource前提、別geometry、
optional BF16x2/MMQ、KV streaming、borrow/relayoutのcapは引き続きレビュー・検証が必要。
countが減ることで以前より大きいborrow/chunkを認めるため、capの実検査が採用条件になる。

Mainがbenchmark→PR#6回復境界後に適用する場合、fresh processごとのGR_UNFUSED off/on、
4096/8192とtail、BF16x2のsupported modes、primary MMQ0/compact0/1、KV full262144設定で
実countとcarve高水位を照合する。borrow capの直前/不足時は既存の拒否を保ち、正常時には
canary・state/logit・owned cleanup/fresh-faultを検査する。同じchunkのquiet速度と、admissionで
chunkが変わる別armを分ける。源コード比較のみで実VRAM fitを認定しない。

generatorはpinが違えば停止し、sourceをメモリ上で変換して正規unified patchとidentityだけを
このdirectoryへ書く。正規patchのblank context-prefixは保持する。source適用/fixture/build/GPUの
結果はまだない。ownerはMain、consumerはpost-benchmark/PR#6 memory-admission点検、次のreviewは
その点検時。実装のpriorityは過去の速いown-fork実装の安全な回復を先にする。
