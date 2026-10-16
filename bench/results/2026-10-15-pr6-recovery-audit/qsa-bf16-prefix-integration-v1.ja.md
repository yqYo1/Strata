# 長大prefillのBF16 QSA候補

SGLang同モデルprefillのBF16 operands/FP32 online状態を参考に独立実装。
新しい実験入口 `STRATA_SYCL_PREFILL_QSA_MATH=bf16` はon-the-fly decode、
`bf16-prefix` は使用中の論理prefixのみをGPUで一度BF16へ展開して使う。
既存startup設定 `STRATA_SYCL_PREFILL_NATIVE16=bf16` と
`STRATA_SYCL_PREFILL_NATIVE16_QSA=1` およびlong layer-majorが必要。
既定値existing、短入力のdecode側layoutによるprefillは従来の入口を維持。

prefixは毎callで再生成し、persistent KVやdecode cacheではない。全入力のspans/domain/
alias/extentを提出前に検証し、同一in-order queueでpack→attention。使用中cellと有効な
page mappingだけを読み、欠けたpage・末尾余白はゼロ化。page表をCPUへ戻さない。
追加GPU命令は1 pack kernel/call、attentionは1 kernel/query×KV-head全体。
CPU tensor演算や新しい同期は増やさない。per-request allocationと終了時drainはwrapperに存在。

使用中ceil(prefix/page)*page*2 KV heads*(256*2*2+4) bytesを、expert cache・residualと
合算して確保前にVRAM admissionする。最大262144で538968064B=514MiB。
長大prefillのouter checked leaseがinput/output/scratchを保持し、retirement後にborrow fieldを
復元してから所有memoryを解放する。allocation failure時は明示的な候補error。

手動componentは小規模2case/24 kernelと11case/60 kernelでPASS、新fault無し、所有終了。
全scratch要素を独立にチェックし、source suffix/余白のInf scale poison、逆転mapping、
未割当・範囲外page、短prefix、実体262144-cell source/active prefix、guards、input
immutability、invalid extent/alias/domain、NaN/Inf診断、repeat、0queryを検証した。
source-only初回buildv1とzero-query順序修正後buildv2を区別する。

モデル入口は既存typed QSAと同じresident Mode0 K8V8/text/rope table範囲に限定。
ホストKVのMode1は依然拒否し、未対応として明示する。componentはMode1の実sessionや
full-model256K・RESTOREの証明ではない。新入口のengine build/model/32K以上の公式bench.sh
測定・モデル品質・全context/RESTOREは未実施。default採用や全体PPの向上は主張しない。
元の256K scalar oracle FAILはqsa-bf16-256k-numerical-attribution-v1で説明し、不合格のまま保存。
