# 実モデルのカーネル時間と次の優先順位

Arc B570 / Ryzen 5600X、現在の`0c281323...`バイナリ。
34,603入力、prefill chunk4096、FP32残差GPU4096行、int8 KV、context設定262144、
専門家cache128、通常MTP4、128生成。公式unitraceの固定ビルドで取得した。
起動・prefill・decodeを含む一つのprocessのaggregateである。
全136項目の元表、commands・source/binary/input/environment、CPU service、
生成ID、終了・回収・fault結果を`model-device-profile-v1.json`へ保存した。
元の128生成ID、W55/R217/D162/A76/C131/E128、CPU68781jobs/101036rowsが
以前の比較と一致し、正常終了・回収、新規GPU障害0だった。

| カーネル/処理 | 回数 | 記録されたdevice duration合計 |
| --- | ---: | ---: |
| prefillのint8選択attention、CH32/Indirect | 108 | 44.066秒 |
| GU重みのFP16展開 | 91,880 | 22.413秒 |
| 汎用FP16展開（複数の用途を含む） | 94,580 | 12.073秒 |
| QSA block score（prefill/decode共用） | 2,292 | 2.802秒 |
| GDN recurrence（prefill/decode共用） | 4,284 | 3.680秒 |
| decodeの順序付きKV検索・CLOCK | 660 | 1.295秒 |
| decodeのKVコピー | 660 | 0.318秒 |

全device operation duration合計167.176秒には、M2D29.275秒、D2M12.633秒等も含む。
これらのコピーは起動・残差・照合等を分離できず、実際のPCIe wire byte数もない。
GPU busy時間の和集合、decode/prefillの排他的wall、転送帯域へ変換しない。
profiler付きのprompt259.773秒/decode40.013秒をclean性能として採用しない。

attentionの108回は12 QSA層×9チャンクと一致し、実際に
`prompt_attention.cpp`の`launch<1,false,false>`を通っている。
KVの各660回はW55×12 QSA層と一致する。
`analyze_model_device_profile_v1.py`がこれらと表の合計、CPU service総和を検査した。
共用の汎用dequant12.073秒をDown GEMMの時間と呼ばない。
R600のその表記は誤りであり、元レポートはそのまま保存する。

CPU thread service合計は35.615秒で、複数workerを足した値である。
既存計測と同様にGUが大きいが、この値を単一wall時間やGPU時間と足し合わせない。
nt1のjobs46071、nt2=15757、nt3=4361、nt4=2592。
形式とntの同時service時間、実際のexpert再訪率は別の観測が必要。

## 次の判断

1. prefillは、実GPU時間44秒の選択attentionを優先する。
   既存の`STRATA_SYCL_PROMPT_DIRECT=1`は同じhi/lo計算の出力方法だけを変えるため、
   int8で最初に比較する。以前のFP16合成結果をint8の合格と見なさない。
   次候補はCH64/head分割。QSA block-score GEMM案とGDN案は優先順位を下げる。
2. 重み展開は、同じlayerの複数chunkで再利用できるか検討する。
   512専門家すべてのFP16重みは約4.69GiBあり、他の所有メモリーと同時に
   無条件でVRAMへ置く案は使わない。展開の一部cacheや処理順変更は正しさとpeakを別途検証する。
3. decodeは、部品検証と追加3processで再現したlossless GU predecodeを進める。
   まずissuer上のbounded expert再訪observerでcold-fill回収と容量を判断する。
   8GiBの仮想LRU結果を実RAM admissionと混同しない。
4. KV検索は改善余地が実測されたが、コピー起動数を減らすだけの案はint8で効かなかった。
   検索並列化は上の候補の後に置く。

現在のattentionはQをhi/lo FP16、整数KV codeを正確なFP16、確率もhi/loで
行列計算している。R600の「Kもhi/lo」という記述は実装と異なる。
single-halfへ精度を落とす変更はDirect比較には含めない。
provider表のstatic SLM4096Bから、dynamic local_accessorを含むtotalやoccupancyを推定しない。

物理256K入力、RESTORE、品質、正式bench.shによる改善採用は引き続き未認定。
プロファイル原文は集計と照合を保存した後、保持規則に沿って整理する。
