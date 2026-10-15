# Main review: repeated prefill and quantized-weight candidates

2026-10-15。R615/616とR617/618の二組のread-only Luna調査を受領し、Mainが実sourceと照合した。
研究reportの原文は保持する。製品の推論sourceはまだ変更していない。

R615の結論通り、第3回32Kだけが停止する確定バグはsourceから見つからなかった。
通常のlayer-major終了では一時所有bufferの解放前にqueueをdrainしている。
ただし進捗markerはlayer入口で、pool完了はmain threadの停止位置を表さない。
Mainの独立診断は既存92acf71eバイナリ、同じ32K入力3件、exec固定のvalidation/traceと
所有GDBを使う。先行chat/agentを省略しているので、不再現でも元suiteの解決とはしない。

R616のpacked-XMX構想は参考になるが、冒頭のactive formatは誤り。
実際のGUは18 IQ3_XXS、22 IQ2_S、21 IQ3_S、23 IQ4_XS、Downは20 IQ4_NLと42 custom Q2_0。
17 IQ2_XSだけを実装する案にはしない。小さいneの歴史的call比率を現行row比率や速度に
置き換えない。oneMKLとの和の順序が変わるXMX案は、FP16 operandが同じでも別の数値検証が必要。

R617に加え、Mainが6形式のcandidate helper、元scalar codebook/dequant、GU交互row、
generic F32の非変更、span検査後だけのdispatch、CPU corpusとGPU fixtureを確認した。
新しい各elementは元のscale/grid/sign演算順とf16_from_f32を保つ。CPUで全scale bitpattern、
grid/sign/stride/guardを元scalarと照合する段階へ進める。これはまだdevice合格ではない。
F32 NaN payloadの比較除外と、scale/codeの全直積や専用halfway corpusでない点は記録する。
GPU public入口は元と同じ2/4-byte alignment制約を保持し、CPUのoffset1/3成功をpublic契約に広げない。

R618に加え、Mainがv2のcaller diff、実際のcandidate headerとC++ fixtureを読んだ。
reread_to>0のrequest全体でcaptureを迂回し、旧checksを保持する。借用callbackは代入前に
RAII guardを作り、ownerより先に解除する。C++ fixtureは元のPython modelだけでなく、
実candidate headerをhost tag stateと元LRU/helperを使って呼ぶ。
元Direct0/1比較は失敗として閉鎖したため、両arm成功を待つという旧proposal文面は適用しない。
現在のGPU診断が閉じてから、共有lock下で両候補のnormal/ASan+UBSan CPU fixtureを実行する。
実tensorのRESTORE、MTP KV、物理256K、速度は別途未検証のままとする。

Mainは後続packed-iq-xmx-probe-v1も読み、実行を却下した。Guard destructorが無条件に
sycl::freeするため、submit/wait_and_throw例外時に未完了queueのUSMを解放し得る。
stack上のqueueもunwindする。既存のhistorical停止原因と結び付ける証拠はないが、
この新しい試作はchecked retirement/fail-stopで所有bufferを保持する修正まで実行しない。
加えて、bound用max operandのmetadataはroundtrip精度で出すこと、最初は既存nativeと同じ
compiler/oneMKL/runtimeで比較することをv2へ要求した。v1はsource proposalとしてのみ残す。

CPUの次の実行recipeはcheck_candidate_cpu_contracts_v1.py。
推論source適用やnative rebuildを含まない。実行結果は別receiptで記録する。
