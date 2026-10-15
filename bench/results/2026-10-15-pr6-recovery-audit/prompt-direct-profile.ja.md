# int8 prompt Directの実モデル比較

既存Direct設定を34,603入力・128生成・通常MTP4の同一仕事で比較した。
両方とも同じ`0c281323...`バイナリ・モデル・入力・固定4096チャンク・FP32残差GPU4096行。
環境の差はDirect設定とprivate cache出力先だけである。
どちらも公式unitraceのdevice timingを使った診断で、clean速度比較ではない。

| 項目 | 従来のIndirect | Direct=1 |
| --- | ---: | ---: |
| 実際のint8 attention | launch<1,false,false> | launch<1,true,false> |
| attention呼出回数 | 108 | 108 |
| attentionのdevice duration合計 | 44.066063496秒 | 29.805022972秒 |
| GU dequant duration合計 | 22.413196秒 | 22.40071秒 |
| 汎用FP16 dequant duration合計 | 12.073215秒 | 12.06434秒 |

attentionの合計時間は約32.36%減った。両設定とも正常終了・回収、新規GPU障害0。
128生成ID、W55/R217/D162/A76/C131/E128、CPU68781jobs/101036rowsも一致した。
prefill/decodeを共用する他のカーネルをphase専用時間とは扱わない。
表示されたprofiler付きprompt時間259.773→225.057秒は、正式なprefill改善値には使わない。
各1processであり、反復幅もまだ評価していない。

`analyze_prompt_direct_profiles_v1.py`で環境差、source/binary、生成IDと仕事量、
実際のkernel名と108回を照合し、比較値と元receipt hashを`prompt-direct-profile-pair-v1.json`へ保存した。
全表は元の二つのprofile receiptに保持し、比較ファイルには複製しない。

次は同じバイナリのDirect OFF/ONを`bench.sh`で反復し、32Kとagent sessionを確認する。
採用前に、thinking有効のMTP強/弱・長時間生成も既存の拡張benchで検証する。
今回の設定はprefill出力だけに適用し、decodeの別実装を採用する理由にはしない。
既定設定、モデル品質、物理256K/RESTOREの認定は変更していない。
