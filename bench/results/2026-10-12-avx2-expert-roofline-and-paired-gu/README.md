<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# Ryzen 5600XのCPUエキスパート測定とIQ3_Sの実験

上流変更を取り込んだ0691bcdfを基準に、CPU処理の個別値を保存する計測を追加しました。
prefillのCPUジョブ計測とgate/upのロード共有実装は既定で無効です。
新しいGPU計測コードは適用・実行していないソース案として保存しています。

256MiBの有効な合成native量子化データを使用し、CPU0に固定した1スレッドで測定しました。
各組み合わせ・NT・手法に5回の個別値を保存しています。
gate/upとdownの入力は事前量子化した別のデータです。実際のpool・活性値変換・転送・モデル速度を含みません。

| native形式 | NT=1 gate/up ms | NT=1 down ms | NT=8 gate/up ms | NT=8 down ms |
| --- | ---: | ---: | ---: | ---: |
| iq3_xxs/iq4_nl | 0.268 | 0.084 | 0.692 | 0.435 |
| iq3_xxs/q2_0 | 0.285 | 0.038 | 0.646 | 0.151 |
| iq3_s/iq4_nl | 0.319 | 0.097 | 0.841 | 0.431 |
| iq3_s/q2_0 | 0.302 | 0.036 | 0.794 | 0.146 |
| iq2_s/iq4_nl | 0.247 | 0.086 | 0.855 | 0.450 |
| iq2_s/q2_0 | 0.244 | 0.036 | 0.828 | 0.164 |
| iq4_xs/iq4_nl | 0.142 | 0.088 | 0.628 | 0.436 |

IQ3_Sのgatherとロード共有は、それぞれ別プロセスのOFF/ON/ON/OFF順で比較しました。
各プロセスの中央値を平均すると、gatherのNT=1 gate/upは約2割遅く、decode用途では採用しません。
ロード共有のNT=1 gate/upは約3.8%短縮しましたが、変更しないdownやNT=8にも数%の変動があり、
計算カーネルだけの小さな差です。既定はOFFのまま、モデル全体のdecode改善は未確認です。

実際のCPU dispatch/pool計算と全float出力一致、scalar/gatherの全640行のビット一致、
free・contrib-llvmのCPU試験、AddressSanitizer/UndefinedBehaviorSanitizerが通りました。
サニタイザーの検査項目はすべて有効です。メタデータを正確に削除するリンカー設定で、
CPU試験にGPUランタイムをリンクせずに検証しています。
[リンカーの仕様](https://lld.llvm.org/ELF/start-stop-gc.html)

エンジン全体はfree・contrib-llvmでビルド済みです。新コードでのGPU推論は未実行です。
最新のGPU実測は32Kでprefill平均371.8 t/s・decode平均7.86 t/sです。
1000/70 t/s、256K全量、RESTORE、モデル品質の検証は未完了です。

次はdevice-USMを使うgrouped GEMMの直接イベント時間と実行経路を測定し、
prefillの計算性能と大きなチャンクに必要なメモリ量を確かめます。
元の失敗状態、個別値、ソース・コマンド・環境はJSONに保存しています。
