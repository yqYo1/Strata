# R936 調査レポートレビュー

- R934/R935のSHA256は指定値と一致し、語数は両方とも依頼された範囲内でした。
- R934は知識収集として受理し、Main本文未読のまま保持します。このdigestには技術要約を含めません。
- R935は判断支援として受理しました。閉じた9 receiptは、BF16 QSAのcomponent timingを裏づけますが、モデル全体の速度・品質・採用は示しません。
- R935の「SGLang's reached source path」は「pinned SGLang source path」に直すべきです。ローカル実行時の分岐・dtype到達は未確認です。
- Mainが実行中の同じBF16演算を使うlate-flags controlのtimingは未完了です。結果は推測せず、receiptが閉じてから候補の優先度を見直してください。
- production dispatch、model PP、品質、RESTORE、全体の物理262144-contextは未認定です。FP32/llama.cppとの厳密一致を採用条件にはしません。
- 新しいOrca scopeは現時点で推奨しません。R934が予定した知識収集を満たし、次の判断にはMain所有controlの結果が要ります。
- 変更したのはこのJSONとdigestのみです。research-index.jsonは編集していません。
