# Direct0公式full bench: 3回目32Kの停止

2026-10-15。**FAILED / incomplete。Direct1は未実行、速度比較・採用なし。**
元receipt `prompt-direct-bench-full-0-v1.json`は失敗のまま保持する。
同一binary92acf71e...を固定していた測定で、新しいhead-split、host時計、checkpoint、
dequant候補は適用していない。Mainは元の製品source/benchmark sourceのhash一致も再確認した。

## 実際の順序

1. native PID3276405/start37651045、server PID3276378、job leader3276375で開始した。
   chat9、4K3、16K3、agent3×6が完了した。agentはcache0のcold再読条件である。
2. nominal32Kの実入力は32089 tokens。最初の2回はPP **197.7 / 198.0 t/s**、
   prompt **162284.0 / 162098.5 ms**、24生成のTG **8.8 / 6.3 t/s**で終了した。
   比較armがなく、3回目が失敗しているので、これを正式な改善値や中央値にしない。
3. 3回目はwatchdog480秒で「layer27 / prompt chunk0」の進捗停止を報告した。
   poolは18/18 jobs完了、5/5 workers parked。verify window1293の表示は過去のcompleted状態で、
   active verify spin waitはなかった。失敗requestのwallは571.934秒、PP/TG完了値はない。
4. **02:18:38 UTCに元nativeがSIGABRT(-6)、wrapper134で終了した。Mainの停止操作より前である。**
   同じserverが新native PID3478501/start37948251を自動起動し、33Kのprefillと940生成を進めた。
   この後続requestを失敗した3回目の成功や元suite完了に数えない。
5. Mainは既存job leaderのboot/PID/start/parent/SIDを照合し、pidfd経由SIGINTで測定を止めた。
   serverはshutdownへ進み、残った新nativeは既存Ownerがpin済みpidfdのSIGTERMで回収した。
   Ownerのsurviving-member error、cleanup、元native abortを全て失敗結果として残す。
   最終的に全session空、direct child回収済み、新kernel fault0、driver dumpなし、boot不変だった。

停止位置はprogress labelであり、実際のprogram counterではない。現時点でGPU hang、
host deadlock、JIT、model data破損のどれも確定していない。persistentなGPU停止を示す証拠はなく、
同じbinaryの新nativeが進んだことも重要な制約である。device reset/unbind等は行っていない。

## 保存した証拠と再現入力

`capture_prompt_direct_failure_v1.py`で、閉じた元receipt・source pins・journal hashを確認した。
全個別sampleと失敗を含む82行のcompact journalをbyte-identicalでGitに保存した。
hash chainはvalidだが、開始だけのlong-generation request81が残るためcomplete=false。
watchdogの代表excerpts、停止前に保存した元native exit identity、最終Owner/fault結果を残した。

full API traceを残さず再現入力を取り出せた。元journalのrequest SHA256と公開bench generator、
calibration260/906からnonce1792027796を復元し、3つ全てのHTTP payload hashを一致させた。
同じserverのtemplate/tokenizer処理で再tokenizeし、3件とも32089 IDsを確認した。
tokensはGit外の`main-primary-prompt-direct-stall-inputs-root-v1`に置き、path/hashを
`prompt-direct-bench-failure-evidence-v1.json`に記録した。現在のconsumerは次の停止箇所診断である。

## 次の診断と比較の扱い

元Direct0/1比較はinvalidとして閉じる。Direct1の既存gateは失敗Direct0を拒否するので起動しない。
元のsource/binaryは引き続き保存する。以後は別owner/root/receiptの**独立診断**として進め、
診断を正式pairの成功や比較gateの代替に使わない。最適化のGPU適用は原因確認後に再計画する。

まずGPUを使わない子processで、制限されたptrace許可、所有するGDBからのstack取得、
detach後の再開・回収を確認する。その後、同じnativeを直接所有し、exec時から固定した
validation/warning/progress出力付きで、停止中の全thread stackを取得できる診断を計画する。
machine-wide ptrace設定、root、未知processへのattach、GPU復帰操作、自動再試行は使わない。

## R613/R614 Main補正

R613のsource上の候補はqueue/library境界、host grouping/wait、根拠があればJITの順で妥当。
ただし原文の「native abortがowner stop後」という順序は逆で、上記identity/timeが正しい。

R614の「別の古いattempt」「2つのlong-generation成功」という解釈も採用しない。
同じ82行journalのrequest75/77が2つの32K成功、79が3回目失敗、81が自動再起動後のlong生成である。
incomplete journalとMain割込は今回の停止と整合し、別attemptの証拠ではない。
その一方、benchのfail-fast option、wrapper lifecycleの上書き防止、失敗後に次requestを
発行しない測定用制御というR614提案は採用候補。通常serverの復帰機能は保持する。
