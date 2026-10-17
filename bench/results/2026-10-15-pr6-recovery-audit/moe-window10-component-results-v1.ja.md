# MoE window10: 拡大した領域のcomponent検証

独立したMain実装3442ad4f、source contract186051c6を既存verify/moe-windowへcommit/pushした。
mode8を残し、別opt-inのmode10を加えた。最大window81920行、operator/job8192行。
Main sourceにはXeStrata由来の実装を使用していない。

CPU Release/UBSanの双方で、capacity9 positive/34拒否ケース、route18 shapeの
逆引き・expert内token一意性・8192 job上限・legacy拒否が通った。
strict SYCL fixtureも新規compile/linkが通った。

|GPU synthetic case|行数|jobs|実測UR kernel enqueue数|結果|
|---|---:|---:|---:|---|
|smoke|3|30|63|全出力・guard PASS|
|hot81920|81920|100|221|全出力・guard PASS|
|balanced81920|81920|512|1045|全出力・guard PASS|
|partial78452|78452|512|1045|全出力・guard PASS|

全件のstatusは0で、独立CPU oracleに対する全要素照合、全allocation guard、
Mainの起動前/runtime-ready/解放後の3段階env/ELF確認、正常終了、所有process消滅、
新規GPU fault/dumpなしを記録した。UR tracingとparameter validationを有効にし、
DirectSubmissionとcopy offloadを無効にした既存の同じ安全設定を使用した。
このfixtureは合成expert出力のrouting/累積/epilogueであり、GEMMや実モデルの
品質・速度・物理262144/RESTOREを検証したという意味ではない。

当該変更のfull engineを別にbuildし、同一binaryのmode8/10で約78K入力を比較する。
現在の本流/defaultには、反復PP602.0/601.5/603.2を確認したmode8までをPR44で統合済み。
mode10の性能採用はまだ行っていない。

閉鎖した成功ログは必要な証拠をJSONへ保存・照合して削除した。
clean6sample/researchコピーは95files/623345B、componentログは58files/4772760B。
元の測定receiptの内容・status・hashは変更せず、manifestにパス/hash/理由を残した。
