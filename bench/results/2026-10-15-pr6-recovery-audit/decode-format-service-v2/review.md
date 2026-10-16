# Decode format service v2 — 未適用修正案

Main reviewのOOB指摘は正しい。v1はnegative receiptでworkers129や壊れた代表layerを安全に
扱えなかった。v1原本を変更せず、v2 source.diffを現在の製品sourceから再生成した。
製品3fileのみの差分で、追加artifact fixture.cppはMain実行用の未適用・未compile提案。
build/test/GPU/Git/service/cleanupなし。active log、環境dump、禁止recordを読んでいない。

## 修正

1. beginの最初のshort-circuitは!envelope()。layer/key検査前にlayers/workers/k/experts範囲を
   拒否する。l>=kLayersも明示し、actual format妥当性と一致を確認する。
2. format_representatives_valid()は全loop前にenvelopeとcount0..16を検査する。代表layerは
   0<=l<layersかつl<kLayers、そのfmtはvalid_format必須。begin/end/completeで共用する。
   completeはこの検査を最初に行うためworkers129によるslot OOBがない。
3. receiptは不正count/envelope/representativeをout.bounded=falseとする。count負数では
   loopせず、17以上も16に上限。代表layer不正時はfmt参照前にskip、invalid formatもskip。
   既存service receipt loopのslot<=128、既存layer loopのl<64も保持する。
4. begin headerへadditive marker service_format=1を追加し、schema=mtp4-route-service-v2、
   W/R/D/A/C/E、role、既存footer契約を維持。
5. 既存tests/cpu/decode_joined_mtp4.cppのwindow()にbegin→既存CPU dispatch(あれば)→endを
   組込み、全CPU/zeroCPU成功fixtureのcompleteが新照合も通るようにした。既存negativecase、
   clock、off、task、phase、final wait、owner fail-stopを一切削除/弱体化していない。
6. fixture.cppは同型再利用、新型、zero-work、未帰属変更、欠落dispatch、pending、
   aggregate mismatch/overflow、加算overflow、workers0/129、layers0/65、代表-1/64、
   不正format、count-1/17、16/17形式上限とnegative receipt安全出力を提案する。

begin/endのcounter計算はv1同様。新clock/wait/allocation/worker writer/atomicなし。
既存both flags=1のjoined scope内だけが呼び、default OFFを維持。service_counterは
GPU/MTP時間を含まず、並列thread CPU時間でありcritical-wall decompositionではない。
16 format組×2group×6role×3uint64のcounterと小snapshotだけで、既存static_assert、
96KiB出力/footer1KiB、combined256KiB条件を保持。実sizeof/buildはMainが確認する。

Mainの初回2×34K/128+short gate正常終了はMain提供の結果であり、この提案で再測定していない。
A/decode-ledger-final.ja.md、summary-v1.jsonの結果を書換えず、format16組の新counter自体は
未GPUrun。v1の初回sampleと同様、service総量をdecode wallから引かない。

## Mainでの次gate

pins.jsonと実sourceを照合→source.diff review/適用→既存CPU fixtureと追加fixtureを直列実行。
可能なら追加fixtureをASan/UBSanでも実行しOOB negative receiptを確認する。ここでは未実行。
既存owner fail-stop fixtureは従来通りCPU-only契約テストで、新GPU fault injectionはない。
その後必要な診断だけでservice_format_close complete=1、全service group/role総和一致、
既存request complete/fault closureを確認する。性能採用は別のclean paired decode条件が必要。

保持consumerはMainのformat別CPU service選定。raw captureや削除は無し。
次reviewはMainのsource review/fixture/build境界。v1はe0eb1622保存というMain提供label、
v2 pinsは今回実際に読んだsource bytesのSHA256で、Git identityは新規取得していない。
