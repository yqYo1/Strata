# Main review: immutable head-split proposal v2

2026-10-15。FINAL後、kernel/diffのv1同一性、probe、capture schema、comparator、
controllerをレビューした。未適用・未build・未test・未GPU実行。

native probeからprocess環境のwriterは除去された。armはexec時の環境を読取検証して固定する。
28決定的cases、全24head出力とguards、12入力のfingerprint、独立FP64 reference、
別process間CH32 bitwise / CH64 toleranceを維持する設計を確認した。
入力/参照の全照合、admission失敗を後続計測へ流さない設計は保持する。
kernelのCH32演算順とgroup一様性は別Luna R612が独立にsource audit中である。

**同梱serial_controller.pyのsuperviseは実行しない。** v2は`killpg`を使い、
`p.wait()`でleaderをreapした後に`group_absent(p.pid)`を数値PGIDへ問い合わせる。
これではsessionを固定する未回収leaderとpidfdを使うMain既存ownerの契約を満たさない。
終了観測、停止対象、元status保存を一つにそろえるため、Mainの適用時は既存
`bench/results/2026-10-14-niko-main-service-port/direct_owner.py`を使用する。
そのSHA256はa29d8257eecc77c4ad9ae6671d092b56f4c09a35c4247c125fed3a8c8bc1f95f。
adapterは新規pathに作成し、元controllerは未実行原本のまま保持する。

freeze-releaseはDirect0/1両方の同一92acf71e binaryによる正式全suiteの完了と全owner退役を
要求する。example JSONやDirect0のみの終了で解放しない。Main build receiptのsource/fixture/
binary pinsを確認し、既存ownerによる有限ログ・deadlineとfresh fault検査を付けて初回起動する。
独立controller/comparator契約、実kernel route、動的SLMやspillも未検証である。

cap262144/ID262143の小pool fixtureをfull physical256K/RESTOREの合格にはしない。
実モデル32K以上・64K・実context境界、thinking/agent/長生成での品質と速度は別の必要検証である。
