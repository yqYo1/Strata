# 専用selectorのclean長文比較計画

2026-10-16 JST。前goal turnは実source分離・full/反復component・実78453モデル・JIT資源取得と
compact化/commit/pushがあり、実進捗と分類した。600超・1000 PP / 70 TGは未達。goalを縮小しない。

同一qualified native beda08、C8192/FP32/R_GPU32768、native/canonical bank1、Direct1、
QSA device batch1、scratch alias1、head release0、no-token-graphを固定する。
変えるのは長文prefill専用radix4 flagだけ。public decode/verify/reference/短文はMapping0。
official bench.sh prefill-onlyのrequested65536・actual78453/reuse0/thinkingON/24生成。
24生成はprefill完了の確認で、decode性能のベンチではない。

Mainの`run_own_selector_clean_cohort_v1.py`を実行。独立したnative/serverで各arm3回、
順序は(0,0),(1,0),(1,1),(0,1),(0,2),(1,2)。各run前のlibrary file hashとsource/binary/model/recipeを
固定し、normal closure、実入力・配置、request journal、新規faultとowned退去を検証する。
このcohortは各processの/proc mapsを新たに収集せず、以前実mapした7libraryの実file hashを再確認する。
環境はexec前に設定し、process内で変更しない。observer/tracing/validation/IGC/profilerなし。
既に撤去された診断3 env keyも継承しない。coarse/fine profileと境界captureは0。
新規runのnative logにprofile出力がないことも確認する。

比較前に判断条件を固定: median PPが1%以上向上し、candidate最小値がcontrol最大値を超える場合だけ
positive screenとする。これは3標本による工学的判断で、統計的有意性とは呼ばない。
満たさなければdefault0を維持し、追加標本または別の候補/設定を検討する。全個別値と実行順を残す。
生成token一致を合格条件にしない。品質・RESTORE・captured graph・physical262144は別に検証する。

wall1500s/run、RSS88GiB、text64MiB/run、file64MiB、job1GiB。最初の失敗でcohortを止め、
元FAILや未実施枠を合格に書き換えない。Mainのexclusive lockで全GPU/build/test/cleanupを直列化。
Mainが終了を確認してから比較器を実行し、原結果を不変に保って成功trace/重複recordを退役する。

並行してLuna R747/R748が読み取り調査を完了。Mainは8-row分割のlocal byte計算に128Bの誤差を
見つけ、正しくは31088B/groupとした。元報告は保持する。hardware counterの権限制限を理由に
全作業を止めず、別の6.1Solが同一baseからdefaultOFFのsource-only案を作成中。
Mainがcohort終了後に全diff・静的契約・小部品を審査し、positiveなら実モデルへ進める。
XeStrata sourceは使わない。上流の次の実確認予定は2026-10-16T01:12Z頃で、今回はまだ再確認していない。
