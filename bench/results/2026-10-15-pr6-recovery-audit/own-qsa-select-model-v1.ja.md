# 自作selectorの実モデル検証

2026-10-16 JST。全部品比較とsourceは`29349cd1`でfeature/syclへcommit/push済み。
Main所有のofficial bench.sh prefill-onlyで、actual78453/reuse0/thinkingON/24生成を測った。
C8192、FP32 residual GPU32768、native/canonical bank1、Direct1、prompt mapping0、
coarse/fine observer1、no-token-graph。生成24はprefill終端確認で、decode性能や品質の評価ではない。

## Arm1 radix4閉鎖

`own-qsa-select-model-a1-v1.json`が正常閉鎖。48層全activation→Full復元、
native/server exit0、全owned退去、同boot、新規fault0、dumpなし。
実nativeは`0e62c5fa74cb6a077aef5c5977617e6d74fbd8c2c712ac8f80f3445946289a36`。
wrapperの実native環境もradix4=1/cyclic=0/prompt mapping0を確認した。

Main comparerで旧FP32 referenceの768行/7864320値が全byte一致・finite。
同runの752store/752next-inputも正常。生成tokenの一致は合格条件にしていない。
SessionGraphs recapture0はexplicit no-token-graph条件の観測で、捕捉graphの復元を実証していない。

| 観測 | 旧fine reference | 新arm1 |
| --- | ---: | ---: |
| PP、tok/s、診断各1回 | 309.9 | 346.6 |
| selection queue、ms | 24399.6 | 17945.8 |
| attention queue、ms | 69909.3 | 69692.0 |
| wrapper、ms | 251647 | 225088 |

selectionはscore producerとtopkを含む非排他的なqueue区間。
input/store/MoE等の時間も変動している。source/binaryと実験の両方に差があり、
このPP差をradix4の因果効果やばらつき幅と断定しない。
採用には同じ新binaryのbaselineとcandidateのclean独立反復が別途必要。
目標prefill反復600超→1000、decode70は未達。

比較完了後、重複したarm1 raw31,457,280Bと成功ログ等を26 files / 32,457,136B退役した。
`closed-own-qsa-select-model-a1-retention-v1.json`はpath/hash/reasonと元receipt不変を記録する。
byte一致する旧fine raw1個を、次のarm2/current-default比較向けに32MiB予算で保持する。
次のrunはpaired64MiB予算に収まり、成功した重複rawを累積保存しない。

## 実際にロードされたライブラリ

Mainがliveな所有nativeのPID/start/ppid/sidを検証して/proc/mapsを読み、
`own-qsa-select-model-a1-loaded-libraries-v1.json`へ保存した。GPU queryや環境変更はしていない。
IGCの実pathは`/usr/local/lib/libigc.so.2.41.5+1788943183`と同版名のlibigdfclだった。
R739のdpkg一覧にあった`/usr/lib`の2.40.13を、実使用版とみなすことはできない。
SYCL/URとZE driverの実pathも記録した。今後のdump/資源解析は実path/hashに合わせる。
filenameの観測は独自ビルドのsource commitやdump flag動作の証明ではない。

## 次

同じnativeでarm2 cyclicを起動済み。終了まではsource/build/test/cleanupを重ねない。
完了後に同じ768行・lifecycle・faultを比較し、selection/attention区間も記録する。
R741はFP16 residualの次の品質gate、R742は実IGC resolverと既存AOT/cache経路を読み取り専用で調査中。
XeStrataの実装は使わず、研究・実装・実行検証の担当を分ける。
