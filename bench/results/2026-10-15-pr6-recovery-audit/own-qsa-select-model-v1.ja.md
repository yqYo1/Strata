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

## Arm2 cyclic閉鎖と適用範囲の修正

`own-qsa-select-model-a2-v1.json`も正常閉鎖。768行/7864320値は旧FP32 referenceと全byte一致・finite。
752store/752next-input、48層activation→Full復元、native/server exit0、全owned退去、同boot、
新規fault0、dumpなし。診断1回のPP348.0、wrapper224178ms、selection17097.5ms、
attention69692.5ms。arm1との差から優劣やばらつきを確定しない。
比較済みの重複raw・成功ログ26 files / 32,464,092Bを退役し、全個別結果を保持した。

Mainのcallsite再確認で、このbinaryの実験flagはprefillのほかdecode/verifyの共有APIにも
効くことが判明した。両runは共有dispatch下の診断として保持し、prefill専用の採用結果にしない。
続く独立したhost dispatch修正は`own-prefill-selector-scope-v2.ja.md`参照。
現在は長文layer-major prefill専用APIだけがflagを読み、従来API・reference・短文・decode・verifyは
Mapping0を維持する。この修正後のモデル全体・clean反復は未実施。

R741/R742はFINAL受領済み。Main補足は`main-review-round741-742.ja.md`参照。
XeStrataは性能参考のみで、実装の移植・翻案・そのsourceを下敷きにした再実装は行わない。
