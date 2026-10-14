# Decode format service proposal v1

未適用・未build・未test。製品source、Git、GPU、service、liveログ、cleanupには触れていない。
R570、R579、Main R576–579訂正、retentionとactual root sourceを参照した。
source.diffだけをMainが閉じた測定境界の後にreviewする候補であり、採用/性能結果ではない。

Mainから渡された閉じた先頭requestはE128/W55/R217/A76/D162、CPU jobs68781、CPU rows101036、
全route104160、q=.9700077、logical blob140503244800 B、mean job nt1.469。
CPU GU25.5329 s、Down11.1015 s、quant約.0994 sは並列thread service。
pool call8.3629 s、GPU boundary wait7.7131 s、decode25.544 sと加減算しない。
二request目および全job/fault closureはこの提案の時点では未確定。実行binary44cafccは
Main提供のlabelのみで、この提案のsource pin/binary同一性の証明には用いない。

`source.diff`は既存joined opt-inだけに、最大16 actual format組のgroup(issuer/workers)×6role
calls/cpu_ns/wall_nsを足す。キーは既存same_format全字段で、GU/Down typeだけでなく
activation、row bytes、offset、blob bytes、geometryも一致させる。初回代表layerを記録し、
既存format l行とjoinできる。16組を超えるとformat_invalid、request complete=false。

既存service累積をdispatch直前とnative return後に読む。新規clock、worker writer、atomic、
allocation、wait、job毎log、prompt scanはない。beforeは各active slotをissuer/workersに
直列集約した12 compactcounterのみ。5workersの場合は6slots×6rolesだけ読む。128worker
対応も既存active範囲だけを読む。新しいServiceCounter[129][6]は置かない。
初回は新規zero observerかつ未binding、以後は前native callの既存done/park acquire return後で、
service_budget pointerがdetachされている。afterも最終park戻り後、次dispatch publication前。
zero-job returnはserviceへのworker bindingを見た仕事がなくzero差分としてdispatchを数える。

format serviceのdispatch数を全attemptと一致させ、全pairの各group/role累積を既存serviceの
calls/cpu_ns/wall_nsと厳密照合する。既存invalid/overflow/read整合性もsnapshotで検査。
差分逆転、16組超過、加算overflow、欠落dispatch/pendingはcomplete=false。既存normal/
coverage/phase/task/row/final-wait checksは維持する。exceptionは既存owner guard/fail-stopが
先に処理する。ownerをcopy/moveせず、poolやformat参照の寿命を延ばさない。

追加stateは16×(12×24+int+padding) + 12×24 +小整数/flags、約5.1KiB。
既存ledger70848 B、service49664 BというMain情報に対しledger約76KiBの設計。
既存96KiB stateおよびcombined256KiB static_assertを維持し、実sizeofはMain compile必須。
追加local一時snapshot288 BとServiceCounter1個は既存4KiB formatter envelope内。
close outputは既存96KiB/footer1KiBを維持、追加最大192 compactcounter行+16key行で
既存output boundが判定する。bounded=falseをcompleteへ反映する既存規約を維持。

利用側はrole0/3 GU、role2/5 Down、role1/4 intermediate quantを区別する。
役割内format割合はそのroleの全pair CPU nsを分母にし、issuer/workers合計か個別を明記。
zero分母はN/A。callsはkernel slice呼出数でCPU expert rowsやunique jobsとは異なる。
CPU serviceにGPU kernel、MTP head時間は入らず、CPU critical wallやkernel latencyと呼ばない。
新counterで特定したformat/roleの候補もclean matched whole-decode測定まで性能採用しない。

## Main検証提案

fixture.cppは未実行のCPU fixture。提案適用後、既存host fixture手順でcompile/runする。
zero-work dispatch、同format二layer合算、別format分離、pending/missing dispatch、未帰属service、
aggregate不一致/overflowを確認する。さらに既存observer fixturesへ、16/17pair境界、
clock失敗/逆転、counter overflow、異常終了、出力上限、OFF pathを追加してMainが直列実行。
worker publication自体はこの単一thread fixtureで証明しない。既存pool park acquireとreturn経路の
source review、既存native task closure checksが別途必要。fault injectionは提案/実行しない。

Mainは適用前にpins.jsonの全source SHAを照合し、製品凍結解除後だけ適用する。
その後CPU fixtures/build/static_assertを通し、必要なら別所有付き診断で新format closeと
全service totals一致を確認する。現在進行中のserverは変更/再起動しない。

保持物はsource.diff、pins、README、fixtureとproposal生成scriptのみ。raw trace無し、削除無し。
次reviewはMainによるsource review/適用判断、consumerはdecode format serviceの次候補選定。
