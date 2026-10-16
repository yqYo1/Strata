# 入力を揃えた次の測定とwave候補の準備

2026-10-15。前のgoal turnはsource/provenanceの実装とGitへの保存、所有jobのverified
waitを行った。今回も同じjob16956を観測した。full OFFはagent3 sessionsと約32Kの
3 requestsを通過し、thinking有効のlong generationへ進んだ。約32Kの表示値は
200.5/199.8/199.5 tok/s、実入力はいずれも32089、reuse0。まだ終端のowner/fault/suite
検証は済んでいないので、正式完了や最終速度の採用とはしない。短い出力のこのcohortを
長文生成品質の合格にも使わない。

その後、long-generation run1 regular-codeが正常なrequestとして戻った。表示は
3346 actual completion tokens、decode9.8 tok/s、drafts accepted/offered=2382/2895、
eligible=true。これは実入力32768以上・thinking込み1024以上・本文512文字以上などの
単一sample gateであり、全3回のstrong/weak分類やtask能力、suite/fault完了ではない。
次のvaried-proseへ進んでいる。

## Mainが作った実行手順

進行中のroot client/nativeと依存物には触れていない。次のscriptはsourceを準備した段階で、
まだ実行していない。

1. `recover_bench_nonce_v1.py`を、full0が閉じた後、旧clientのまま実行する。
   元calibration2件のbody hashから唯一のtimestamp tagを検証し、別のprovenance receiptを
   保存する。元結果を書き換えない。回復できなければold OFFをmatched pairへ転用しない。
2. `qualify_bench_input_tools_v1.py`は同じnonblocking measurement lockを取得し、
   full0の所有process/session退役を確認してからhost testsを実行する。旧clientと提案client
   をASTでも比較し、Suiteの全既存methodはconstructor以外が同一、その他module pathsは
   CLI/main/compare metadataと新validator以外が同一であることを確認する。
   fixed-workload/request-identityの実host test各6件以上、入力/source hashesとOwnerの
   正常終了を保存する。実行していないテストのPASS receiptは置かない。
3. Mainが閉じた境界で`bench-fixed-workload-v1/source.diff`を適用する。
   `tools/strata_bench.py`だけの変更で、native binaryは同じb1cc9cb4…のままにする。
   実際のclient hashがhost-qualified proposalと一致することを確認し、commit/pushする。
4. `run_matched_iq_on_v2.py`は引数なしで、この実証済み旧full0を参照してONを測る。
   OFF/build/host/provenanceの実receiptと退役、同boot・driver・model・native source/binary・
   linked libraries・runner/wrapperを確認する。engine argsは同一、環境差はIQ flagと
   artifact/cache pathsだけ。新clientの差はレビュー済みfixed ID/provenanceだけとし、
   回復した元tagを`--workload-id`へ渡す。3回の32K、全agent、両long generationを測る。
   native/server正常終了後、旧/新両journalの全startを照合し、不一致ならFAILのまま保存。
   native終了、元sample、入力差分を消したりtoleranceを緩めたりしない。

旧clientと新clientはsource hashとmetadataが違うため、汎用compare表示ではdescriptiveに
なる。このMainの別assessmentは、その差分の限定と全wire input hashを検証して初めて
比較の可否を判断する。上のscriptを用意しただけで入力一致や性能改善を主張しない。
current full0が失敗/不完全ならこのON転用scriptは拒否する。その場合は元結果を保持し、
新clientでfreshなOFF/ONを作るか、別の次の有用な実験を選ぶ。goalをblockedにしない。

## 新しいwave census候補

Solの`layermajor-wave-census-v2`をMainがsource reviewした。既存の任意census ownerだけを
拡張し、2/4 chunk窓のexpert union、per-expert ne合計の16-bin分布、GU/Down issuerの
expert ID一致を保存する。全raw route IDsは不要で、通常off pathのqueue/math/branchは
変えない。partial窓、nonzero position、同ne別ID、重複/欠落issuer、32-bit加算、counter
overflow、output capと/dev/fullを独立set/map参照のCPU fixtureで検証するsourceがある。
validatorは元のv1実モデル検証を先に呼び、wave fieldsを追加検証する。

提案header/diff/fixtures/validatorは未build・未test・未適用・未GPU実行である。
4 MiB owner/1 MiB outputの制限は維持し、実sizeofや実モデルunionはまだ測っていない。
これを使う次のdiagnostic binaryも、比較中のnative binaryへ上書きしない。

## R637の訂正とメモリ寿命

MainがR637 v1のcap32771を誤りと検出し、FINAL後に訂正を依頼した。実capは2051で、
R637 v2は三branchを再計算した。Mainもsource constants、count allocator、三branchと
compact/MMQ/fused条件を読み、v2提出後に冒頭とtable前文の二つの転記残りを修正した。

| Chunk | GDN counted B | QSA counted B | Compact MoE counted B | Region max B |
| ---: | ---: | ---: | ---: | ---: |
| 4096 | 590350336 | 607128320 | 543837696 | 607128320 |
| 8192 | 1180698624 | 1120979712 | 1087671808 | 1180698624 |

これはsource-derived allocation容量で、実際のfree VRAMではない。QSA/GDN/MoEは
同じregionをaliasする。waveの先行chunkデータは次chunkのattention scratchと同時に
liveなので、全regionをfree扱いできない。expert段階にもDmとmicrobatch scratchが
同時に要る。既存chunk分を二重計上せず、同時生存のpeakとchecked fenceから再配置を
設計する。256KのFP32 residual10 GiBは既存のGPU-prefix/host-tail ownerであり、
余剰VRAMとして足さない。具体的wave fit・速度・品質は未検証。

## R638: MTPを単に後回しにはしない

E-9は既にchunk内batchで、window外の不要cellも`first_needed()`で省く。MTP promptは
K/V-onlyで、expert decode graphとは別である。target prefill timerにはcallback処理が
入り、`R_rows`は再利用されるchunk bufferなので、後回しには別の所有bufferまたは再計算が
必要になる。遅延すればそれだけで速いという根拠は得ていない。MTPをoffにした数値を
製品の改善に扱わず、現在のMTP有効の比較を続ける。

すべての新host/build/GPU/cleanupは進行中のfull0と直列化する。調査の完成と1000/70の
達成は別で、元の目標・物理256K・RESTORE・品質の未完了範囲は維持している。
