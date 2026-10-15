# Canonical bank / layer-major profileの初期qualification

2026-10-16 JST。XeStrataは性能参考値のみ。今回のsourceはclean own-fork base9177ad4fと
独立したcanonical/profile提案、Mainのfixture・pack64修正から構成する。XeStrataのsourceを
参照・移植・改名・書き直しに使っていない。既存の派生診断撤去とproject-only skill方針は継続する。

## 実装と出自

default OFFのSTRATA_PREFILL_CANONICAL_DENSE_BANKを追加。native bankと同じ排他leaseで、
decode配置のcanonical arenaをlong prefill中だけ退役し、当該層をprivate bankへ入れる。
HC96 norms、lookahead、PLE、nonlayerはpersistent poolへ残す。元の変換済みRAM bytesを
startup upload時に保存し、終了時はD2HせずFull配置へ復元する。両destination予約を先行し、
PLE rebinding、SessionGraphs/Verifier（norm-bearing commit含む）retirement/restore順を維持する。
短入力とflag OFFは従来のFull配置を使う。context262144の制約は維持する。

default OFFのSTRATA_PREFILL_LAYER_MAJOR_PROFILEも追加。旧TIMING guardを変更せず、
GDN/QSA/MoE・residual input/store・expert uploadのcoarse queue interval、CPU source/waits、
outer NativeDensePhase entry/run/restoreを観測する。12eventを既存drain後に再利用し、
perexpert timerや新しいpending-event wait/queue drainは足さない。short warmupは通常経路。
queue intervalにはhost submission gapを含み、pure kernel/排他的critical path/PCIe wireとは呼ばない。

combined initial diff25ffa24bはsource integration receiptに、適用後の全source hashを保存した。
fresh actual129step SYCL buildはPASS。初期binary25da43abの7productは後述修正で置換したため、
old/new hashと理由を再build receiptへ記録した。旧clean reference8fdcは別buildで保持する。

## 見つけた不具合と修正

初期native-only componentは全300行列でPASS。canonical初回はcaptureのadmissionでFAIL、
fixture exit -6。実pack headerはalign64だが、新案がoriginal arena alignmentを256固定としていた。
新しいdriver fault/dumpはなく、全processは退去した。元FAILを変更していない。

canonical-align64-fix-v1/source.diff（7ede43999112a8c005bbb6c64a2e26421e548a2097cec50e0f4755f576dd0d57）で、
元pack alignment64/256とinternal bank stride256を区別する。loaderは実alignmentを保存し、
adoptは対応する値を照合する。owner size/RAM budget/unsupported alignmentのエラーも分けた。
CPU policyに64/256と非対応値、componentに不正capture alignment32拒否を追加した。
再build14stepと新policy実行はPASS。現binary:
1694705b193d8afbcbd975632cf46489a6863e2fd91cf1197230033dcfb1f3df。

## 閉鎖した検証

- coarse-profile-host-v1.json: actual observer headerをhost API shimで13arm検証。最大12owner、
  48層×32chunkで12256mark、全owner一回破棄、API失敗/NaN/負時間/二重begin/early exitで
  incomplete出力。各50行/11921〜14173B。これは実runtimeや256Kモデルの合格ではない。
- canonical-bank-component-*-v2.json: 14arm全PASS。actual NativeDense全300 GGUF payload/metadata、
  canonical777tensorの初期・48active overlay・Full復元、exact PLE name-to-pointer、HC lookahead、
  Prepared cancel、canonical第一/第二OOM、native reserve OOM、restore OOMを確認。
  canonical bytes1538035200、bank32284672、persistent82821120。payload rereadはloaderと
  descriptorを共有し、独立GGUF metadata oracleとは呼ばない。
- 8uncertain allocator armは各別processで、productionの正しいoperation messageを一度出し、
  exit -6。信号名だけでは合格にしない。全14processはclosed/sameboot、freshfault0/dumpなし。

## 未完了と次のgate

actual Prefill/Verifier/SessionGraphs/MTPを通る新candidate modelは未実行。selected FP32 reference
比較、captureされた実graphの復元、callback/copy/drain/graph不確実性の全注入、task品質、
物理262144/RESTOREは未合格。default ONや速度採用はしていない。
次は同78453入力・FP32/R32768/Direct1の診断比較を行い、既存の31457280B referenceへ照合する。
その後GPU保持prefix増加とcoarse bottleneck計測、clean独立反復へ進む。

直前Direct clean cohortは3成功/4attemptで、4回目にVRAM headroom判定で終了した。
Direct0=278.2、Direct1=273.9/298.4のみ。残りは未実行、優劣・ばらつき除去を主張しない。
この失敗を成功へ変えていない。prefill反復600超→1000、decode70目標は未達で継続する。
