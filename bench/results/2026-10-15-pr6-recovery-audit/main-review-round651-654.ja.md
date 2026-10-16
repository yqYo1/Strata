# 適応eventの失敗処理とprofile候補の独立確認

2026-10-15。前ターンは実exec16956を直接pollし、同一nativeのPID/start時刻と生成の進行を
確認したverified wait。今回も同じhandleがrunningである。full OFFの32K三回、agent三session、
code長文三回とprose二回は完了し、最後のproseはthinkingからansweringへ進んだ。
終端receiptはまだない。新しいtest/build/GPU/cleanupは重ねず、root source/client/native/profileを
固定してsourceレビューだけを進めた。現在の約200 PP/6--10 TGをPR6回復と扱わない。

## 受領した修正案とMainレビュー

6.1 Solの`checked-adaptive-event-v1/`を受領した。generate.cppのSYCL serve/standaloneの
primary event記録二箇所だけを新helperで包む未適用差分である。成功は元のrecord一回から
returnし、同期、数学処理、cache policyを増やさない。失敗はそのissuerでcodeを保持し、
直ちにcudaGetErrorStringのTLS-backed detailを出力・flushして既存cleanup_fail_stopへ進む。
この既存関数は_Exit(EXIT_FAILURE)するので、pending公開、後続consumer、DMA/cache/commit
ownerの普通のC++ unwindingへ進まない。Mainもcompat.cppのrecord/query/同期/error-stringと
両callerを照合した。単にfalseを返す処理とは退去の意味が異なる。

fixtureは実proposal helperをincludeし、fork childのpipe markerとexitで成功、status77失敗、
record callback例外を区別するsourceである。成功RPCDDD、失敗RE/Rを要求し、owner destructorと
publication/consumerへの到達を検査する。ただしfake API/ownerであり、実SYCL event、DMA、
pending vectorやnative caller integrationを実行した証拠ではない。assertを有効にしたMain所有の
CPU normal/sanitizer実行、その後の実integration buildは未実施。

stage/peer event、先行H2D失敗の部分退去、device map upload、commit_exchanges短countは未修正。
単一SYCL primary/no-stage/no-peer以外へ一般化しない。この修正案だけで正のadapt-swapsを
有効にしない。current adapt0 baselineの過去/現在のGPU障害原因を特定したという主張もない。
R652もproduction serverの自動再起動までは確認していない。_Exitはエラー後の通常継続をやめる
設計であって、driver復帰や次processの成功を保証するものではない。

proposalのpins.jsonには受領時research-indexのhashも入っている。このindexは追記される可変の
研究目録であり、future host資格確認ではcurrent source pinと混同しない。元indexのbytesは
commit5883b757から復元でき、受領時hash d216959a2c9639b90a1ca9768e780cf81f59798e7446550197d393d68eb1dcb7
を持つ。proposal原bytes/manifestsは維持する。source/header/fixtureと原artifact hashesは実検証の
入力として別に固定し、mutable registryの更新をcode変化や実test成功に読み替えない。

## Lunaの反復調査

R651は既に作成した196632-byte CPU-history-first profileを限定して読み、既存writerの順位・
全24576pair・逆引き・legacy128 quotaを独立確認した。候補hash d97b2f48...、9816既知CPU pair、
training10159rowsは一致した。未知GPU頻度を観測zeroに変えていない。native loader、held-out
hit/速度/品質は未確認のままである。同容量training改善を本番性能改善と数えない。

R652は失敗eventでunwindしてはいけない契約を確認し、R653は新proposalを独立レビューした。
R653は厳密baseからのin-memory再構成hash fd0613c6...と両callerを照合し、限定範囲内の具体的
blockerを挙げなかった。ただし実testは一度も行っていない。record例外以外のdescribe例外は
fixture未網羅という限界も残した。Mainはdescribe実callbackが現在のC APIであることを確認した。

R653とR654のscopeをずらして並行依頼し、既存research-indexと関連reportを共有した。実装担当と
調査担当は別で、Mainが実testを直列管理する。R654のFINAL後、Mainはbench.shの実在/trackと
実process間nonceの扱いに誤りを見つけたため、原reportを保存して別v2の訂正を依頼した。
調査途中への介入ではない。

訂正版R654 v2を受領し、Mainもroot bench.shが実在するlauncherであることを確認した。比較は
ユーザー指定の./bench.shを維持する。source/version/run番号だけでは、時刻nonceを含むcalibration
から算出されるfiller長まで一致しない。将来の同じworkload-id指定に加え、実際のordered body hash
比較が必要である。profile-only pilotは別process・同一128 quotaで行い、nativeのrequest別cache
hits/lookupsをcompletionに対応させる。clientの80行excerptだけでは全hit行の保持を保証しない。
これも未実行の設計であり、元のtraining本文が残っていないため学習入力との完全な独立性は
証明できない。v1原bytes/hashを残し、誤ったlauncher/入力同一性の説明はv2で明示訂正した。

full OFF終端後は、元clientのままnonceを二つの実body hashから回復し、既存の比較tool CPU検証を
行う。固定入力の同一性と、生成履歴を含むagent後続入力の自然な違いを分ける。IQ ON比較は
real gateを満たした時だけ実行する。wave census/PCIe helperもGPU再占有前の直列CPU検証対象。
新しい未実行proposalを増やすより、これらの実行済み証拠を優先する。

1000 PP/70 TG、PR6性能回復、物理262144/RESTORE、task品質、upstream非推論機能/default反映は
引き続き未完了。今回のsourceレビューをそれらの達成と扱わない。
