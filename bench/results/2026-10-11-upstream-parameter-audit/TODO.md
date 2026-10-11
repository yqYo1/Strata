# 本家の設定調整を確認する TODO

2026-10-11 のユーザー指示。担当: メインエージェント。
XeStrata 更新版の評価後、最新本家 v0.1.43 の比較を確定する前に扱う。

- [x] 直近の過去レシピを確認し、明示設定と実効設定の差を記録する。
- [x] 保持した索引付きレポートの範囲で設定探索の履歴を確認し、本家単体の探索とローカル版の設定流用を区別する。未保持・未索引の試行まで存在しないとは断定しない。
- [x] 現行本家の対応設定と、現時点での実行可否をソースから確認する。未修正 pristine の長入力経路は実行条件を満たさず、新規性能は未測定と記録した。新ツールチェーンの実行資格は別途確認する。
- [ ] 不足があれば、本家単体で prefill と decode の設定を個別に調整する。チャンク幅、実効キャッシュ量、CPU ワーカー数、CPU/GPU 分担、KV 常駐量、推測幅などを、対応範囲とメモリ上限に従って選ぶ。
- [ ] 32K 以上の同じ入力・モデルで比較し、選択に使った個別試行と、選択後の独立した検証を記録する。decode は少なくとも3回取り、ばらつきも報告する。既定設定、同一設定での比較、各実装で調整した設定での比較を区別する。

初回確認では、v0.1.40.2 と v0.1.41 の本家測定にも `--prefill 8192`、`--pool-workers 5`、`--expert-cache 128`、`--pcie-frac 0`、MTP4 などを明示していた。箱出し設定だけの測定とは言えない。ただし v0.1.41 本家の実効チャンクは6144、実効キャッシュは144スロットで、要求値8192/128と異なる。設定を指定したことは、その設定が最適である証拠にはならない。

別の「tuned」比較で実効8192/128を測った対象は control と integrated であり、本家 pristine は含まれていない。今回確認した範囲では本家単体の設定探索を証明できないため、この TODO は未完了。最新本家 v0.1.43 の pristine 長入力性能はまだ測定していない。

根拠のパス、SHA-256、要求設定、実効 INFO/PP、元の測定ステータスは [audit.json](audit.json)。この確認では GPU、ビルド、テスト、サービス操作、ログ削除を行っていない。GPU実行には同期・寿命の確認を先に満たし、未修正の危険な経路を設定探索のために実行しない。

追加の読み取り調査: [R357](../2026-10-11-parallel-round91/round357-official-pristine-parameter-search-coverage.txt)で保持した直近2版の長入力記録を確認し、単一レシピの反復であることを確認した。[R358](../2026-10-11-parallel-round91/round358-official-v0142-effective-tuning-controls.txt)に現行本家の対応設定と実効値の確認方法を記録した。安全な実行経路の資格確認と、本家単体の設定探索・検証は引き続き未完了。

履歴確認を拡張した [R361](../2026-10-11-parallel-round92/round361-historical-official-pristine-settings-search.txt) は、過去索引v113の465件のコンパクトレポートと直近の関連調査を対象にした。この保持範囲でも本家単体の設定候補・選択基準・最良設定の独立検証を確認できなかった。履歴確認は完了し、実際の本家単体の設定探索と選択後の検証は未完了として区別する。比較結果を「最適化済みの本家に対する性能差」とは表現しない。

[R367](../2026-10-11-parallel-round93/round367-official-independent-parameter-selection-matrix.txt)に現行本家で対応する設定に絞った選別案を残した。prefillは固定4096/6144/8192と独立したauto方針、decodeは実効ワーカー数・推測幅・キャッシュを別軸として扱う。要求値、admission後の値、実使用値をそれぞれ保存し、選択後に独立した3回以上のdecode検証を行う。これは未実行の候補表であり、最適化完了ではない。

2026-10-11の再確認では、上記6件の元の設定記録・実行controllerと、関連3報告のSHA-256がauditの保存値に一致することを確認した。[再確認記録](evidence-verification-v2.json)に元の個別statusを保持した。結論は同じで、箱出し設定だけではないが、本家単体の最適設定探索・独立検証は未完了。新しい性能測定は行っていない。

「箱出し」の意味もソースで区別した。v0.1.41のsetupが生成する標準configはprefill=auto、expert-cache=auto、spec=4、spec-min-p=0.5。一方、過去の本家測定は8192/128/4/0に指定し、262144 contextで標準のprefill借用を無効化していたため、installer標準configそのままの測定でもない。spec=4は標準と同じで、workers=5やpcie=0も自動選択と一致し得る。[既定値のsource監査](default-source-audit-v1.json)を根拠として、設定を明示した事実と、標準値を変えた事実と、最適値を探索した事実を分ける。今後はinstallerの標準auto方針も比較対象に含める。

追加の[R140 source安全監査](../2026-10-11-xe-quiet32k-balanced-campaign/research/research-official-v0142-long-prefill-source-identity-safety-admission-round140-v1.txt.lossless.json)では、本家0.1.42のtracked sourceを確認した。現在のHOST_USM/ATOMIC_HOST_USM capabilityでは、長いprefillのstagerに残るvolatileポーリングを設定だけで避けられないため、未修正本家の長入力は未実行。新コンパイラを用意したことはこのプロトコルを修正した証拠にならず、設定探索完了とも扱わない。

[R141のphase別設定探索案](../2026-10-11-xe-compact-moe-scratch-source-qualification/research/research-official-xe-phase-separated-parameter-search-contract-round141-v1.txt)を追加した。prefillの固定4096/6144/8192と標準auto方針、decodeの独立した設定選択と選択後の3回以上の検証を区別する。これは未実行の選定案であり、本家の最適化完了ではない。

2026-10-12 の更新取得では、本家 v0.1.43 (`7695859aa13afb248bdb4d36f9b2e95707db079d`) と XeStrata xe0.1.42 (`da3d426079019daa6e01388d03b0d94ae67070b7`) の未変更 worktree を分けて用意した。これは取得・ソース確認であり、新しい長入力測定や本家単体の最適設定探索ではない。取り込み用には本家の要求全体の時間・トークン集計3修正だけを先に移し、CPU32テストと全指定lintの新規指摘なしを確認した。推論の本家全体を取り込んだ版とは扱わない。独立した設定探索と選択後の検証は未完了のまま。取得時の失敗と別途の検証、最新調査は [更新境界の記録](../2026-10-12-latest-functional-sync-boundary/main-corrections-and-next.json) に保存した。

2026-10-12 の [R169設定だけでの実行経路調査](../2026-10-12-unitrace-v711-health-qualification/research/research-latest-pristine-parameter-only-safe-long-input-paths-round169-v1.txt) では、本家 v0.1.43 の SYCL に Xe0.1.42 の全カーネル先読みは無いことを確認した。一方、stager/PLEのhost-USM volatileポーリングは残り、現行B570のcapabilityでは既存設定だけで同期契約を満たす長入力経路を確認できない。GPUで未変更の本家設定探索を行った証拠は追加されていない。履歴・設定の監査は完了、本家単体の設定選択と独立検証は未完了として保持する。

次回の本家比較を確定する条件:

- 標準auto設定、同一設定での比較、本家とローカル版それぞれで選んだ設定での比較を分ける。
- 本家単体の設定候補・選択基準・要求値と実効値を残し、prefillとdecodeを別々に選ぶ。
- 同じモデルと32K以上の入力を用い、選択に使っていない独立した検証を行う。decodeは少なくとも3回、各値とばらつきを残す。

箱出し設定だけではなかったという履歴確認は完了した。本家単体の最適設定探索・独立検証は未完了であり、それが済むまで「最適化済みの本家との比較」とは扱わない。今回も新規GPU測定は行っていない。

2026-10-12 の [R177設定選別案](../2026-10-12-xe-host-issuer-model-qualification/research/research-official-independent-config-search-round177-v1.txt) を追加した。最初の比較では現在の262144 contextとKV方針を固定し、標準autoは独立した方針として扱う。固定チャンクの選別とdecodeのワーカー選別を分け、medianで選ぶdecodeは選別時も複数回、選択後の独立検証も3回以上行う。この案は未実行であり、本家の最適設定探索は未完了。今回完了した64/32KのGPU実行はローカル版の観測コード検証で、本家pristineの設定探索や性能比較ではない。

2026-10-12 の [R185 の独立設定選別案](../2026-10-12-prefill-emb-ple-alias-source-qualification/research/research-official-independent-parameter-search-contract-round185-v1.txt) では、本家 v0.1.43 の pristine ソースを再確認した。標準 auto、固定4096/6144/8192、条件付き auto:16384 を prefill の選別候補とし、decode は実効ワーカー数の重複を除いて別に選ぶ。選択後は独立した32K以上の入力と3回以上の decode 検証を必要とする。これは未実行の案であり、現行 host-USM 同期契約の問題を設定だけで避ける実行経路も未確認のため、本家単体の最適設定探索・独立検証は引き続き未完了。
