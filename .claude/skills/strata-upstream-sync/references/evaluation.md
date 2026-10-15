<!-- SPDX-FileCopyrightText: 2026 yayoi -->
<!-- SPDX-License-Identifier: LGPL-3.0-or-later -->

# 計測と部品別採否

## 比較する構成

上流単体は source を変更せず、fork のパッチを適用せずに build/run する。
唯一の上流はNiko1221/Strata。XeStrataはライセンス制約により性能の参考値だけに使い、
独立した実行可能性・設定・性能を記録し、本家の更新評価とは分ける。
速い結果でもそのsource・派生コードは採用対象にしない。
fork は現在の採用構成、過去の速かった構成、部分移植後の候補を比較の起点にする。
過去の良い設定を忘れて最近の遅い診断設定を新しい基準にしない。

上流の箱出し設定だけで「上流は遅い」と判断しない。
この環境で chunk、cache/VRAM 配分、worker/task、backend、speculative decoding 等を
関連するボトルネックに沿って調整し、その探索範囲、実効設定と最良構成を残す。
設定探索は事前に範囲、run期限、ログ容量、打ち切り条件を定め、失敗時に自動再試行しない。
未変更の上流に対象backendや互換APIがなければ、実行不可の理由と検証範囲を記録する。
未変更という条件を破る移植や別benchmarkで、その上流単体を測ったことにしない。
可能な必須機能の移植・CPU検証と、推論部品の評価は続ける。
上流で実行できなかった機能テストと、fork側で通った適合テストの証拠を区別する。
prefill と decode の最良構成が別ならその違いも記録し、両者へ同じ設定を強制しない。
設定探索と source/部品の比較を分け、設定差を kernel の効果として報告しない。

## モデル速度

速度比較には **Niko1221/Strataの未変更 `bench.sh` とその標準クライアント** を使う。
名前だけで選ばず、どのcheckoutのscript・依存clientを実行したかを固定する。
両側へ同じ client SHA を使い、対応するサーバーを localhost に別々に起動する。
上流の実際の help と現行のプロジェクト測定手順を確認する。
この作業の標準例は `./bench.sh --url http://127.0.0.1:18095 --long --think --runs 3`。
32K項目に加え、標準suiteの **agent sessionも毎回評価する**。全turnの個別値とsession単位の
完走、入力/read/reuse、生成数、TTFT、decode、draft offered/acceptedを保存し、報告にも含める。
このagentはモデル自身の返答を次turnへ含めるfile-review会話であり、実際のtool実行ではない。

ユーザー指定の長いdecode評価には、このforkの拡張 `./bench.sh --long --long-generation --think --runs 3`
も両側へ同じclientで使う。Nikoの無変更標準結果と、変更したbenchmarkの追加項目を識別する。
追加項目は実入力32768以上、thinking込みの実生成1024以上、fresh prefixを確認する。
regular-codeとvaried-proseはMTPが強い/弱い候補であり、名前だけで分類しない。
**thinkingは有効にする**。1Kは出力の例で、上限ではない。v1は1152全部がthinkingとなり、
v2のthinkingOFF案はユーザー意図に合わず未実行で撤回した。v3は上限8192でscreeningし、計測harnessのCPU上限で日本語要求が中断した。
v4は上限16384、thinkingON/xhigh、別の思考上限なし、context262144を使う。
agentもthinkingONで各turn最大8192と実際の思考/回答の履歴を引き継ぐ。
thinking後の本文も512文字以上出たことを確認する（文字数をtoken数と呼ばない）。
全量thinkingなら上限/context/課題を見直す。入力と出力余地を確認し、workload版を混ぜない。
3回以上の有効sample、各回256以上のoffered、受理率の各回60%以上/40%以下で確認する。
分類が確認できなければ未完了と記録し、速度に基づいて都合の良いpromptを選び直さない。
要求開始前のbody hashと各結果をdurable journalへ保存し、hash chain/stateを確認する。
未完了sessionはagent集計へ入れず全個別値を保持する。
全個別値・request/output hash・終了理由・実生成数・draft countsを残し、比率は合計countsから
求める。欠けたusageやstream piece数、早期EOSの短文を1K tokenと扱わない。
1Kは24-token decodeより長い観測窓を得る初期条件であり、安定性を保証する長さではない。
ばらつきが残れば独立process反復や生成長を増やし、MTP有効/無効の比較なしに速度向上を
MTPの効果と断定しない。[実装と計測手順](../../../../docs/BENCHMARKS.md)を参照する。
独自 GEN、kernel/帯域測定、デバッグの所要時間を、この速度比較へ代入しない。
独自 GEN は品質・同期・lifetime 等の診断に使い、過去の速度記録は元の意味で保持する。

短すぎる入力だけで比較しない。標準の長文ケースを使い、実入力・read・reuse を確認する。
例えば上流の約32Kケースが実入力32049なら、その数を記録し、32768以上と偽らない。
標準ケースと厳密32768以上の検証は区別する。64K程度でも prefill1000 t/s の目標評価は
許容されるが、上流が対応する長文指定をまず確認し、方法を追加・変更した場合は明記する。
推論性能とモデル読み込み・server startup・全request wallを別に残す。

少なくとも各3回の個別値、実行順、中央値と範囲を残す。
差がばらつきと重なる部品・phaseは、その対象だけでも反復を増やして判定する。
同条件の交互実行を使い、cold/warm、長いdecodeと24 token等の短いdecodeを区別する。
平均だけで採用せず、成功以外の全結果、未完了要求と停止理由も残す。
生成履歴が異なる場合、処理 workload の違いも限界として記録する。

source SHA/diff、binary/hash、compiler/flags、driver/SYCL/LevelZero、hardware、モデル/
入力の識別、全argv/実効env、KV/context/chunk/cache、MTP/draft、CPU配置、reuse、
実生成数、各phaseの時計位置、ログ/validation負荷を記録する。
過去の速い値があれば session の user-facing 報告から該当の元測定へ遡り、上記を照合する。
測定方式が変わっただけで古い値を棄却せず、比較可能な範囲と追加検証を示す。

## 部品の比較

実モデルで使う量子化・shape・配置・token数の入力を、同じ履歴から用意する。
CPU/GPU、MoE GU/Down、射影、Attention/GDN、逆量子化、転送、host発行・同期等の
各候補について、現行の実経路を確認してから測る。
正しさ・同期寿命を確認し、単体の elapsed/event/host time の範囲と負荷を明記する。

単体の改善に加え、その部品を入れ替えた同条件モデル測定を取る。
全体差が小さい場合でも実部品の利益が測定で確かなら、dispatch を限定して保持する候補に
できる。ただし全体改善と呼ばず、退行・品質・保守の影響を別に判断する。
律速箇所とこの hardware の計算・RAM/PCIe/VRAM 上限から、期待できる改善を検討する。
モデル全体の差だけからどの kernel が優れるかを逆算しない。

## 数値、256K、GPU の検証

生成文の完全一致を一般的な品質条件にしない。
計算経路や精度が変わる候補は、同じ固定履歴・位置の logits/尤度・分布、state、
継続/RESTORE と必要な task 成績で評価する。異なる生成文の後半を同じ履歴と比較しない。
精度を落とす場合、適用可能な一次資料を確認し、必要ならこのモデルで品質劣化を測る。
資料と実測の範囲を越えて「劣化ゼロ」と言わない。

最大 context262144 を設定・起動できただけでは全コンテキスト検証にしない。
全入力262144で出力余地がないケースと、出力枠を残す近傍ケースを別々に検証する。
テンプレート等も含む実際の入力・消費・出力token数と最後のセルを確認し、
出力余地がないケースで正常な拒否や出力ゼロが仕様ならその結果を記録する。
ただし入口の拒否だけでは262144位置の実計算を証明しない。実消費を確認する probe と、
近傍での通常生成、範囲外の拒否を分ける。engine側の最終消費位置・最後のcellや
各実chunkの位置範囲等で消費を確認し、request accountingだけを実行証拠にしない。
probeが未対応なら未完了として残す。
関連する容量・状態・処理順変更には実物理入力とRESTOREの検証を用意し、
少量のfixture、64Kの成功や論理capacityを256K完走と代用しない。
未実施の範囲は採用判断と報告に残す。

未知の SYCL/LevelZero 完了状態や危険な寿命を既知の完了と扱わない。
GPU障害がある版は、ソース/API契約と停止箇所を調べ、対策後に bounded な小さい確認から
進める。初回診断はLevelZero/URのprogress・error出力を使い、速度はログ負荷を外した
別プロセスで測る。queueの依存・drain、pointer/graph の寿命も実装の採否に含める。
GPU reset、サービス停止、回復scriptの実行をこのスキルで自動承認しない。

## 保存

対象repoの ARTIFACT_RETENTION 等の方針に従う。source/コマンド/環境/全個別値/
正しさ・fault/終了証拠と採否をcommitし、成功した長大traceや重複pollingを全件残さない。
未解決の機構ごとの代表診断、現在の比較・RESTORE consumer が必要なfixtureを残す。
owner、byte budget、consumerと見直し時点を明記する。
削除はrunが閉じた後に測定と直列で行い、path/hash/reason manifestを残す。
失敗の元statusを変えず、最後に必要なfixtureやactive runを削除しない。
