# 長文prefillの実プロファイルと転送会計

2026-10-15。計算上のVRAM容量は `prefill-capacity-plan.ja.md`、今回の実行は
`prefill-window-profile-long-v4.json`。両者の条件を混ぜない。

## 結果と解釈

実34603入力、batched34602行、C4096、FP32 residual、GPU保存4096行、K8V8/resident32768、
physical capacity262144、MTP4。binary SHA256は
`b1cc9cb45554e89c060690420dc77b16fab8f512c55992c5faf367551ad3c8c0`。
collectorはIntel PTI `6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033` の実build。
同一queue等の既存安全設定を維持し、native source/binaryを変更せず実行した。

unitraceをstartup中はpauseし、INFO/READYを検査後、1要求の直前にresume、DONE後にpauseした。
要求はprefill、重み・状態の復元、1 token生成を含む。DONEはprompt233330.6 ms、decode5026.5 ms。
この値は**clean throughputでも純prefill区間でもない**。
正常QUIT、native/wrapper/control全exit0、全owned process退去、private collector session解放、
同じboot、新規kernel GPU fault0、driver dumpなし。GPU障害は再発していない。

全8066326イベント・raw SHAを検査したstreaming解析と、unitrace自身の独立した
device timing表は全operationのcalls/合計時間が一致した。
`prefill-window-interval-summary-host-v2.json` に照合結果を保存した。

| 指標 | この収集窓での値 |
| --- | ---: |
| 可視GPU先頭から末尾まで | 238.371 s |
| GPU operation時間の和 | 140.145 s |
| GPU稼働区間のunion | 140.105 s |
| compute区間のunion | 100.154 s |
| copy区間のunion | 39.950 s |
| compute/copy区間の重なり | 1.690 ms |
| GPU稼働区間の間のgap | 98.266 s |
| CPU wait区間のunion | 75.817 s |
| waitのうちGPU稼働と重なる部分 | 75.147 s |
| waitのうち可視GPU稼働と重ならない部分 | 0.670 s |

主要prefill attention108回が44.052 s、GU dequant33758回が8.251 s、
GDN recurrence396回が3.541 s。共有kernelの数字には復元/1 token生成も混ざる。
waitを75.817 sの削減可能なCPU往復と読むのは誤りで、ほぼ全て有用なGPU処理を待っている。

gapの追加照合では、記録されたCPU APIと重なるgapは30.427 s、そのうち
`zeModuleCreate`19.316 s、`zeCommandListAppendLaunchKernelWithArguments`9.745 s。
残る67.839 sはこのtraceのCPU API内ではない。SYCL側CPU処理、submit間隔、収集の負荷等を
分離できていない。98.266 s全てを本番のhost制御による損失や、batch化で消える時間とは扱わない。
外部monotonic/UTC時刻への補正は行っていない。

## 実copy量とsourceの対応

Mainは全size histogram（省略のない18/20/4/7 bins）を独立に再計算した。
`verify_prefill_copy_accounting_v1.py` と `prefill-copy-accounting-main-v1.json` が再現物。

| sourceから対応が付くもの | copy APIへ要求したbytes |
| --- | ---: |
| 全48層、512expertを一回ずつupload | 50292326400 |
| host residual store | 58727710720 |
| host residual reload | 58727710720 |
| GPU residual prefixのstore + reload | 15770583040 |
| QSA KVの既存prefixを各chunkで再stage | 1868562432 |
| PLE入力行をupload | 354324480 |

全copy要求187820945340 Bのうち2079727548 Bは未分類で残す。
入力全体34603 tokenのうち、batched passは34602行。GPU4096行、host30506行、
host側は4096行×7と1834行のtail。47層境界の各方向は
`47*(7*167772160 + 75120640) = 58727710720` B。

R681 v1のresidual説明には34603/30507/1835行という一行ずつの誤記がある。
**報告のbyte数自体は正しく、上の34602/30506/1834に対応する**。
またR681が未分類としたH2D354324480 Bは、Mainが`prefill.cpp:1764`のPLE uploadと照合した。
元reportのbytes/hash/statusは変更せず、この訂正と独立計算を優先する。

PTI sourceで`M`は`ZE_MEMORY_TYPE_UNKNOWN`、`H`はhost、`D`はdevice、`S`はsharedである。
M2D/D2Mという表示だけでRAM/PCIe転送と決めない。今回はsourceのowner経路とsize分布を照合した。
[]のbytesはAPI引数の長さであり、実リンクを流れたwire bytesの計測ではない。
KV host mirror等のkernelからのmapped-host書込みはcopy API集計に含まれない。
copy40秒という数字だけでPCIe律速と結論していない。

## 次の実装候補と調査

R678は一層のKV stage寿命を調査し、現在のhost/resident mirrorを保ったまま
同一層のprefix再uploadを省く案を切り出した。Solの`layer-major-kv-prefix-reuse-v1`はsourceのみ。
この案は新しいKV owner/commit/RESTOREを導入せず、**VRAM容量は増えない**。
今回34602行では省く候補は約1.87 GBで、hidden反復転送と比べ小さい。
256Kでの転送量・実時間は別に評価する。

R679はFlexGen/ZeRO-Inference/vLLM等のlayer-wise offload一次資料を調査した。
一層のweight/KVを入れ替えつつhiddenをGPUに残す設計と、実装済みのall-state所有配置は異なる。
容量表の「一層KVでC8192も収まる」はindexer/page stateも一層分とした場合。
他11層のindexer等372021936 BをGPUに残すと、C4096は約0.978 GiB余り、C8192は161033920 B不足する。

R680は既存`native_qsa_indexer_append_batch`が内部で一行ずつkernelをsubmitすることを確認した。
実収集では対応するkernel415284回、GPU service1.917 s、全GPU operation1017611回。
targetのbatched行だけなら`12*34602=415224`であり、差分60回を純prefillへ押し込まない。
restore/last prompt token/1 output等を含む窓である。
真のdevice batch化は投入回数を減らす候補で、1.917 sだけから効果を予測しない。
Mainは実sourceの行ループを確認し、別の6.1 Solにsource-only候補作成を依頼した。
pos0 seed、half-round順序、tail/dead/spare/block_posの単一writer、causal top-k、
非zero prefix、最終physical262144、RESTOREのexact比較をMainのserialテストで管理する。

BF16 storage v2もSolから受領したが、v1が漏らしていた既存CPU audit fixtureのenum対応を含む
置換proposalであり、未適用・未build・未実行。FP16/BF16の実品質は未合格。
head releaseの744699900 Bというpayload回収を実測済みとは言わない。
短入力のdecode配置経由prefillと既存FP32 referenceは維持する。

Mainは続いてhead-scope/KV-prefixの**提案したproduction helperそのもの**を
通常C++17とASan/UBSanで実行し、4組ともPASSした
（`prefill-owner-helpers-host-v3.json`）。GPU owner本体、graph、DMAはこのfixtureにない。
head fixtureの元の一行ifはGCCの`-Werror=misleading-indentation`で失敗し、一改行だけを加えた
Main v2 fixtureを使用した。次の試行ではMainのinclude配置不足でKV fixtureがcompile失敗し、
提案headerとbyte同一のproduction include path overlayを作って解消した。
元v1/v2失敗結果とwarning条件は保存し、test logicやWerrorを弱めていない。
その後Mainはhead-releaseだけをexact diff/pinsで実sourceへ適用し、新しい別build directoryで
SYCL本体全体をcompile/linkした。全command0、Werror、source固定、range-size safety flagと
prefillの`-fno-fast-math -ffp-contract=off`を確認した。`prefill-head-release-build-v1.json`が結果。
新binaryは`c03ca79d5a75faaed075c9ce354fd5647f78a39853150a55992753d7d93ba42e`。
通常無効のままで、元b1 binary/archivesは変更していない。
NativeHeadのみの実packed-byte/logit比較fixtureもproduction librariesへリンク済み
（`prefill-head-component-build-v1.json`）。続いて実GPUで二回実行し、521472000 B全体と
248320個のprojection logitが両cycleでbitwise一致した（`prefill-head-component-v1.json`）。
確保addressも両方で変わった。実free増加は521539584/521535488 B、全exit0、新規fault0、dumpなし。
初回確認なのでUR API tracing/parameter validationを有効にし、runtime error0。
target head単体の証拠であり、MTP duplicateとの合計744699900 Bや全engineの空き量の実証ではない。
二回目は64 MiBのfillerを挟み、free後の即時counterが64 MiB小さかった。
applicationのfreeは成功しているが、allocator/driverのpoolingや計数時点を分離していない。
全engine admissionへ単体のfree差分を無条件に足さない。
47,983 BのAPI traceは次のwarm combined lifecycle/allocator比較をconsumerとして一つだけ残す
（`prefill-head-component-retention-v1.json`、64 KiB予算）。
MTP/verifier/モデル/RESTORE/物理最大context検証は引き続き別の未実施gateである。
R683の独立sourceレビューでも明確なdangling-owner/default-off退行は見つからなかったが、
GPUのcompletion/graph lifecycleを証明する実行ではない。

次のcombined gateは`run_prefill_head_lifecycle_v1.py`。
公式bench.shのprefill-onlyを二回、actual input/read>=32768、thinking ON、24出力上限で行い、
同じserve内の繰り返しhead退避/復元とMTP subset再構成・decode/終了を観察する。
parameter validation/warningとSTRATA_TRACEを付けるため**clean性能試験ではない**。
この準備時点では未実行。別のclean三回測定と実RESTORE/最大contextは後続gateである。

## 失敗と保存

collector harness v1は正常INFOをREADYと誤読してGEN前に失敗。v2は513 tokenの正常な
EOS/finish=stopをlength限定で誤拒否した。両方FAILEDの元結果を残し、native退去と
private collector objectの解放を確認した。native/GPUの故障ではない。
v3 componentは正常終了したが、flow IDごとのcategoryをそのまま数えて40 MBの重複receiptを作った。
元のcategory保存fixtureはprivate領域に一つだけ残し、Git用は7群へ集約した。
v4ではこれを修正し、host28casesと実componentをgateにlong収集を完了した。

初回interval解析v1のlong出力はgeometry別6734群/4.6 MBとなり、宣言した2 MiB上限を超してFAILED。
v2は同じrawをbase kernelごとへ集約し、全events/calls/時間を保存、独立表と照合した。
表との初回joinもSIMD_ANY suffixを扱えず失敗したが、全該当群を合算するadapterで一致を確認した。
これらの失敗statusは別receiptに保存し、GPUを再実行した結果へ置き換えていない。

raw timeline約1.60 GBは全イベント集計・gap解析・独立表照合の完了後に削除する。
`prefill-window-raw-retention-v1.json` が各path/hash/bytes/理由と、残すfixtureのconsumerを記録する。
原本のrun receiptは変えず、元trace pathが退役したことはこのmanifestで示す。
大きなtimeline/失敗した4.6 MB中間集計をGitへ追加しない。

プロファイルは旧binaryで取得後、head-releaseのproduct sourceだけを通常無効で統合した。
BF16/KV-prefix/device-indexer-batchはまだ未適用。
公式bench.shの直前clean結果は実78453入力で219.1/222.3/221.5 tok/s、中央値221.5。
600/1000 tok/s、品質、物理最大contextの実行、RESTOREの新しい合格はまだ得ていない。
