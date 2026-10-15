# CPU GU再利用観測のMain検証

2026-10-15。目的は展開GUをキャッシュする前に、実モデルのCPU所有expertの
再使用順・容量別miss/fill数を測り、単体の高速化が実際に償却できるか判断すること。
推論方式、重み、GPU所有expert、既定設定は変更しない。

## 適用とCPU検査

Sol v2提案の`source.diff`を適用した。原提案はそのまま保存し、Mainの差分は製品ソースと
`apply_build_cpu_gu_reuse_v3.py`に記録した。新しい観測は
`STRATA_CPU_GU_REUSE_OBSERVER=1`と既存joined ledgerの両方が必要で、既定はOFF。
issuerが本来のCPU job配列のlayer/expert/ntを記録し、既存poolのparked復帰後に完了数を確定する。
worker側にallocation、LRU更新、新しい同期点を加えない。

最初の`cpu-gu-reuse-build-v2.json`はCPUコンパイル失敗である。
引数なし書式wrapperが`-Werror=format-security`に該当した。GPU実行前の失敗で、
GPUハングや数値FAILではない。Mainは文字列だけのoverloadを`%s`出力に分け、
`%s %n 100%`が文字列のまま出力される検査を追加した。警告を無効化していない。

`cpu-gu-reuse-build-v3.json`は通常O2・ASan/UBSan・既存joined lifecycle検査と
実SYCL本体ビルドが全て正常終了した。CPU oracleは製品LRUと別のvector LRUで、
2 policy×4容量、nt1-onlyの独立履歴、形式/nt別仕事量、実expert由来、fill/eviction/resident byte保存、
unsupported・overflow・出力上限・ULEB/base64復元を検証する。
状態は1,772,192B、出力上限256KiBで、実データをGB単位に展開する機能はまだない。
実行本体SHA256は`92acf71e6b4fd914f68c8cefcae7e03f170adf62b5e3b94cf31af747344183f7`。
本体の再buildにより旧Direct比較の`0c281323...`とは別binaryになったため、clean比較は
新binaryのOFF/ON双方を改めて測る。

## 実モデルの小規模資格確認

`check_cpu_gu_reuse_v1.py small`は同一新binaryの観測OFF/ONを別processで実行した。
513入力、257チャンク、8生成、通常MTP4、int8 KV、context設定262144。
これは速度比較ではない。両方で8生成IDが既存数値基準と一致し、MTP仕事量と
CPU仕事量6855jobs/11931expert-token rowsが一致した。OFFでは新footerが出ず、
ONではcomplete=1、unsupported133を含む全仕事量の保存を確認した。
別のPython consumerが98304個のULEB countを復元し、全expert/nt、format、
8 LRUsのhit/miss/fill/eviction/resident、既存joined job bucketと照合した。
正常終了・所有process回収・新規GPU障害0、boot/driverに変化なし。
診断付き各1回の小入力時間差を速度の根拠にしない。

実使用判断は続く34603入力・128生成の`long`測定で行う。
GPU expert cacheの`--expert-cache 128`は全48層で合計128 slotを意味し、
`--expert-cache-per-layer`は配分policyである。各層128 slotではない。
この既存legacy配分は47層に各2・最後の層に34で、balanced配分案は未適用。
request-cold仮想LRUのpayload予算は256MiB/1GiB/4GiB/8GiBで、
実RAM確保・worker同時アクセス・request間永続cacheの証明ではない。
実装時はpage/metadataも予算に含め、8GiB live available RAMの余裕が必要。

## 展開費を含む採否

R603の調査でfill費の重要性を再確認した。Mainは元の4 process・12 ABBA blockの
個別数値から`analyze_cpu_predecode_amortization_v1.py`で再計算し、
`cpu-predecode-amortization-v1.json`に全24組の値と元receipt hashを保存した。
概算閾値ではなく、この再計算値を判断に使う。

実clean設定は`STRATA_IQ_SINGLE_EXACT=0`なのでpacked baselineはGGMLである。
比較対象を勝手に既存exactへ置き換えない。GGMLに対する中央値の節約と同一processの
fill CPU時間から得た必要使用回数は、cold: type18=15.52–20.73、
type22=23.62–36.29、type21=10.56–12.48回。
overwrite refill: 2.91–4.14、4.85–6.56、2.23–2.64回。
既存exact helperに対する別計算も保存した。
miss時に展開後のkernelを使う場合は、そのmissの呼出し自身も使用回数に含む。

これはsingle-thread component CPU serviceの算術であり、model wallの保証ではない。
初回page touch、slotの再利用、allocation syscall、workersによる並列fill、RAM競合、
lookup/pinning費、request間保持で変わる。nt2..4は速度未測定なので、all-nt policyの
hit数からnt1 cacheの採算を主張しない。単にretouchやhit率が高いだけで採用しない。

## 次の測定と残る範囲

実long観測後、公式`bench.sh`でDirect OFF/ONを同一新binary・同一設定で比較する。
3回のnominal32Kと3 agent sessions、thinking有効のstrong/weak長文生成を対象とする。
nominal32Kは実入力が32768未満になることがあるため、実token数を必ず記録し、
32768以上のlong-generationを長文prefillの主要根拠にする。
profiler/validation/ledgerはclean測定で無効にし、安全性の既存設定は維持する。

今回で1000 PP/70 TG、PR6の速度回復、物理262144全入力・RESTORE/cancel/繰返しの
認定はしていない。fork defaultにある新しいCLI/server機能とfeature/syclの統合も
未解決の別作業であり、履歴保存を機能統合済みとは扱わない。
