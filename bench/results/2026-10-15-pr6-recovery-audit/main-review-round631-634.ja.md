# Stage再利用・decode容量・比較入力の次の判定

2026-10-15。Mainのsourceレビューと、read-only Luna R631/R632/R633/R634の報告を
受領した。build/test/GPU/cleanupの所有権は引き続きMainにある。進行中のclean full OFF
は同じexec handle16956で生存を確認しており、再起動していない。ここでの追加作業は
読み取り、source-only proposal、Gitの保存である。

## 現在の比較計画で直すこと

`tools/strata_bench.py:Suite.__init__`は`self.nonce = str(int(time.time()))`を使い、
calibration・長文prompt・agent・long generationのprefixを作る。現行clientの
`comparison_identity`はこの実nonceを含まない。同じCLI/source/設定でも、独立armの
実リクエストは変わり得る。現行Main clean-bench recipeの設定/hash一致だけでは
入力一致を証明できない。生成された返答を次turnへ入れるagentや、calibrationで決まる
長文長にも依存するので、固定nonceだけで全リクエストが一致するとも言わない。

現在のfull0は変更せず完了させ、個別速度・長文/MTP・安定性・全request journalを残す。
元receiptの成功/失敗は変えない。**旧recipeのfull1をそのまま起動して厳密な比較が成立した
と扱わない。** 次のmatched比較には、fixed workload IDと実request bodyのhash照合を
導入する。`bench-fixed-workload-v1`を別source-only提案として用意し、閉じた境界で
Mainがhostテスト・適用・依存pin更新を行う。進行中のpinned clientは書き換えない。
現在のfull0の入力を後続armで再現できるなら、全body hashとclient差分の非計測部だけの
変更を確認して再利用できるか評価する。再現できなければ新しいmatched OFFも測る。
変動の重なる効果は、独立process・逆順でも確認する。

Mainの`recover_bench_nonce_v1.py`もsource-onlyで準備した。jobが閉じた後、pinned
clientを書き換える前に、所有process/sessionが退役済みであることとjournal chainを
確認し、元commandの有限epoch範囲からcalibration200/800双方の実body hashに合う
唯一のtimestampを探す。未実行であり、nonceの回復自体は性能/品質の合格ではない。
これによってcurrent full0を無条件に捨てたり、旧receiptへnonceを後書きしたりしない。

## R631: stageは一層分の共有領域

MainとR631の独立照合は一致した。`m.stage`は単一で、INT8 4224 B/page、1056 B/cell。
現在のlayer-majorは一層の全chunkを連続して処理するため、そのpass内だけは、append済み
の連続prefixを使ってrestageを省ける可能性がある。通常chunk-majorや異なる層・request・
RESTOREを跨ぐ再利用は根拠がない。owner、cell prefix、generation、不連続position、
cancel/error/failed drainでの無効化を含むsource gateが必要で、本体へは未適用である。
論理copy byte算術からwall時間の改善を見込まない。

## R632: decode容量screen

Mainも現packの48行`native_experts.txt`と`ExpertCache::layer_slot_range/open_sized`を
読んで計算を照合した。全blobは256 B alignment済み、各層一slot分の和98227200 B、
最後の層2662400 Bである。

| Total slots | Layers0–46 / layer47 quota | Exact arena B | Delta B from128 |
| ---: | ---: | ---: | ---: |
| 128 | 2 / 34 | 281651200 | 0 |
| 192 | 4 / 4 | 392908800 | 111257600 |
| 256 | 5 / 21 | 533734400 | 252083200 |

これは実allocation/free差ではない。`--expert-cache`の一値だけでscreenできるが、
resident expertの集合は単純な追加ではなく、層ごとの再配置となる。明示容量は自動縮小
されず、arena確保の成功だけで後続Prefill/MTP/Verifierとreserveのfitは保証しない。
capacityはfresh startupでだけ変え、adaptive swapsは0のまま、実free/slot/bytes、
route owner、CPU service、MTP workと生成速度を読む。cache退役proposalやbalanced
quotaは混ぜない。CPU/GPUでactivation/accumulationが異なるためtoken一致だけを
品質判定にしない。

過去の128出力に36.734 CPU thread-seconds、97% CPU routeという条件が維持される
なら、同じ六つの参加threadでは70 output/sに必要な20.09 CPU s/sを供給できない。
これは現条件のservice制約であり、ハードウェアやモデルの不変の上限ではない。
route/work、MTP acceptance、CPU arithmeticの変更で制約自体が変わる。

## R633/R634の扱い

R633のbounded2/4chunk expert waveは全prefix遅延案と区別する。実コードの先例と
route-slot順を保つFP32 Dmの案があるが、より大きいoneMKL GEMMの丸め、wave memory
admission、state/checkpointの時刻、MTP callbackの契約をまだ検証していない。
未実装・未採用であり、報告のwave plane byte subtotalをlive memory/速度に読み替えない。

R634は外部のAVX2実装から、そのまま移植すべき未導入IQ arithmeticを見つけていない。
既存のnt>=2 weight decode共有、sign reconstruction、prefetchと重複する案を追加しない。
IQKや真のQ8 weight再量子化は別のモデル表現・品質課題である。既存prefetch knobの
小さいCPU screenは可能だが、現在のGPU/CPU route変更より優先する証拠はない。

## 次のSG diagnostic controller

Solの`iq3xxs-subgroup-controller-v1`はsource-onlyのまま保持し、次の検証にはv2を使う。
v2はpin済みOwnerとqualified parent-only ptracer shim/StackProbeを再利用する。
nativeはMainのdirect child、GDBは別Owner/session/rootで一度だけ。URの新しい成功
kernel create/enqueueだけをprogressとし、poll/warningの反復でquietを隠さない。
60秒quiet、deadline前のreserveを置き、captureしたarmはexit0でも比較PASSにしない。
8armのimmutable environment、全配列、actual route、fault/closure、full2GiB予算は維持。

Mainはsourceをレビューしたが、controller parser/ownershipとbuild/IR gateは未実行である。
ビルド成功だけからIR review完了やSG16/32の実device admissionを作らない。
初回CPU qualification・実lowering確認・owned GPU correctnessを実行するまで性能に
使わない。これらは1000 prefill/70 decodeという元目標の達成証拠ではない。
