# 次の実計測候補とprefill容量の確認

2026-10-15。今回のprogressはfull OFFの正常終了、nonce回復、比較tool六＋十一test、wave censusの
normal/ASan/UBSanとvalidator四test、PCIe helperのnormal/ASan/UBSan、および実client最小変更である。
元host-v1のfile-cap失敗は別のFAILとして保存した。結果とsourceはdfb129f7までpush済み。
現在のsource/receiptについてはfull-off-closed-host-qualification.ja.mdを参照する。

## 実際に開始したIQ ON

run_matched_iq_on_v2.pyを起動し、exec10041がrunningであることを直接pollした。
新rootはB/main-selective-iq-clean-bench-full-1-root-v2、bench-full-1は
2026-10-15T06:47:30.150413+00:00開始、leader323620、native323699/start39561505。
nativeは同じb1cc9cb45554e89c060690420dc77b16fab8f512c55992c5faf367551ad3c8c0である。
元nonce1792036964、同一engine argsを使い、IQ選択だけ1とした。実nativeの所有identityを確認して、
必要な八つのruntime/selector環境変数だけを読み、IQ1、DirectSubmission0、same-queue1などを確認した。
その他の環境変数は出力していない。root src/include/binary/profileを変えず、計測中のtest/build/GPU/
cleanupを重ねない。短い標準phaseが進んでいるが、32K以上の性能比較や実body一致はまだ未確定。
新最終receiptはselective-iq-clean-bench-full-1-v2.jsonへownerが終端時に保存する。

## R655のsource監査を実測と照合

LunaはMTP expert releaseが既にrootへ入っており、decode cache退去案の四source pinは古く、
checked-prefill-retireはprompt後の退去なので今回のprefillの空き容量を増やさない、と確認した。
Mainはその区別を採用する。後者の古い1662831872 B accountingを、今すぐ使えるVRAMと数えない。

ただし既存MTP releaseを0/1で再測定するという最小案は、今の次armには採用しない。閉じたfull OFF
native logのSHAは73eb6a97c02da078f9c1898ddc9e726230c132ede5c3d055a85d71c343667e7aで、
元receiptのnative_log_sha256と一致する。最後の実長文では、before-detach1715826688 Bから
after-detach2423623680 Bへ、実free queryが707788800 B増えていた。release1はすでに使用済み。
layer-major行はn33287/C4096/FP32、GPU4096 rows=167772160 B、host1195663360 B、
one-layer cache1363148800 Bを示す。これは実測行であり、物理262144入力の結果ではない。

現plannerのone-layer cache+512MiB headroomだけで1900019712 Bとなり、この例のrelease前freeより
大きい。既存releaseを外すだけでは同じadmissionを通せない可能性がある。R_GPU4096を固定したまま
freeを増やしても、指定prefixが自動拡大するわけではない。

容量上の目安として、このafter-detach freeからcache/headroomを引くと523603968 Bである。
4096行単位、一行40960 Bという今回の形状なら、他の状態が完全に同じ場合の自動prefix上限は
12288行となる。さらにlegacy128 decode-cacheの281651200 requested Bを全て実freeできると
仮定しても、同じ計算では16384行に留まる。これは条件付き算術であり、その退去/auto実行の証拠や
速度予測ではない。固定4096以外の数値資格は未完で、旧auto24576 FAILもそのまま残す。
追加のheadroomには、より大きな未使用ownerまたは実行中のscratch寿命/aliasを調べる必要がある。

R657/R658を新たにLunaへscopeを分けて依頼した。前者はMTP expert以外の実readerとdecode専用owner、
後者は現在のPrefillの実行中allocation/lifetimeを調べる。prompt callback、KV/recurrence、window_R、
prepared waveの生存を無視したfreeは候補に含めない。新code/controllerは要求せず、具体的な容量と
readerの根拠を先に求める。既存report/indexを共有し、FINALまで介入しない。

## R656と次のdecode診断

decode-host-phase-observer-v2の五つのbase hashは現在のrootと一致した。ただし未適用・未buildで、
現b1からobserver結果は得られない。Mainもsource監査の範囲を確認した。request ownerとborrowの
正常detach/逆順destructionを持ち、host issuerの排他的leafを計る。GPU実行時間、worker CPU秒、
親RunTotalを足してidleを求めるものではない。prose4.6対他の速度差の原因はまだ特定していない。
ON完了後の候補は、この既存案のCPU contract/borrow fixtureとisolated SYCL buildを経た、
32K以上入力・128出力のbounded joined/host診断である。clean速度は別の非計装processで測る。

## PCIe sysfsの読み取り値

GPU APIやDMAを呼ばず、実PCI親子pathのsysfs属性だけを読んだ。以下は表示値である。

| PCI node | vendor:device | current GT/s × width | max GT/s × width |
| --- | --- | --- | --- |
| 0000:00:01.2 | 1022:1483 | 16.0 × 4 | 16.0 × 8 |
| 0000:01:00.0 | 1022:57ad | 16.0 × 4 | 16.0 × 8 |
| 0000:02:02.0 | 1022:57a3 | 16.0 × 4 | 16.0 × 4 |
| 0000:03:00.0 | 8086:e2ff | 16.0 × 4 | 16.0 × 8 |
| 0000:04:01.0 | 8086:e2f0 | 2.5 × 1 | 2.5 × 1 |
| 0000:05:00.0 | 8086:e20c | 2.5 × 1 | 2.5 × 1 |

endpoint単独の表示値からDMA帯域上限を決めない。現在stackの既存転送実測との整合と、途中nodeの
報告仕様を確認するまでは、これを性能低下原因や復帰不良の証拠にしない。deviceはxeにbindされて
いる。変更、再bind、reset、service操作、権限昇格や新転送測定は一切行っていない。

現ONの終了・実入力照合・各phase採否判断が次の実作業である。性能1000 PP/70 TG、PR6回復、
task品質、物理262144/RESTORE、upstream/default反映は未達のまま維持する。
