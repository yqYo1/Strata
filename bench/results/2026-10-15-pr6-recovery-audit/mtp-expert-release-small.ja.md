# MTP expert重みの一時解放: buildと小規模GPU gate

2026-10-15 JST。Arc B570 10 GiB / Ryzen 5600X / RAM 128 GiB、実root SYCL engine、native IQ3_S。
675 MiBのMTP expert重みを長いlayer-major prefillの間だけVRAMから解放し、終了後にRAMから復元する。
`STRATA_PREFILL_RELEASE_MTP_EXPERTS=1`の明示指定時だけ動く。標準設定は変更しない。
decode用の128-slot expert cacheを解放する別案は未適用である。

## 実装と所有権

元のexpertファイル707,788,800 Bを初回だけprivate vectorへ読み込み、SHA256を計算する。
今回の元データのSHA256は`09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f`。
675 MiBと小さい余裕を8 GiB host headroomを残して入場判定する。復元ごとのfile read/SHA scanや
VRAM→RAM保存は行わない。復元は同じimmutable RAM ownerから64 MiB以下のviewを順にH2Dし、
各copyのchecked completion後、全量成功したallocationだけをpublishする。

退役時はMTP queueをdrainし、normal/coupledのround/step graph四familyをchecked destroyして
expert allocationをfreeする。KV-only prompt graph、dense/state/KVとVerifier final_R borrowは残す。
Prefillの一時residual/cache ownerが終了した後に復元し、decode graphは新しいexpert pointerで
必要時に再作成する。return/cancel/例外経路も復元を経由し、不明なcompletionではownersを残して停止する。
異常経路はsource-reviewedであり、GPU障害注入で確認したものではない。

## 検証

`mtp-expert-release-build-v1.json`はcompileとSHA256 CPU known-answer vectorsを通過。
GPU binary SHA256は`44cafcc2ed2206b7f49fd79b9e0405a6eb9d09c6bf045ecda83085598f5800da`。
実際の生成器/MTPの`-fsycl-id-queries-range=size_t`をcompile_commandsで確認している。

`check_mtp_expert_release.py small`、receipt `mtp-expert-release-small-v1.json`。
513入力ID/512 batched、chunk257+255、context262144/KV8 resident32768/cache128perlayer/pool5+host/
MTP4 splitOFF、生成8。既存の同じdtypeのselected-head gateをcontrolにした。
UR/Level Zero parameter validation・warning出力とprogress traceを有効化し、copy-offloadと
DirectSubmissionは無効、compute/copyは同じqueue。正常終了/owned reap/session empty、boot同一、
新規kernel GPU fault 0、devcoredumpなし。小入力は寿命検証用で性能比較には使わない。

| 保存dtype | detach ms | RAM→VRAM restore ms | detach前後のfree増分 B | restore前後のfree減分 B | 再capture累積 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 0.794 | 345.806 | 707,792,896 | 710,025,216 | 1,338.420 |
| FP16 | 1.906 | 166.618 | 707,792,896 | 710,025,216 | 1,375.886 |

free値は実device free-memory queryの差であり、allocator/graph等の全ての差をMTP payloadへ
帰属させない。payloadは707,788,800 B。両armとも約675 MiBの増分を観測した。
初回round captureはFP32 1,328.085 ms / FP16 1,363.097 ms、その他のstep/round captureは
1.540–7.790 ms。初回cold kernel/graph準備を含むため、1.3秒を繰り返し復元の追加費用とはしない。
restore単体の2標本も帯域上限や定常中央値の測定ではない。

両armでcontrolとの最初の全248,320 logits、9位置の全vocab selected logits、metadata、
生成8IDがbitwise一致。FP32とFP16相互の一致を条件にはしていない。
normal round/step graphの成功をログで確認し、復元元はoriginal=1の初回hash付きRAMと
original=0の再uploadだけ。小入力のresidualは両armで全512rowsがVRAMに収まった。

## 限界と次のgate

この結果は初回standalone成功経路である。長い入力での実capacity、既存decode graphを持つ
繰り返しserve、coupled、短文bypass、物理262144入力境界、RESTORE、独立task品質と正式性能は
まだ証明しない。次は同じ34,603入力/34,602 batchedのFP32/FP16各armで19位置と末尾logitsを
比較し、実際のresidual GPU/host/staging量を記録する。
追加の速度採否はthinking有効の`bench.sh`長文/agent session/長生成の別clean反復で行う。

大きなraw selected logitsは現時点の同dtype比較と回帰用consumerだけに残し、比較終了とcompact
evidenceの検証後にmanifest付きで退役する。成功full API/compiler/model traceを永久保存しない。
