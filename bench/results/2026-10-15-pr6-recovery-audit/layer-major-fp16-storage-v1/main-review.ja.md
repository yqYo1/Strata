# Main integration review: FP16 storage v1

2026-10-15。元のSol候補 `source.diff` SHA256
`f413ef2031997b73f41e7cced5ab24d078b8fee033d31ce6cab69552170fce2c` を
7つの実sourceと4つの依存のbase hash照合後、incremental applyした。
候補の未測定レポートは変更していない。

Mainは追加で次を修正した。

- `layer_major.inc`のtraceでint64_t定数NEに`%d`を渡していた不一致を修正。
  `layer-major-dispatch-build-v1.json`はこのコンパイル失敗を保存し、状態を上書きしない。
- postopsテストのreadback/uploadでは、ローカルhost vectorの破棄より前に
  完了不明の転送でfail-stopする共通copy helperを追加。Allocation destructorだけでは
  ローカルreadback vectorの寿命を保証できない。

`layer-major-storage-build-v1.json`でstrata、CPU plan test、GPU postops testをbuild済み。
CPU plan testは262,144行のFP32 10 GiB / FP16 5 GiB、staging込みのadmission、
prefix/host相対offset、odd tail、checked arithmeticとlease終了を検証しPASS。
これは実物の256Kモデルstateを保持・復元した検証ではない。

型の変更は残差storage所有者と入力pointerだけに閉じ、FP32既定ではcast kernelも
staging allocationも増やさない。FP16は47層間で一度pack、consumerでexact unpack。
既存to_f16は非飽和RNEで、overflowはInfとなる。host tailは同一in-order queueで
stageへH2D→unpack→model→pack→D2Hし、全ownerは既存checked leaseより長命。

Mainの直列GPU検証は `check_layer_major_storage.py`、結果は終了時の
`layer-major-storage-small-v1.json`を参照する。動作中のprefixを完了扱いしない。
最初のFP16合成入力512 batched rowsでは最終残差relative L2約0.0973、
first-logit KL(reference||candidate)約0.0270 natsだった。事後castの0.000211とは
比較対象・丸め回数が異なる。層をまたいだ差の増幅やroute変化を含み得るが原因は未特定。
これだけで実用品質の劣化率や採用を断定しない。自然な長文の複数位置logits/NLL、
タスク結果、clean performanceを比較してから判断する。

現時点でdefaultはFP32、dtype selectorはopt-in、quality/performance/RESTORE/
physical256Kは未認定。特に「CPU cast誤差0.021%だから品質劣化0.021%」とは主張しない。
