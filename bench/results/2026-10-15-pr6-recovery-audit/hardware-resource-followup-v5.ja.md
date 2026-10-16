# 資源別の判定とPCIe階層の追補

`hardware-resource-followup-v4` の観測値・成否は変更しない。
この追補は端点だけを読んだPCIe情報を補い、既存の資源比較で判断できる範囲を整理する。
生の個別測定値は各receipt、接続階層は `own-pcie-hierarchy-v1.json` に保存した。

| 資源 | 実測からの現在の判断 | 次の比較で切り分けること |
| --- | --- | --- |
| PCIeを含むhost/device経路 | 新しい320MiBコピーはH2D約4.35、D2H約3.89GB/s。古い長入力診断のAttention区間では全copyの占有9.11%。転送を無視はできないが、転送だけが支配的という証拠ではない。 | 同じ配置・サイズ・環境で容量を確認し、実モデルの要求bytes・service・依存・重なりへ対応させる。古い6.02/5.54との差の原因は未確定。 |
| CPU/RAM | expert実サイズのRAMコピーは3coreで48層換算3.903秒。既存モデルのsource-copy4.852秒との差は約0.95秒。発行threadのCPU時間の70.8%は明示GPU待ち。 | CPU使用率だけで律速とせず、実staging、未分類のhost処理、発行回数とGPU依存を区別する。RAMの物理trafficは未取得。 |
| GPU演算・VRAM/cache・SLM・同期 | 古いAttention区間のcompute占有77.31%。QSA60回のservice計33.963秒。これらはGPU内を優先調査する根拠であり、演算器の飽和測定ではない。 | 同じ数値仕事のまま再利用、転送bytes、作業分割、発行を個別に変える。一般GEMMやD2Dコピーの性能をAttentionの利用率へ流用しない。 |

SLMの使用量を減らす案は既に試している。独立したhead8+4案は、実local引数合計を
44496から31088 bytesへ減らしたが、query128/256のcomponentでは約2.85/3.16倍遅かった。
両armの全出力一致、FP64参照、guard等は通過した。KV読込等も変わるので、これだけで
SLM容量やVRAM帯域が主因とは言えない。この案を未実施の新候補として再提案しない。
個別値と不採用判断は `own-heads8-isolated-bench-v1.json` と
`own-heads8-isolated-decision-v1.ja.md` にある。

GPUの物理traffic・演算活動・stall/occupancyの数値はまだない。
このため「演算性能が不足」「VRAM帯域が不足」のどちらかを確定したとは報告しない。
収集に失敗したOA経路の追加試行は止め、既存試験と重複しない制御比較を選ぶ。
Orca Luna R914の測定案はnative Codex Luna-max R916へレビューを委任した。
Mainは調査本文を結果到着だけで読み込まず、必要な範囲だけを利用する。

## PCIe接続の実際の階層

Mainが同じbootでGPUのsysfs実体から全親bridgeを読み、設定変更なしで取得した。

| PCIアドレス | vendor | current speed / width |
| --- | --- | --- |
| 05:00.0 | Intel | 2.5 GT/s / x1 |
| 04:01.0 | Intel | 2.5 GT/s / x1 |
| 03:00.0 | Intel | 16.0 GT/s / x4 |
| 02:02.0 | AMD | 16.0 GT/s / x4 |
| 01:00.0 | AMD | 16.0 GT/s / x4 |
| 00:01.2 | AMD | 16.0 GT/s / x4 |

上流側の対応するlink表示はGen4 x4で一致する。16GT/s・4lane・128/130符号化を仮定した
片方向の転送速度はtransaction等のoverheadを引く前で約7.877GB/sと計算できる。
これはこのsnapshotの上流linkからの理論計算で、実測payload上限や利用率ではない。
端点側の2.5GT/s x1表示もそのまま保存した。内部bridgeの実装理由までは確定していない。
この後刻のidle snapshotだけから、過去cohort実行中の全link状態も断定しない。

PCI階層全体を確認する方法は
[Linux PCI Support Library](https://www.kernel.org/doc/html/v6.16/driver-api/pci/pci.html#c.pcie_bandwidth_available)
の説明とも整合する。ここではkernel helperを呼んだのではなく、各nodeのsysfs値を取得した。
今後も端点1個の表示だけで物理上限を決めず、実行中の値・実測bytes/timeと照合する。

## PR#6の別系統の進捗

CPU専用target28個のC++ compileに限定した `-fno-sycl` 追加後、生成111commandの
差分確認は通った。build v2は55/113段階で旧DPCTの64bit shuffleとLLVM22の型解決エラーにより
exit1。所有プロセスは終了し、GPUもlegacy binaryも未実行。GPUハングではない。
原本vendorや歴史的198project filesを変えずにprivate headerで対応する作業をSolへ委任した。
モデルを使わないprovider試験の提案はMainが全patchをレビューし、opt-in targetへ追加した。
これもまだbuild/native未実施である。600再現、モデル品質・state・physical262144・RESTOREの
適格化は未完了。今回、新しい推論最適化の採用や速度向上の主張はない。
