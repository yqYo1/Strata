# CPU GU展開再利用の部品検証

Ryzen 5 5600X、clang/znver3、単一CPU affinity=0、通常ページ。
`cpu-predecode-probe-v1`のソースをMainで確認し、提案のレシピをそのまま実行した。
製品への組み込みはしていない。元の提案manifestの「未実行」は提案作成時の記録であり、
Mainの実行結果は別の`cpu-predecode-probe-v1.json`に保存する。

## 検証範囲

対象はIQ3_XXS(type18)、IQ2_S(type22)、IQ3_S(type21)のGU、H=2560、FF=640。
元のDとsubscale、符号付きint8 codebook値を持ち、モデル再量子化をしない。
activation=-128のPSIGN wrap、maddubs飽和、形式ごとのFMA・水平和の順序を保つ。
exactはGGMLと既存single-exact、customは既存multi-token実装と比較する。
raw gate/upとSwiGLU、元packed全バイトへの独立復元、guard pageを検査した。

通常・ASan/UBSanの各ビルドで6,276ケース、149,400値比較、1,792 field blocksが通った。
nt1〜4、空/部分範囲、実寸全行、alignment、±0、subnormal、Inf/NaNを含む。
有限値はsigned zero込みbitwise、NaNはclass、Infはsign/classが判定条件。
GGMLの既存static archive内部はsanitizerの対象外で、この限界は結果にも記録した。
全所有プロセスは正常終了・回収済み。GPUは使用していない。

## 最初の計測

1/16/64個の異なるexpertを使い、各条件432サンプル、合計1,296サンプル。
同じ仕事量・入力に対し、比較順とABBAの向きを固定seedで入れ替えた。
CPU service、wall、TSCを別に測り、TSCをcore cycleとは呼ばない。
各条件の全個別値と12ブロックの比を元結果に残す。

64expertではpacked/expandedの両方が32MiB LLCを超える。
expanded/GGMLのpaired thread-CPU時間比の中央値は18=0.6260、22=0.8030、21=0.5298。
既存single-exactに対しては18=0.7001、22=0.7439、21=0.5701。
ただし周波数設定はpowersaveで、paired比の幅も大きい。
例えばtype22の対GGML比は0.5904〜1.0561。初回だけで安定した改善と判断しない。
同一coreの独立プロセスで64expert条件を追加反復した。

`check_cpu_predecode_repeat_v1.py`による追加3process（各432、計1,296サンプル）も
正常終了・回収済み。対GGMLのpaired thread-CPU中央値比は次の通り。

| type | 最初のprocess | 追加0 | 追加1 | 追加2 |
| --- | ---: | ---: | ---: | ---: |
| 18 IQ3_XXS | 0.6260 | 0.6202 | 0.5878 | 0.6007 |
| 22 IQ2_S | 0.8030 | 0.8054 | 0.7199 | 0.6884 |
| 21 IQ3_S | 0.5298 | 0.4875 | 0.5115 | 0.4755 |

wallの中央値比も同様の改善方向で、既存single-exactに対しても各processで改善した。
全個別値は`cpu-predecode-repeat-v1.json`、各ABBAブロックの両clock比は
`cpu-predecode-repeat-summary-v1.json`に保存した。初回を含む独立4processで
部品計算の改善が再現したため、次は実モデルの再訪観測へ進む。
nt2〜4は正しさだけの合格で、性能比較はまだしていない。

この結果は合成packed重みの部品結果である。
実モデルのexpert分布、再訪率、複数workerでの帯域競合、展開代の回収は未測定。
expandedはmetadata込みで2.38〜3.24倍のpayloadを読み、無条件の全モデル展開は行わない。
実際のCPU GU再訪を上限付きのobserverで測り、利益を得られる場合だけ容量付きcacheを検討する。
モデル全体のdecode速度、品質、物理256K、RESTOREをこの結果で認定しない。
