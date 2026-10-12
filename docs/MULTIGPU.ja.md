<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# 複数の GPU で動かす（層の分割）

1 台の PC にある 2 枚以上の GPU に、モデルの層を分けて載せます。
最初の GPU が前の層を、次の GPU がその続きの層を受け持ちます。
各 GPU は自分の層の密な重み（PLE のテンソルはどの GPU にも）、セッション（GDN の再帰状態、QSA の K/V）、エキスパートのキャッシュを持ちます。
VRAM が増えた分だけ多くのエキスパートを GPU に置けるので、CPU が計算するエキスパートが減ります。
メーカーの違う GPU（Intel・AMD・NVIDIA。NVIDIA を含むときは contrib-llvm のビルド）も組み合わせられます。

GPU の間の受け渡しは、検証のウィンドウごとに 1 回、CPU のメモリを通します（1 トークンあたり 12,804 個の float）。
prefill は、GPU 1 枚のときと同じく、各 GPU のエキスパートのキャッシュから領域を借ります（塊の大きさは全 GPU で共通）。

明示的に有効にする機能で、オプションがなければ何も変わりません（upstream の層の分割の移植）。

## 有効にする

setup で、使う GPU を `--gpus` に並べます（番号は `--check` の表のもの、最初の GPU が前の層を受け持ちます）。

```sh
./setup.sh --gpus 1,2                    # インストールのとき、または起動のとき（そのモデルの設定に残ります）
./setup.sh --gpus 1,2 --layer-split 36   # 分割点を自分で決めるとき（既定は auto）
./setup.sh --gpu 1                       # この起動だけ GPU 1 枚で（保存した分割は残ります）
```

setup は最初の GPU を `"gpu_pci"` に、後の GPU を `"split_pci"` に PCI アドレスで書き、サーバーがエンジンに
`--layer-split` と `--split-device` を渡します。
エンジンを直接動かすときのオプションは次のとおりです。

| オプション | 意味 |
| --- | --- |
| `--layer-split auto` | 分割点をこの PC で測った値から決めます（[下](#分割点の自動決定)）。GPU が 1 枚しか見えなければ分割しません。 |
| `--layer-split K[,K2..]` | 層 K から次の GPU に渡します（K は 2 以上で、増えていく順）。分割点ごとに GPU が 1 枚要ります。 |
| `--split-device D[,D2..]` | 分割点ごとに使う GPU。エンジンの番号（1 から、[GPU の番号](#gpu-の番号)）か PCI アドレスで書きます。省くと、CPU 内蔵でない GPU を番号の順に使います。 |
| `--batch-groups G` | `--batch N` と一緒に使います。スロットを G 個の組に分け、GPU ごとに別の組を同時に計算します（[下](#まとめて-decode-するとき)）。 |
| `--batch-cpu-split N0,N1` | 実験的、既定は無効です（[下](#cpu-のエキスパートを段ごとに分ける実験的)）。 |

必要なもの:

- `--serve`（サーバーから起動するときは付いています）と `--expert-profile`。

会話の退避（`--conversation-cache-mib`、`--conversation-save`）は、各 GPU のセッションを写して戻します。
保存したファイルは同じ分割点のエンジンだけが読みます。`--layer-split auto` は測った値で分割点が変わることがあり、そのときは前のファイルを読みません。

## GPU の番号

エンジンが使える GPU に、0 から番号を付けます。
0 は setup が `STRATA_GPU_PCI` で選んだ GPU です（エンジンを直接動かすときは `--gpu <PCI アドレス>` でも同じ、upstream #852）。
残りは、CPU 内蔵でない GPU、メモリの多い GPU、計算ユニットの多い GPU の順です。
CPU 内蔵の GPU は、`--split-device`（setup では `--gpus`）で指定したときだけ使います。

起動のログは、GPU を `CUDA0`、`CUDA1` のようにこの番号で示します（upstream と同じ書き方で、メーカーとは関係ありません）。

## 分割点の自動決定

`--layer-split auto` は、モデルを読み込む前に次の値を測って、分割点を決めます。

- 各 GPU が VRAM を読む速さと、空いている VRAM。
- CPU が RAM を読む速さ（CPU が計算するエキスパートの速さを決めます）。
- 各 GPU が持つ重みとセッション（どちらも自分の層だけ）、出力ヘッド（最後の GPU）、prefill に残す分。

ここから各 GPU に入るエキスパートの数を出し、`--expert-profile` のよく使われる順にキャッシュを埋めたときの、
1 ウィンドウの時間を分割点ごとに見積もって、最も短いものを選びます。
選んだ分割点はログに出ます（`layer split auto: K=...`）。そのあとは、その分割点を指定したときと同じです。

見積もりの前提は、環境変数で変えられます。

| 変数 | 既定 | 意味 |
| --- | --- | --- |
| `STRATA_SPLIT_ROWS` | 3 | 1 ウィンドウで計算するトークンの数（MTP の下書きを含む平均） |
| `STRATA_SPLIT_MASS_EXP` | 0.8 | よく使われる順で r 番目のエキスパートが使われる割合を (r+1)^-a としたときの a |
| `STRATA_SPLIT_CPU_GBPS` | 測った値 | CPU が RAM を読む速さ（GB/s） |

`--vram-reserve-later-mib N` は、2 枚目以降の GPU に残す VRAM です（既定は `--vram-reserve-mib` と同じ、upstream 5c4105c0）。
モニターをつないだ GPU は、つないでいない GPU より多くの余裕が要ります。
モニターが最後の GPU にあるなら、`--vram-reserve-mib 300 --vram-reserve-later-mib 1800` で、最初の GPU のキャッシュにその VRAM を回せます。

## まとめて decode するとき

`--batch N`（[BATCHING](BATCHING.ja.md)）と層の分割を一緒に使うと、各 GPU は自分の層のスロットのセッションを持ちます。
`--batch-groups G` なしでは、1 つのウィンドウが GPU を順に通るので、ある GPU が計算している間ほかの GPU は待ちます。
`--batch-groups G` は N 個のスロットを G 個の組に分け、GPU 0 が組 2 を計算する間に GPU 1 が組 1 を計算します。
G は N を割り切る数です。
1 組のスロットが少ないほど、重みを 1 回読んで計算できるトークンが減るので、組を細かくしすぎると遅くなります（下の測定では G=2 が最も速い）。

### CPU のエキスパートを段ごとに分ける（実験的）

`--batch-cpu-split N0,N1` は、`--batch-groups` で 2 枚の GPU を使うとき、CPU のエキスパートの計算を
N0 と N1 のスレッドの 2 つの組に分け、2 つの段の CPU の計算を同時に進めます。
RAM を読む速さが足りない PC では遅くなります（下の測定）。RAM の速い PC 向けの実験的な機能で、既定は無効です。

## 1 つの会話で 2 枚とも働かせる（`--pipeline-windows`、opt-in）

分割では、検証の窓を GPU が交代で受け持ちます。
最初の GPU が自分の層を計算して引き渡したあとは、最後の GPU が残りの層と head と下書きを計算する間、待っています。
`--pipeline-windows 2` では、最後の GPU がこの窓を検証している間に、最初の GPU が**次の**窓を始めます（upstream bfa532e9）。
次の窓は推測です。この窓が全部受け入れられ、そのボーナストークンが下書きの層の予想どおりだと見込みます（下書きの層は、検証中の窓の下書きをなぞって先へ進めます）。
推測が当たれば次の窓は半分済んでいます。
外れれば、最初の GPU は状態を戻し（窓の実行中に取った再帰の状態のコピー）、次の窓を本当のトークンから作り直します。
推測の窓は、下書きの層の見積もりで残りそうなときだけ始めます。
`--pipeline-windows 1` は、検証の窓を通る短い prefill（`--short-read`）だけを重ねます。

GPU 2 枚、ちょうど 2 段、`--serve` のときに使えます。

- **費用**: 各 GPU に 2 つ目の検証の窓。各 GPU のエキスパートのキャッシュから 160 MiB を外し、`2` では最初の GPU に再帰の状態のコピーを 2 つ置きます（その GPU の GDN の層 1 つあたり約 3 MiB）。
- **同じ文章**: 最後の GPU は検証の済んだ窓だけを計算し、窓のどの行もほかの窓と同じ計算をするので、出るトークンは直列のときと同じです。
  `STRATA_IQ_MT_MIN=1 --pcie-frac 0 --adapt-every 0` で、エキスパートのキャッシュが同じなら、貪欲な出力はビット単位で一致します（下の測定）。
  パイプラインは自分の VRAM をキャッシュから外すので、このフラグのない実行と比べるといくつかのエキスパートが GPU から CPU に移り、丸めが変わって僅差の選択が入れ替わることがあります。
- **使えない構成**（ログに理由を 1 行出して無効にします）: `--batch` のスロット、補助のキャッシュ（`--expert-cache-device1..3`）、3 段以上の分割や 1 枚の GPU 上の分割（`--split-device 0`）、下書きの層なし。
  繰り返しのペナルティ（`penalty_last_n`）、組み合わせた下書きのサンプリング、対数確率（`logprobs`）を求める要求と `--lookup-chain` は、直列で decode します。
- **カードの上では、ホストを待っている窓の後ろに何も積みません**。B70 では、その窓の実行中に同じカードへ出したグラフやコピーが戻らず、エンジンが止まりました。
  そのため、prefill では窓の commit をその窓の完了後に積みます（上流は窓、commit、次の窓を続けて積みます）。
  適応する段は、どちらのカードにも窓がない判定のときに動かします（上流は窓の横の別スレッドで動かします）。
  `--adapt-async 1` では、各カードのコピーを、そのカードに窓がないときに decode のループが積みます（upstream 13eee452）。
- `STRATA_PIPELINE_*` の調整・試験の変数（THETA、FORCE_MISS、SWITCH、LOG、TRACE など）は、`STRATA_PIPELINE_DEBUG=1` のときだけ読みます（upstream 2bfac510）。

## 測定

Arc Pro B70（32 GB）と RTX 4070（12 GB、PCIe x4）、Core i7-14700、128 トークンの decode:

| モデル | B70 だけ | B70 + 4070（`--layer-split auto`） |
| --- | --- | --- |
| IQ3_S | 62.4 tok/s | 68.7〜69.6 tok/s（K=36） |
| UD-Q4_K_XL | 20.8 tok/s | 38.3〜39.0 tok/s（K=35） |
| IQ2_XS | 73.5 tok/s | 84.7 tok/s（K=45） |

IQ2_XS では、手で K=38 にすると auto より約 3% 速くなりました。

約 5,000 トークンのプロンプト（同じプロセスで 2 つ目のプロンプト）:

| モデル | B70 だけ | B70 + 4070 |
| --- | --- | --- |
| IQ3_S | 4,884 ms | 5,110 ms（K=36） |
| IQ2_XS | 4,439 ms | 4,208 ms（K=45）、3,661 ms（K=42） |

8 つの会話をまとめて decode（IQ3_S、`--batch 8`、`--layer-split auto`、最初のバッチのトークンからの速さ）:

| 構成 | 速さ |
| --- | --- |
| B70 だけ | 55.2 tok/s |
| 分割、1 組 | 73.1 tok/s |
| 分割、`--batch-groups 2` | 77.1〜79.0 tok/s |
| 分割、`--batch-groups 4` | 64.9 tok/s |
| 分割、`--batch-groups 2 --batch-cpu-split 10,9` | 71.8 tok/s |

`--pipeline-windows 2`（IQ2_XS、コンテキスト 32K、約 4,900 トークンのプロンプトのあとに 256 トークンと、短い会話 3 つ、貪欲、2 回ずつ交互）:

| 分割 | 直列 | `--pipeline-windows 2` |
| --- | --- | --- |
| K=24、長いプロンプトのあと | 115.6、119.8 tok/s | 145.8、125.8 tok/s |
| K=24、短い会話 | 87.9、88.2 tok/s | 90.6、87.4 tok/s |
| K=47（auto）、長いプロンプトのあと | 105.9 tok/s | 87.7 tok/s |
| K=47（auto）、短い会話 | 81.3 tok/s | 59.3 tok/s |

4070 が 1 層しか持たない auto の分割（K=47）では、重ねる計算がなく遅くなります。
K=24、`--resident-experts`（`--adapt-async` は既定で有効）では、長いプロンプトのあと 111.4 → 117.7 tok/s、短い会話 83.2 → 70.7 tok/s でした（1 回ずつ）。
K=24、`STRATA_IQ_MT_MIN=1 --pcie-frac 0 --adapt-every 0`、最初の GPU のキャッシュを `--expert-cache 10000`、直列では最後の GPU の予約を `--vram-reserve-later-mib 860` にそろえると、8 つの答え（思考あり・なし）がすべて一致しました。

## いまの限界

- 1 台の PC の中だけです。ネットワークの先の GPU は使えません。

## 独立した replica（Linux SYCL）

`replicas` は明示的に有効にするサーバ機能です。各 GPU group が独立した engine、モデルの RAM、
KV、会話キャッシュを持ちます。単一 engine の層分割とは別で、既定の動作は変わりません。
setup の `--gpus` で選んだ GPU を `--replicas N` で等分します。保存される識別子は PCI アドレスです。
直接 config に書く場合は、例えば `"replicas": [{"devices": ["05:00.0"]}, {"devices": ["06:00.0"]}]`
とします。同じ GPU を複数 group に入れることはできません。各 group 内で複数 GPU を使う場合だけ層分割を行います。

`--replica-ram-mib`（config: `replica_ram_mib`）と `--replica-vram-mib`（`replica_vram_mib`）の
正の予約量が必要です。RAM 予約は各 engine のモデル、KV、cache、セッションを含むピークを見積もって指定します。
モデルファイル容量の保守的な下限以上でなければ受け付けません。開始・reload・故障後の復旧前に、
ホストと memory cgroup の全 ancestor の空き RAM、各 GPU の空き VRAM を確認します。
予約量は native allocator の上限を変更せず、native の割当検査を置き換えません。
標準の `/sys/fs/cgroup` 配置を読めない場合、情報不足のまま開始せず error を返します。

Linux では親の許可 CPU mask の physical core を等分し、SMT sibling を一緒にします。
独立した子 launcher が自分にだけ affinity を設定し、同じ PID で指定 engine を exec します。
threaded server の `preexec_fn`、親や他のプロセスの affinity 変更は使いません。
必要な core/topology がないときは error です。明示的な `"replica_cpus": false` で分割を無効にできます。
launcher の Python executable、`serve/cpu_affinity.py`、config の実行可能ファイルと実際の engine identity も検証対象です。

要求は負荷と bounded な会話 prefix hint で割り振ります。engine 自身の prefix 検査が最終判断です。
Monitor `/metrics` は replica ごとの状態・要求数・cache log と合計を示します。log/profile/会話保存先は
replica ごとに分かれます。単一ファイルの `/slots/0` SAVE/RESTORE と動的 VRAM 操作は replica front では提供しません。
初回は逐次起動し、失敗したら作成済みの子をすべて閉じます。稼働中の一台が終了した場合は他の台が応答し、
予約量を再確認して復旧します。shutdown は復旧 thread を閉じてから所有 engine を終了します。
単一 engine 用の native calibration は、別の replica 無効 config で行ってください。
この機能から throughput や速度向上を約束するものではありません。
