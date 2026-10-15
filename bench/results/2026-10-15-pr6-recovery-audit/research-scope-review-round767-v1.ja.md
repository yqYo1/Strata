# R761–766 調査スコープレビュー

R764とR765は現在の判断に直結する。R764はFP16/BF16残差保存、品質評価、GPU常駐容量の判断材料で、同一モデル系統のFP8報告を動機付けに使う一方、StrataでのFP16対BF16の実測はない。16文脈の提案も品質認定ではなく、まずMainが実測するためのスクリーニング設計である。R765はhost copy呼び出しを調べる根拠になる。Intel LLVM v7.1.1の不変ソースには、特定のbatched-queue上限で非blocking copyがqueue drainを踏む経路がある。ただし、現在のバイナリや実行時queueがその経路を使った証拠はない。Mainのレビュー記録では両原文とも既読なので、読み直す必要はない。

R761はT=8192のstable groupingを5段階で具体化し、追加scratchと正しさ条件を算出した設計候補。host側のexpert数・順序依存は残り、測定されたgrouping時間は先行waitを含まない。新しい境界計測でこの依存が効くと分かった場合だけ原文を開けばよい。R762はmapped host-USM kernel accessと明示queue copyの別々の比較案で、R765のordinary pointer対malloc_hostという問いと混同しない。R763はデータベース、疎行列、stencilの5機構を集めた広めの知識報告で、反例と適用限界もある。Mainは64 MiBのdescriptor arenaだけではrouted GU/Hh全体を表せず、約500 MiB/チャンクの行データを別計上するよう補正した。R766はGRF、SLM、再計算、DPASの資源トレードオフをIntel Max 1550とAMD MI200の反例で説明する。B570での資源割当や性能の証拠はないため、特定QSA候補のJIT情報が揃うまでは知識保管とする。

新規性と深さは全体に十分で、明白な水増しはない。R761とR762は同じ境界計測を前提にするが、安定groupingとメモリアクセス方法という別の問いを扱う。R765は実際のcopy呼び出し内部に焦点を移しており、R762の代わりにはならない。R763はスコープの広さが上限に近いが、異なる方式と失敗例を比較している。R764は実験設計が広いものの、数値範囲、NLL、タスク結果、性能を切り分け、基準を捏造していない。R766もコンパイラ資源の値と速度を同一視していない。

次の研究は、実測後の判断に合わせて開始する。第一候補は、host `ne/order` がまだ全体時間に効くと確認された後、oneMKL grouped/batched、固定bucket、device descriptor consumerを比較すること。第二候補は、8192行chunkと32768行GPU常駐時に、逐次実行、resident prefix、二重bufferのどれが有効に重なるかを容量・event・故障時の寿命まで含めて調べること。第三候補は、R764測定で誤差が少数の層・枝に集中した場合だけ、dtype固定、層/枝別dtype、高精度例外経路を比較すること。各案の質問、反証、出典上限、語数予算はJSONに記録した。

### レビュー対象とハッシュ

| 報告 | SHA-256 |
|---|---|
| R761 `research-stable-device-grouping-round761-v1.txt` | `c93126f7e7f2be20a939080a5b1f3dd7135fb30e7003231e5299ec806bb5fbd9` |
| R762 `research-mapped-host-vs-dma-round762-v1.txt` | `7b4e26167ff885c40a1cac909768943a4fff9ac6e1cc469cd98d4ded56e20e73` |
| R763 `research-nonllm-pipeline-residency-round763-v1.txt` | `593e7cc4e9232ec3854ee79dc2ac2eae30b2cced1759732946bc0003cd9dc219` |
| R764 `research-residual-storage-quality-round764-v1.txt` | `968387036b56645d314599444ee73be9cc828b315b0185cae359c16c86e78ade` |
| R765 `research-ur-v2-hostcopy-source-round765-v1.txt` | `10a6985575d394bf8445f5f7fc5194f75d11bd5574396439e1fc9ae83f20f9e7` |
| R766 `research-gpu-liverange-rematerialization-round766-v1.txt` | `e6a1809eb4488b24eedca13575720b5d01a6053b0383dc42a203f58cdaa38a45` |

機械可読な短い索引、Main reviewの参照先、今後3件の条件付きスコープは隣接する `research-scope-review-round767-v1.json` に記録した。
