# 本番と同じ8192 queryでのQSA資源比較

2026-10-16、B57010GiB・driver1.17.39395+14・同一boot。量子化K8V8、
FP32 Q/O、24 query heads、2 KV heads、dim256、2051 selected cellsの合成fixture。
kernel実装は変更せず、全armで同じ823485888Bをdeviceに確保する。

| arm | 変更点 | 独立process中央値3個（ms） | その中央値 |
| --- | --- | --- | --- |
| A | shared KV、1 commandに16384 workgroups | 595.107476 / 595.066085 / 595.051588 | 595.066085 |
| B | query間の参照先を128組に分散、16384 groups/command | 716.144373 / 715.797057 / 715.757266 | 715.797057 |
| C | Aと同じ参照、1024 groups/commandを16回 | 624.157486 / 624.179827 / 624.212448 | 624.179827 |

各process2warmup＋10timed、合計9process/90sample。順は
A0 B0 C0 C1 B1 A1 A2 C2 B2。各processの全201326616B出力を
独立comparatorでcanonicalと比較し、全FP32有限・全payload byte一致PASS。
全所有process正常終了・新規fault0・dumpなし。詳細traceを外した測定。
全sample・source/binary/runtime/environment・比較receiptをJSONに保持した。

Bは選択されるunique K/V/scaleを2165856Bから277229568Bへ128倍にし、
**20.29%遅く**なる。参照先の変更によるworking-set感度はある。
これはphysical DRAM trafficやcache counterの測定ではないため、VRAM帯域律速とは断定しない。

Cの悪化は**4.89%**。既存128-query比較の2.63倍悪化は大きく縮む。
本番は既に8192queryなので、小さいgridで見えた並列数不足だけを
本番QSAの遅さの原因とする説明は棄却する。launch発行時間はA0.088953ms、
C0.3270715msに対しwholeの差は29.113742msであり、追加のCPU発行時間だけでも
差を説明できない。ただしXMX/SLM/barrier/register/cacheの支配までは未確定。

後半processと並行してread-only研究workerを起動した。CPU isolationや
機械全体の物理利用率は測っていない。個別値と順を残し、小さな差を過剰解釈しない。
このfixtureはuniformな合成selected IDsであり、model PPの測定や高速化採用ではない。
attention service約34秒の説明に必要な条件を絞る結果として使う。

全candidate frameはnative＋全量比較をGit保存後に削除済み。全cohortの確認と
summary保存を終えたため、canonicalと成功したraw診断も用途を終えたものとして整理する。
再計算は `analyze_qsa_large_resources_v1.py`。既存出力は上書きしない。
