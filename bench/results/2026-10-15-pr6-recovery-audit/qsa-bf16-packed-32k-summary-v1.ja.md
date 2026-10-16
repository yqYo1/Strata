# GPU BF16 KV pack:32K physical-pool comparison

8192-query QSA component with 2051 selected cells/query. Per-query rotated selections actually cover all32768 logical/physical cells; all input/guard checks and four-query BF16 contract oracle. Same initialized allocation footprint both arms. Whole physical BF16 pack included in every packed timed call. No CPU tensor work or host transfer within timed call.

| Path | Independent process medians(ms) | Median(ms) |
| --- | --- | --- |
| base | 565.116063, 565.047755, 565.331562 | 565.116063 |
| packed | 390.186218, 390.135866, 390.238254 | 390.186218 |

Pack込みlatency reduction 30.9547%, speedup 1.44832x; 3process×10samples per arm. Extra decoded payload 67371008 bytes; both benchmark arms allocate it.

256K比較は最初の既存経路で数値gateに失敗し、速度samples無し。独立diagnosticでは両経路が同じ10要素(query2/head13)で参照から外れ、全出力FNV一致。プロセス正常終了、freshfault無し。旧失敗statusは変更していない。原因未確定なので256K速度やモデル採用を認定しない。

Full32K pool is not a32K whole-model prompt/bench.sh result; model PP/quality/physical262144-context/RESTORE/production selection remain unqualified.
Minimum payload/footprint is not measured physical GPU traffic or device utilization. No CPU exclusivity claim while read-only research workers ran.
The benchmark repeat check compares float values, so it does not distinguish signed-zero bits. Cross-process final FNV fingerprints preserve bit distinctions but are not a retained full-byte comparator. The separate component fixture repeats uint16 BF16 outputs exactly.

Continue investigating the numerical boundary and real-session initialized/live extents before model integration. The32K component signal is stable and substantial; it does not qualify whole-prefill adoption.
