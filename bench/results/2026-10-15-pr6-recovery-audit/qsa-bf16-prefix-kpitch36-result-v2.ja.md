# K contiguous＋local pitch36を不採用

基準は現在のscore33/明示的4-byte整列の実装。同一binary・入力・GPU領域・精度/32セル更新順、8192 query×width2051/physical32768、pack込み。各arm3独立process×2warmup/10sample。

基準process中央値333.229348/333.387611/333.3656645ms→中央値333.3656645ms。
候補398.6458945/398.5985605/398.985118ms→398.6458945ms。遅延19.5822%増、採用しない。

全6process独立4query oracle、全出力finite/guard/input不変/repeat bitはPASS、output全体FNV一致（保存tensor全byte比較ではない）。物理262144を含む11ケースと初回URはPASS、正常終了/新規GPUfaultゼロ。GPU内Kの読み出しを連続にしlocal matrix K pitchを36要素へ増やしても、全依存経路は速くならなかった。実機bank/占有率/命令列は未計測のため原因は断定しない。

Kstride72byteの行列ロードlowering・レジスター/SLM資源を小さなIGC生成コード診断で確認する。別カーネルのIGC register-allocation既知報告は本件原因の証拠ではない。現在のモデル最速prefill中央値464.1tok/sを維持。新候補のモデル実行・速度採用・full-model256K/RESTOREは実施/認定していない。
