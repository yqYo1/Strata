# 現在のQSA prefixカーネルのnative命令とlocal引数

2026-10-18 JST、Intel Arc B570、Level Zero driver1.17.39395+14。
現在のfull-engine archiveと同じ実装をリンクした、既に数値検証済みの
小さなfixtureで`DumpZEBin=1`を起動前に追加した。専用ディレクトリ、
180秒のGPU実行期限、30MiBの出力上限を設け、tensorや引数内容のdumpは
有効にしていない。2例の数値・ガード検査が成功し、正常終了、新規GPU障害0。
その後、排他ロック下のCPU処理だけで`ocloc disasm`を実行し、正常終了した。

対象は実モデルの長大prefillに結び付けた`prefix_batch_impl<33,false,false>`。
今回直接取得したnative ELFはfixtureプロセスのものであり、実モデルの
プロセスから新しく取得したELFと同一だという検証までは行っていない。

|観測対象|結果|解釈の範囲|
|---|---:|---|
|nativeコードのDPAS命令|静的に16命令|`dpas.8x8`、BF16入力・FP32出力を実際に確認|
|GRF descriptor|128|実行時の演算器使用率やoccupancyではない|
|SIMD幅|16|ソースのsubgroup指定と対応|
|静的SLM descriptor|32 bytes|動的local引数の容量を含まない|
|起動時のlocal引数|9個、合計28224 bytes|12起動すべてで同じ要求を確認|
|local work size|128|実際のUR起動引数から確認|

local引数のサイズは8192、16384、1024、64、64、64、256、2112、64 bytes。
kernel作成時の名前とhandle、起動時の同じhandle、native ELFのkernel名を
照合した。作成時の32 bytesだけから共有メモリ使用量を判断してはいけない。
要求合計28224 bytesも、driverがalignmentとallocation encodingを反映した
最終確保量、常駐workgroup数、bank conflictの実測ではない。

これで、この対象に行列命令が生成されていることは確認できた。
静的DPAS命令数からXMXの稼働率や性能上限は計算できない。
spillの新しい実測、pitch変更時のnative resource比較、物理帯域の飽和率は
未確認。PPの新しい測定・製品既定値の変更もない。

現在の実モデル性能は別の比較のままで、78453入力token、context131072、
chunk8192、thinking有効の後続3回でserial中央値725.7 token/s、subgroup候補
733.5 token/s。候補の24-token decodeは遅かったため採用を保留している。
物理262144 context全消費、RESTORE、長い生成の検証は残っている。

再現用の取得・逆アセンブル手順は`run_qsa_native_zebin_v1.py`と
`run_qsa_native_disasm_v1.py`。保存先は1回専用で、再実行には新しい版が必要。
取得時・逆アセンブル時の元レシートは変更していない。
`extract_qsa_native_evidence_v1.py`が元ログのハッシュ、選択kernelの全起動、
生成命令とmetadataを照合し、compact evidenceに保存した。

- [照合済みの証拠](https://github.com/yqYo1/Strata/blob/8063aacc6749d18089f7a53db5d3e47b745f348c/bench/results/2026-10-15-pr6-recovery-audit/qsa-native-bound-evidence-v1.json)
- [GPU取得レシート](https://github.com/yqYo1/Strata/blob/8063aacc6749d18089f7a53db5d3e47b745f348c/bench/results/2026-10-15-pr6-recovery-audit/own-qsa-native-zebin-serial-smoke-v1.json)
- [CPU逆アセンブルレシート](https://github.com/yqYo1/Strata/blob/8063aacc6749d18089f7a53db5d3e47b745f348c/bench/results/2026-10-15-pr6-recovery-audit/qsa-native-disasm-v1.json)
- [削除記録](https://github.com/yqYo1/Strata/blob/8063aacc6749d18089f7a53db5d3e47b745f348c/bench/results/2026-10-15-pr6-recovery-audit/qsa-native-retention-v1.json)

正常時の全APIログ、非対象module、重複metadata等31ファイル、2097466 bytesを
検証後に退役した。選択したnative ELFとassemblyの保持用途だったpitch比較は直近の作業から
外れた。再生成可能な旧バイナリとして退役対象に変更した。必要時は固定
コミットのソース・取得手順から新規に生成し、元の測定と区別する。
対象ELFのSHA256は
`049221cb25a6c6e9c36f9e96a7cdee08d4309e857c21aa14e468beec9d8ab896`。
Orca R1056のsource調査とnative Luna max R1057のレビューはこの実行と別の
証拠であり、調査報告だけでdriver上の動作確認を済ませた扱いにはしない。
