# BF16 local dense16/VNNI配置の単体比較

Arc B570 / xe1.17.39395+14 / pinned LLVM22.1、同一binary・入力・割当。現行score33に対しQ/Pをdense16 tile、K/VをBF16 pair-packed B tileへ変更。global source座標、BF16値、FP32オンライン状態/累算、32cell更新順、barrier数、local payload28,224bytesは同じ。全11component（physical262144含む）、独立oracle、guard/入力不変、traced二arm、全6独立processともPASS、所有終了、新規GPUfault0。

各processはwarmup2+sample10。process medianは現行333.381292/333.342006/333.3924735 ms、新配置311.2022445/311.120328/311.073618 ms。中央値333.381292→311.120328 ms、latency **−6.6773%**。全output FNV一致、sampled oracle一致。保存全出力のbyte比較ではない。

IGC診断ではselected SIMD16 attentionの静的命令4977→3901、静的DPAS16→24、SLM load576→306。SIMD16/GRF128、predicate flag spill store22/load28は同じ。GRF spill size headerは両方なし。packはSIMD32/1154命令で同じ。dynamic local配列28,224Bとzeinfo static slm_size32Bは別物。数学上の行列積と32cell更新順は同じだが、compilerの展開・分割により静的DPAS命令数は増えている。命令数は動的実行回数・資源飽和率・帯域・改善原因の分解を意味しない。

モデル全体の速度・品質・256K全消費・RESTOREは未検証。次は専用opt-in経路として統合し、現行score33との同一binaryモデル比較を行う。既存の精度、短入力/decode経路を保つ。
