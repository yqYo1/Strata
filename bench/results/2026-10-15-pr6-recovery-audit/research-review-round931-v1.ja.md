# R931: R928 QSA比較レビュー

R928は1,566語で、SGLangのQwen4-Exp QSAをselectorとsparse attentionまで追っている。R893の歴史的Strata経路やR732の条件付きgather論点とは異なる。プロファイル上のQSA attention 33.96秒は次の調査対象になるが、v8の合成資源比較も物理律速を示していない。

1. 数値基準を訂正する。SGLangはQ/K/Vをqueryの実dtypeで演算し、online max・normalizer・V累積をFP32で保持し、probabilityをV dtypeへ丸める。FP32累積はFP32 operandを意味せず、StrataのQ/P hi-lo積は必須条件ではない。既存native-BF16 QSAは現行精度作業に沿う候補として再評価できるが、速度・採用結果はない。
2. selector差は実在する。SGLangは512個の圧縮blockを選んで全セルへ展開し、tailを加える。Strataはcell重み付きtop-kで最大2051セルを返すため、n=2052–2054では有効幅が異なる。R928の「Triton top-k 512」は誤りで、実体はJIT C++ CUDA。出力順も未規定なので、ordered ID一致を契約にしない。
3. MainはR928の13–17行と21–29行だけ取得すればよい。次はMain所有のSYCL-only fixtureでn=2048–2055、tail全残数、境界tie、262144を検証する。設定dtype・TileLang/prefix分岐の実行記録とQSA内部stage計測がそろうまで、新しい研究スコープは勧めない。
