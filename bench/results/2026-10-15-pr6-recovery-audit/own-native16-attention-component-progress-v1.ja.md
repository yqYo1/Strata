# BF16 QSA attention の小入力確認

Main の private build/smoke は正常終了。旧FP32実装のprefixはbyte同一。R787の補助関数を実際の登録済みDevice allocation全範囲/context検証へ補強し、host copyの例外時もdrain前に参照先を破棄しない。数値式/oracle/toleranceは変更していない。

page4,width33、nq1通常/nq3maskedの全出力をFP64 oracleと比較し、finite/guard/input immutableを確認。max_abs_rawは0.000883624と0.00105065、fixture規定budgetに対する最大比は0.393173/0.436476。extended guard/specialも完了。flushed UR launch traceあり、fresh xe fault0、dumpなし、全owned PIDをreapしsession空。これは小規模component PASSのみ。full fixture、モデル品質・速度、物理262144は未検証。R818に組込み用source proposalが届いたが未適用。

詳細は own-native16-attention-{build,smoke}-v1.json。
