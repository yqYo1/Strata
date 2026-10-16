# 既存Direct設定のint8初回検証

同じ`0c281323...`バイナリ、513入力、257チャンク、通常MTP4、8生成。
`STRATA_SYCL_PROMPT_DIRECT=1`で行列attentionの最後の出力方法だけを変更した。
短い入力は正しさとlifetimeの検証用で、速度比較には使わない。

最初の`check_prompt_direct_small_v1.py`は、詳細UR APIログが64MiBの上限に達して
SIGXFSZで終了した。結果は未完了/失敗のまま保存した。
元owner内の最終journal取得も既に満杯のaggregate log上限に引っかかったため、
`check_direct_failure_faults_v1.py`が新しい有限ownerで同じcursor以降を取得した。
終了プロセスは全て回収済み、新規GPU障害0、dumpなし、boot/driver変更なし。
途中終了した計算の数値や性能は評価しない。

`check_prompt_direct_small_v2.py`では警告・エラー・parameter validationと
通常の進捗だけを残し、同じ入力・同じ数値比較を実行した。
正常終了・回収、新規GPU障害0。基準と以下が全て一致した。

- 9位置の248320語彙分のselected logits全バイト。
- 最初のlogits全バイトとmetadata。
- 8生成token IDs。

`prompt-direct-small-v2.json`にcommands/source/binary/input/environment、全hash、
exit/cleanup/faultと比較結果を保存した。precisionやsoftmaxの演算順を変える案ではない。
実モデルの34K長文、clean反復性能、物理256K、RESTOREの合格はまだ示していない。
次は同じ既存設定の実34Kプロファイルで`launch<1,true,false>`とその回数を確認する。
