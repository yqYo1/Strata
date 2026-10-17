# window10: ログなし初回・後続長文比較

Arc B570 10GiB / Ryzen5600X / 128GiB RAM。同一binary、公式bench.sh、thinking有効、
入力78453 token、context131072、INT8 KV、chunk8192、全GPU BF16 residual。
各armは同一engineで4つの異なる要求を実行し、全件reuse=0。TRACE/UR layer/profileは無効。

|MoE mode|初回PP(tok/s)|後続3回PP(tok/s)|後続中央値|後続標本SD|
|---|---:|---|---:|---:|
|8|602.4|634.3 / 630.5 / 634.4|634.3|2.223|
|10|618.8|652.0 / 648.1 / 652.2|652.0|2.312|

window10の改善率は初回2.722%、後続中央値2.790%。
同一入力hash、各要求の完了、単一engineのPID/start ticksと起動前exclusive claim、
正常終了、所有process消滅、新規GPU fault/dumpなしをMainが再確認した。

初回はarmごと1標本、後続3回は一つのprocess内の標本であり、新規processの反復とは区別する。
free VRAMやcache/graphの状態が変化するため、初回と後続の差をJIT時間とは断定しない。
24 thinking-tokenは完了probeで、一般品質、長時間decode、物理262144消費とRESTOREは別途未検証。
速度目標1000 PP / 70 TGは未達であり、最適化を継続する。

[全個別標本・検証結果](moe-window10-first-subsequent-analysis-v1.json)
