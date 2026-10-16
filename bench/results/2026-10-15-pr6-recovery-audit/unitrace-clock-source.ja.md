# unitrace外部時刻ずれのCPU再現

2026-10-15 JST。既存PTI6c0d6b0bの変更していないUniTimer headerだけを
通常C++でcompileし、GPUを初期化せず32回測った。
BOOTから作ったepoch offsetへRAW timestampを加える既存式の結果は、
REALTIMEより約3.152秒前になった。BOOTTIME−RAWの実測差を算術で考慮すると、
最初のsampleの残りは[-60,+90]ns区間となった。全32 samplesを
`unitrace-clock-source-v1.json`へ整数nsで保存した。

既存の`UNITRACE_SystemTime=1`ではheader結果がRAWそのものになることも
別CPU process32 samplesで確認した。これは内部flagであり、実GPU traceでの
RAW-mode外部対応を認定したものではない。今後使うならcollector側もRAWの境界を記録する。
profiler binaryは変更・再buildしていない。

今回の再現はclock domain混在の原因を裏付ける。過去のGPU traceを任意量だけずらして
UTC合格へ変更しない。過去・今回のcollection原結果、外部UTC gate falseはそのまま保持する。
内部のkernel選択・依存順序の証拠と外部epoch attributionは区別する。
source clockの疑問を消費していた500709Bの過去raw traceと、抽出済みのGEMM raw traceは、
次の直列cleanup境界でこの記録・既存compact evidenceを確認して退役できる。
