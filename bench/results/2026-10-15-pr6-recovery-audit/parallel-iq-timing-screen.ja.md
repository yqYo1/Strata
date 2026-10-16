# Parallel IQ: clean resident dequant + oneMKL screen

2026-10-15、Arc B570、driver 1.17.39395+14、既存custom clang22/oneMKL2026.1。
推論本体はまだ変更していない。source/build/環境/個別sample/終了とfaultは
parallel-iq-timing-build-v2.json、parallel-iq-timing-smoke-v2.json、
parallel-iq-timing-screen-v2.jsonに保存した。今後のwide測定とは別の閉じたscreenである。

v1 timingは毎trial後にshared-USMをCPUで読むため実行しなかった。
Mainはv2の全sourceを読み、device-USM、明示copy、checked drain/sticky async、
不明completionでのfail-stop、heap host stagingの寿命を確認した。
元IQ archive memberをlinkしないこともlink mapで確認し、同じ候補iq.cppと既存runtimeを使った。
device bufferの全配列/7種guard/input/referenceをtiming前後だけで検証し、trial間にcopyしない。
smokeの両arm正常終了と全GEMM output bytes一致を確認してからclean screenへ進んだ。

各armは独立process3件、first armを交互にし、各processでwarmup10回後、20回batchを7trial。
展開は毎repeatやり直す。reported値はprocess内のcompleted batch wall/repeatsの中央値、
次の表はその3つのprocess中央値の中央値（us）。host submitも別に全trial保存している。
GEMMはne64、GU N1280/K2560、Down N2560/K640。FP16 operand/FP32 outputは元と同じ。

| Role / format | OFF us | ON us | median wall reduction |
| --- | ---: | ---: | ---: |
| GU IQ3_XXS /18 | 346.657 | 378.511 | **-9.2%** |
| GU IQ2_S /22 | 316.557 | 213.646 | 32.5% |
| GU IQ3_S /21 | 303.780 | 193.866 | 36.2% |
| GU IQ4_XS /23 | 361.156 | 197.509 | 45.3% |
| Down IQ4_NL /20 | 211.093 | 59.086 | 72.0% |
| Down custom Q2_0 /42 | 172.835 | 76.602 | 55.7% |

5形式ではONの最も遅いprocess中央値もOFFの最も速い中央値より短かった。
type18はrangeが重なるが、3組ともONが遅く、全形式一括の採用はしない。
特にtype20のOFFは128.583/432.042/211.093usと変動が大きい。
中央値だけで安定性を主張せず、全7trialと全個別processを保持する。
全36processの全half/GEMM reference、guard、input検証が通り、fault0/dumpなし/boot不変、
全所有processは正常exit/reap/session空。source時のmathを緩めて結果を通していない。

これはwarm device-residentなsynthetic expert componentのcompleted wallである。
転送、実modelのrouting/format-ne分布、CPU expert、queue overlap、prefill wall、decode tpsは
測っていない。kernel時間の合計やLLM全体の改善率には置き換えない。
R620の指摘通り、現行layer-major用のformat×ne censusがまだないため、過去のchunk-major
global histogramから改善率を加重平均しない。

現在の次段階はdequant単体（ne4だけ、neはworkを変えない）とGEMM ne4/16/128/512の
追加180process。既存ne64と合わせ各形状3process/armを揃える。
type18はper-element経路を採用せず、R619のsubgroup metadata共有を別候補として作成中。
実際に改善したrole/formatだけの統合、実model/bench.shでの32K以上と64K、agent session、
thinking長文生成、prefill/decode別、RESTORE/cancel/物理256Kは引き続き未完了である。
