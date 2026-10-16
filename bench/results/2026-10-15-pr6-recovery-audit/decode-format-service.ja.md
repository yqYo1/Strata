# 実decodeのCPU format別service

2026-10-15 JST。b6838d28、同serverへ34603input/max128を2回、513input/max8を1回。
FP32固定4096残差配置、context262144/K8resident32768/cache128perlayer/pool5+host/MTP4normal。
3request全て正常length、owner正常exit/reap/session empty、freshGPUfault0/dumpなし/boot不変。
全7format×issuer/worker×roleのcounter総和が既存serviceと完全一致し、dispatch/route/phase照合も通過。

| GU type | CPU thread秒 初回 / 二回目 | GU内割合 初回 / 二回目 |
| --- | --- | --- |
| 18 | 9.296287 / 9.393422 | 36.52% / 36.55% |
| 22 | 9.475080 / 9.566850 | 37.22% / 37.23% |
| 21 | 6.375114 / 6.442336 | 25.04% / 25.07% |
| 23 | 0.311721 / 0.296642 | 1.22% / 1.15% |

18=IQ3_XXS、22=IQ2_S、21=IQ3_S、23=IQ4_XS。GUは18/22/21に分かれ、
IQ3_Sだけを対象とする最適化では全GUを改善できない。format×nt時間は未観測。
今回の長い2requestは128生成ID・route/W/R/D/A/C/Eが同じで、CPU serviceも約36.5/36.7thread秒と近い。
GPU境界host waitは7.418/7.479秒、pool呼出し8.320/8.362秒。loop25.412/19.029秒差を
CPU最適化の成果としない。cold graph準備等が別scopeに含まれ、これらを加算分解しない。
正式bench.shの性能は更新していない。

既存exact helperのtypes18/21/22は別CPUgate384casesをbitwise通過し、本体binaryは不変。
次はSTRATA_IQ_SINGLE_EXACT=1のみを変えた同条件診断でwork/CPU service/出力を比較する。
単体IQ3_Sの14〜15%time減を全model改善率へ外挿せず、採用時はclean公式benchを別実行する。

R587のsource確認ではdecodeKVは選択pageのmissだけをstageし、全history転送ではない。
R588のtimerはlaunch後からevent完了までの残りhost waitで、GPU active時間そのものではない。
次の詳細profilerはMainの一window限定・GPU作業と直列の別診断として設計する。
原R588の8GiB RSS/300sは実modelに不適切で、既に訂正済みR576と同じ誤りのため採用しない。
88GiB RSSと実起動/34Kに足りる有限wallを事前記録する。現在ITT依存は無く、追加前にprovider制御を確認する。
新たな承認はユーザーの継続最適化依頼の範囲で不要。
R587のceil(width/4) page上限は任意IDには成立せず、block selectorの構造に限定して導く必要がある。
末尾と閾値の二つのpartial blockを含む場合も確認し、一般resolveのcorrectness証明には使わない。

保持するcompact receipt/summary/追加診断には個別全標本・counter・fault/ownerと必要なmemory/
KV/capture情報を保存した。成功full traceはcommit後にmanifest付きで退役し、元数値statusを変えない。
