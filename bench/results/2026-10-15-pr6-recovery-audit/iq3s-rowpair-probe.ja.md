# IQ3_S nt1二row候補の棄却

2026-10-15 JST。Ryzen5600X、1core affinity、同じ現model buildのGGML CPU/base archives、
Intel LLVM znver3、FP収縮/結合の自動変更OFF。製品sourceは未変更。
`check_iq3s_rowpair_probe_v2.py`でbuild/CPU correctness/clean timingをGPU runと直列に実施した。
元v1はfixtureの__rdtscp用x86intrin.h不足でcompile FAIL。元receipt/recipeを保持し、
v2では独立fixtureのheaderだけを修正した。製品用source.diffは同じ。

通常buildとASan/UBSan buildの各353caseがbitwise PASS。344のrow/pattern/alignment/guard-page
数値caseと9の拒否/empty caseで、GGML/existing-exact/新candidateの最終ffと新candidateのraw g/uを
照合した。既存static archives内部はsanitizer未instrumentedであり、その全accessの証明ではない。

各regimeで12round×3比較×ABBA4sample=144sampleを全件保持した。全armで同じ重み/activationと
workを使用し、開始終了CPU/AUXのmigration観測なし。TSCをcore cyclesとは呼ばない。

| regime | GGML ns/output row中央値 | existing exact | 二row候補 |
| --- | ---: | ---: | ---: |
| one-expert hot | 525.6336915 | 447.2151370 | 547.3365235 |
| 64expert / 90.112MB sequential | 550.7337345 | 475.4715695 | 692.3893250 |

新候補はGGML比でhot約4.13%、stream約25.72%遅い。既存exact比でも遅く、**製品へ適用しない**。
既存exactはGGML比timeがhot約14.92%、stream約13.67%短い。この合成IQ3_S/nt1形状に限定する。
他format、nt2..4、並列pool、実modelの改善率やquality/adoptionを証明しない。
modelにはGU18/22/21/23が混在し、IQ3_Sは10/48層。次はactual format serviceから優先対象を決め、
既存STRATA_IQ_SINGLE_EXACTの有効化も同じmodel条件で独立評価する。

保持は全個別標本/数値結果/source/commands/compiler/archive pinsと、元compile診断。
成功full compile/test stdoutの重複はcompact receiptの検証・commit後にretention reviewする。
原候補はGitから再現可能で、原FAILをPASSへ書換えない。
