# R806 R805レポートレビュー

R805は1,597語、7資料で割当範囲内です。直接のcolumn-major B、再利用するpacked/VNNI B、SLMまたはsubgroup内の再配置を、変換費・再利用・tail・局所メモリ・barrier・register圧力と合わせて比較しています。契約表と、IR/ISA・実行binary・device計測を組み合わせる限定的な実験案があり、知識収集として範囲と深さは適切です。

R801のR798はdense GEMMのtile形状、padding、grouping、K分割の比較です。R805はB operandの配置と、API契約から実binary/ISAまでの証拠の違いを扱い、調査軸が異なります。R796のoneDNN型/API/queue契約とも別の問いです。以前の収集レポート本文は再読していません。

報告は、joint_matrixのlayoutやAPI適合、GPU queue、DPAS仕様から実機のcodegenや演算場所を推定できないと明記しています。実際のArc binary、IR、ISA、device counterは調べられておらず、layoutの勝者や性能差は未測定です。7資料の版・revision・hashは報告内にありますが、外部資料/hashは独立確認していません。例示されたローカルsource revision `9c1a2020` は短縮prefixなので、具体的なsource比較時にfull revisionまたはfile hashが必要です。

Main提供の現状では、BF16の全体速度低下はありますが、coarse queue区間にhost submit gapが含まれ、GEMM配置の原因とは特定できません。R803の6条件は実行中で結果未確定です。したがって本文の即時参照は不要です。R803完了後、具体的なshape・runtime・binaryに結び付いた再現性のあるGEMM費用またはpacking/transpose判断が開いた場合だけ、「Candidate paths」「Compact contract table」の関連行、「One prospective discrimination protocol」、必要なsource entryを参照してください。

現在の情報から新たな研究依頼は推奨しません。R805に限定的な比較手順が既にあり、まずR803の実測とMainによるcodegen判断が必要です。研究index本体は変更せず、追記案とhashは `research-scope-review-round806-v1.json` に記録しました。

## 運用メモ

- MainはR805本文を現時点で読まない。`main_original_needed_now=false`。
- 次の調査候補はなし。R803の実測後にMainが判断する。
- Orcaのsettlementはworker成功、releaseはterminal終了を記録しています。
