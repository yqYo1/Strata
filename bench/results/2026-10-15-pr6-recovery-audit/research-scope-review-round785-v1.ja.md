# R782/R783 レビュー（R785）

R782 と R783 はどちらも decision support として依頼範囲内で、報告書はそれぞれ 1,566 語、1,485 語です。R782 は HC/mHC と GDN の型境界を vLLM/SGLang の固定 commit で示し、FP32 の係数・累積・再帰状態を保存する条件を整理しています。R783 は GEMM 出力型と消費側のつながり、Intel `joint_matrix` の候補を比較しています。両者は型境界とカーネル/API選択という異なる問いを扱っており、今回の実装に必要な情報があります。

R782 の vLLM/SGLang 参照は commit 固定で、ファイル・関数・ライセンスも具体的です。R783 はファイルと関数を示しますが、vLLM-XPU、LLM Scaler、Intel LLVM の主な URL が可変 branch で、参照元の commit は記録されていません。報告書が示す Strata の行番号も、固定された source snapshot ではありません。実装へ再利用する場合は現行 source で確認し、immutable pin と完全なライセンス・notice review を追加してください。両報告は設計参照であり、コード移植の承認、B570 上の実行保証、性能・品質結果ではありません。R783 の vLLM-XPU interface は BSD-3-Clause notice を持つと報告され、リポジトリ root の Apache-2.0 と区別されています。

今回の実装で読む箇所は次だけです。

- **R782** — `## Contract evidence` の HC/mHC 行と GDN state 行、`## Mapping to Strata` の冒頭2段落。BF16 residual の producer/consumer 型、FP32 境界、再帰状態の分離に使います。
- **R783** — `## Candidate kernel/API contracts` の **Current Strata seam** と **Owned joint_matrix**、および `## Constraints and bounded validation` の冒頭段落。出力型と累積型の区別、Strata の呼出し境界、実機 tuple の照会条件に使います。FP16 GU 推奨や第三者実装例は今回の BF16 経路へ一般化しません。

R783 が推す初回実験は FP16 MoE gate/up 経路です。今回の BF16 residual/GEMM とは別候補なので、フォーマット判断として転用しないでください。R783 自身も BF16 には入力・重み形式の別判断が必要で、joint-matrix の組合せを実機で確認すべきだとしています。どちらの報告も Strata の性能、自然長入力の品質、262144-context 適合性は測っていません。

次の新規スコープ候補は、HC4 BF16 residual-combine のスレッド割当てと隣接カーネル融合の比較です。R782 は型境界、R783 は GEMM/API を扱い、この residual mapping の比較は未着手です。公開ソース比較だけで開始でき、B570 上の速度主張は含めません。品質対策の調査は、Main が matched FP32/BF16 自然長入力で回帰を確認した場合に限る条件付き候補です。詳細な eligibility metadata と入力・出力 hash は同名 JSON に記録しました。

Operational digest（400字以内）：

> R782はHC残差の16bit入出力、FP32係数・累積、GDN状態との境界に使えます。「Contract evidence」のHC/GDN行と「Mapping to Strata」冒頭2段落のみ参照。R783は「Current Strata seam」「Owned joint_matrix」「Constraints and bounded validation」冒頭のみ。FP16推奨をBF16へ移さず、実機tupleを照会してください。R782はcommit pin済み、R783は主要URLが可変branchです。
