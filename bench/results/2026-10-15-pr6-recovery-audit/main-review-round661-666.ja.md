# Residual品質と未適用head/NLL案のsourceレビュー

2026-10-15。R661〜R666のFINAL後にMainが報告と対象sourceを確認した。
別GPU/test/build/cleanupは実行していない。採用、性能、品質、物理256K/RESTOREの合格ではない。
新しい長文prefill優先方針と処理順の本体はlong-prefill-execution-order.ja.mdを参照する。

R661の外部資料はQwen authorsのFP8 residual stateの定性的な記述であり、今回のFP32演算を保った
FP16 layer-boundary storageの品質損失率を定量化していない。Mainもpaperのinference efficiency節を
前turnに確認したが、今回の形式・context・繰り返しroundtripを合格扱いにしない。
R661のcurrent ON has been measuredは実行開始/途中観測と解釈し、suite完了の証拠には使わない。

R662が区別した三つの数値を維持する。synthetic513入力のFP16 recurrent screenの約9.73% rel-L2は
選択されたfinal residualで、約0.027 KLは一つの次token分布。saved tensorのFP16 rel-L2約0.000211は
一回だけのpost-hoc cast。自然34603入力の19相関位置の小さなKLも全入力PPL/task品質ではない。
現在BF16 residual storageは未実装。提案された品質閾値をユーザー承認済みと呼ばない。
既存selected-prefill-logits callbackはstandalone限定で、serveは事前拒否する。旧auto24576 FAILは維持。

## Indexed head v1/v2

v1は既存のfull NativeHead Q6_K重みからdvocabの行を直接読む、default-OFFの案である。
数学・量子化・還元順・subsetの順・sampler/remapを保持し、重複dheadを作らないことを狙う。
213MiB前後という値はrequested duplicate bytesであり、実freeの変化や速度ではない。
Mainはv1 patch、helper、host/GPU fixture、generateの借用ownerの宣言/破棄順を読んだ。

R664の「subsetなしでindexed_head_がtrueとなり初回forward失敗」という結論はsourceと一致しない。
該当行が書くのはindexed_bind_complete_とweightsであり、indexed_head_ではない。
has_draft_subsetはdhead_またはindexed_head_を読む。indexed_head_=trueの代入はmap upload成功内にある。
したがって報告された初回失敗は裏付けられない。ただしno-subsetにindexed完了markerを付ける不整合はある。
原R664はbytes/hashを保持し、このMain補足で訂正する。

Solの別v2案はそのmarkerを実indexed成功に限定し、shared bounded map validation、保守的preflight、
MTP logits ownerのsubset<=full容量条件を追加した。非Q6/真正no-subsetは明示したprefer方式で旧経路へ戻す。
R663のrequire-mode提案は仕様として採用しない。不正mapをindexedへ公開することも認めない。
Mainはv2の変更箇所とREADMEを確認したが、startup fileのpreflight/bind間の変更は原子的に防いでいない。
captured mapの再bind固定と全三production bind callerのretire leaseは後続native gateでも確認が必要。
v2 fixtureはni256/2560、production full248320/subset106299/T1、三graph replayを追加したsourceで、未実行。
新しいdecode性能探索には使わず、長文prefillの容量確保に必要な場合にのみMainが順次検証する。

## Teacher-forced NLL v1

全initial-batched inputの各prediction位置で既存GR read→NativeHeadを使い、goldのNLL一値だけを保存する案。
同じqueue上でhead→stable FP32 logsumexp→次headを発行し、chunk単位にreadback/waitする。
rawには全input IDと位置順のgold/NLLが入り、262144 inputを表現してもraw3145736 B、metadata<=4096 B。
Writerは連続coverage、有限で非負のNLL、exclusive create、flush/close、完了linkの公開を確認する。
完了markerはprompt/MTP restoreとdiagnostic ownerのchecked retirementの後である。
Mainはpatch/consumer/host/GPU fixture sourceを読み、R665/R666のscopeを確認した。

R665のspan指摘は正しい。現row_target_nllはalignment/overflow/alias/USM contextを調べるが、実allocation
の残りbyte数をraw pointerから保証しない。production callerのallocate/offset計算は別のowner契約である。
APIコメントを明示し、extentを検査したと主張しない。undersized USMを単にenqueueしてGPU OOBを誘発する
negative testは実行しない。registry等でhost事前拒否できる経路に限って検証する。

R666のscope/model依存checkは区別する。mode/length衝突は事前拒否、実vocabのID・実head・filesystem衝突は
後段setup時の検査であり、すべてをruntime前の拒否とは言わない。consumerはtoken同一性と全coverageを
比較するが、同一model/source/binary/configは外側recipeが必須。storage_dtype_placementもそのrecipeでpinする。
relative pathは現sourceで禁止されていない。immutable closed captureを外側ownerが確認してから比較する。

Main追加: Writerの262144-input表現能力とCLIの実admissionは異なる。既存generateの二つのmax-context guardは
positive max-newとprompt+generationの余白を要求する。262144 input/context262144の物理検証がこの案で
できると未確認のまま呼ばない。最終入力tokenもbatched n_prompt-1の外なので、full-context/state検証は別gate。
診断はFP16品質判断を助ける道具であり、品質の受入閾値/全context/RESTORE合格の代わりにならない。

両proposalはsourceのまま保存し、今の計測binary/sourceを変えない。新優先方針ではprefillの実時間と容量を
先に検証し、必要な診断/改善だけを適用する。decode70、prefill600/1000の達成を先取りしない。

source-proposal-integrity-v1.jsonは三proposalのmanifest payloadと現在rootのbase SHAを実照合した記録で、
patchの適用/再構成やfixture実行ではない。Git全staged diff --checkは原R661末尾の空行一件と、二つの
source.diff内の空行contextを表す一文字spaceを警告した。原報告/patch/hashを維持し、この三artifactだけを
除外したsource/docsのwhitespace確認と区別する。full diff --checkをPASSとは記録しない。
