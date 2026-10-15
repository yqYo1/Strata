# R802 natural-long sparse screenのMainソースレビュー

manifest、run_screen.py、compare_screen.py、hostfixture.pyのコード全体を確認した。workerはテスト/ビルド/GPU実行なし。Mainもこのレビュー段階では実行していない。元のv2 source/ABIを変えず、閉じた比較の数値screenだけに使う。現在のcoarse GPU診断と同時にhostfixtureを実行しない。

32,863 tokenの固定自然文・thinking有効入力、位置255+512iの64行を、同一binary/引数でFP32-a→FP32-b→BF16の3プロセスに渡す。外部expert profileはSHAで固定し、各armの私有コピーをreset。CLIの全48層・chunk数・入力IDs・生成1token・配置、正常終了・kernel境界・boot・device dump、runtime/model/source/observer metadata/bytes/hashを確認する。元の別キューFAILを正常終了扱いにせず、独立した単一キューgateを要求する。生成tokenの一致は要求しない。

比較はmax-shifted FP64 math.fsumによるNLL、2つのFP32 control間と各control対BF16の全64位置、sampled PPL比、top1、logit差、分布KL/L1を記録。大きな共通logit offsetによる桁落ちを避ける式で、全vocabularyのfiniteチェックを行う。小さなnegative KL等を丸めて隠さず、onecorpusの相関した位置を独立標本/統計的品質保証と呼ばない。許容値・自動採用・一般/task品質PASSは設定しない。

source/hash・実環境・preparer pinの読み取り照合は整合。hostfixtureは手計算uniform/ties/共通offset、header/長さ、位置/next-token/completion、非target NaN/Infの4群を含む。GPU比較前にMainがこのfixtureを所有プロセスとして実行し、結果を保存する。動作確認後にのみscriptを採用済みとする。

rawは64x248320xFP32の約60.6MiB/arm、3arm約181.9MiB。256MiB tensor budget、384MiB全job budgetと64MiB text/file cap、48h以内の明示review期限と比較終了時の見直しを要求する。使い切ったcorpus/fixtureを恒久保存する規則ではない。source shardは既存のsize+mtime identityを継承し、暗号学的full-shard hashを取得したと偽らない。
