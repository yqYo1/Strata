# 今回の16-bit化の範囲

ユーザー指定どおり、比較の主基準は同じモデルのSGLang/vLLMの実際の該当処理。
今回QSA kernelに採ったのはSGLangの「Qをscale・log2変換してQ dtypeへ丸め、BF16 QK、
BF16 P/V、FP32 online softmaxと行列累積」の設計。vLLMのQK後score scalingは別契約として
区別する。両者のsource確認は実行時CUDA branch/loweringを本機で確認したという意味ではない。
既存FP32/ggmlの結果は診断として残す。差があるだけで不合格とする、同じ生成tokenを要求する、
またはBF16丸め境界の差に合わせてoutput epsilonだけを緩める、という基準は採らない。

|部位|今回の形式・扱い|
|---|---|
|長大prefillの保存hidden/residual|既存native16経路のBF16。GPU/host保存とも2 bytes/value|
|projection/MoEの入出力|既存native16のFP16/BF16の型付き経路。全engineが一律BF16という意味ではない|
|typed QSAの投影値・Q/Kの正規化/RoPE後のtensor・出力|BF16。局所RMS/rotationはFP32|
|persistent KV|既存K8/V8コード＋per64 FP16 scale。量子化形式は変更しない|
|QSA matrix operands|Q/K/V/PはBF16。新prefix候補は使用中KVのみを一時的にBF16展開|
|QK/PV matrix累積・softmax最大値/normalizer/持越しaccumulator|FP32。FP64 oracleの固定順序と同じbit値は要求しない|
|QSA indexer・選択score、routingなど|既存のFP32計算/決定が残る。今回selectorの意味は変更しない|
|GDN recurrent state/conv内部|既存FP32を維持。これも一律16-bit化済みとは主張しない|
|PLE・最終callback|既存native16経路にFP32へのbridgeが残る|
|旧prefill scratch arena|BF16の型付きviewを使うが予約量自体は今回縮小していない。長大residualと新prefix workspaceは別の容量計画|

今回追加したprefixは永続状態ではなく、毎call・同一GPU queueで生成して使用する。
保存/復元すべきdecode cacheを増やすものではない。CPUで行分けしてKVを戻す処理も追加しない。
モデル入口は既存typed QSAと同じMode0 K8V8/text/owned RoPE範囲。Mode1・full-model256K・
RESTORE・代表的task品質は未認定のまま。既存ログのnative16共通bannerが「QSA/GDN internal
FP32」と総括する箇所は型付きQSA/indexerを十分に区別しないため、現在の凍結測定後に表示を整理する。
追加のQSA BF16 operands/prefix bannerとsource entryを、実際の新経路の証拠として記録している。
