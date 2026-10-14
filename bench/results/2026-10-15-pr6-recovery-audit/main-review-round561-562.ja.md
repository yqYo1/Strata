# QSA layoutとstreamed KVの復旧契約レビュー

2026-10-15 JST。Luna R561/R562を受領して読んだ。実装・build・GPU試験の合格ではない。
主エージェントは公開セッション記録354件を元のbyte offsetから再読し、role/channel、timestamp、
元message SHA256を照合した。`public-history-reverification.json`は351性能記述と13採否説明の
重複を除いた354件の再照合結果で、過去のwhole-log digestを新たに計算したものではない。
原文は従来のcanonical indexにあり、再開メモや生のセッションコピーは増やしていない。

## layout=1の復旧範囲

旧`1eb89482:sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp`の式を直接確認した。
queryの次元dをlocal memoryの`(d%8)*32+d/8`へ格納し、laneが`lane+j*32`を読む。
この対応は元の`lane*8+j`次元を読ませる置換であり、KとQの組は変わらない。
旧計算全体の演算順を移植するのではなく、現在の式・reduction・mergeを保持する。
現在の`src/kernels/sycl/decode_attention.cpp`に同じSG32/64cell split構造がある。
現在のmatrix prompt attentionとは別のフォールバックで、layoutだけでlaunch数は減らない。

既存decodeのentrypointは自然順を維持し、prefillのfallbackだけで候補を選ぶ。
matrix側のadmissionは変更しない。旧測定ではlayout=1を指定していたことだけで、
現在のdefault matrix/fallback比率や新候補の改善幅は証明されない。
まず同じカーネルのOFF/ON、全KV形式、page/選択width境界、最大論理cellを検証し、
実モデルで実際のfallback利用とprefill/decodeを別に確認する。

## 現在のSYCLによるKV stagingの照合

R562のCUDA参照だけでは現在の実装の証明にならないため、
`src/kernels/sycl/kv_stream.cpp:264`の`kv_stage_from_host`を直接確認した。
format別のsource/destination spanを検査し、同じin-order queueへ配列ごとのmemcpyを提出する。
root prefillは先行ページをstageへコピーし、その後に当chunkをappendしてidentity poolでattendする。
resident slot数と全論理ページ分のhost/stage容量は別である。
current native indexerは`src/kernels/sycl/native_qsa_indexer.cpp`で連続posのappendを順に提出し、
block+1のspareも維持する。旧CUDA batch実装を現在の動作として扱わない。

R562の「chunk 0を全層、次にchunk 1という順序が不正」という説明は誤り。
現在のchunk-majorも各層の過去stateを保持し、正しい依存順である。
layer-majorもascending chunkと各層のstate維持が成立すれば有効な別の依存順になり得る。
また現在の`run_impl`はPLE対象層の有無とは別にchunk tokenで`ple_prev`を更新するため、
旧ループの開始時token history復元を不要と断定しない。callbackとdraftには最終層の残差を
token順で一度だけ渡す必要がある。これらは実装前の詳細確認事項である。

次の大きな候補は、現在のrootのchunk計算を再利用するレイヤー順とexpert再利用。
full layer expert cache、残差全量/部分VRAM、既存decode workspaceの寿命と実VRAM admissionを
組み合わせて検討する。未確認のまま旧ソース全体やCUDA参照へ切り替えない。
