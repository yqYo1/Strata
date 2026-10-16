# 次の部品検証と長文prefill候補の照合

2026-10-15。MainがR629/R630、INT8 head-split fixture v3、実際に構成される
root SYCL sourceを読んだ。進行中の同じ所有job `16956` を観測して存続を確認した。
`check_selective_iq_bench_v1.py full 0` はagent sessionまで進行中で、終端receiptは
まだない。測定中は新しいbuild/test/GPU/cleanupを実行していない。

## Attentionの検証範囲

R629の重要な区別を採用する。QSAの論理context上限262144と、attentionが一度に読む
選択位置の最大数2051 (`idx_top_k=2048`, `idx_block=4`) は別である。
fixture v3はactive count/capが2050/2051で、最後のslotだけlogical262143を指定し、
table[65535]=2、5 physical pages、40 backing rowsにする。末尾はKV headごとに
physical row19/23へ解決される。INT8 K/V各40*256、FP16 scale各40*4、Q24*256、
steps4要素という実際のconstructor/row/oracle契約をsourceで確認した。

これでCH32/64の最後のpartial tileを通る。物理256K poolの確保・使用、実際のselectorが
そのIDを出すこと、モデルの256K/RESTOREを証明するものではない。CH32のbitwise比較と
CH64の既定誤差基準は変更しない。ソースのみの確認で、build/host/device gateは未実行。
旧v2 serial controllerのPID/reap契約は採用しない。

R629のDirect0/Direct1成功を待つ記述は過去の実験計画である。過去のDirect0は失敗し、
Direct1は未実行のまま閉じている。これを恒久的な新しい許可条件にはしない。
現在の実行を直列化するのは同じmeasurement lockとMainの所有process管理であり、
進行中のIQ比較が閉じた後にMainが次の実験を選ぶ。追加のユーザー承認は要求しない。

## R630の訂正

- 約172秒は`selective-iq-model-v1`の診断実行で、cleanな正式速度測定ではない。
  Nativeは正常終了したが、collectorはn_promptと実prefixの1差で失敗した。元の失敗を
  保持し、別のclosed-evidence assessmentで32088 prefixのcensusを検証済みである。
  R630の「正式run」「172秒clean sample」という呼び方は採用しない。
- `src/kernels/sycl/kv_stream.cpp`の`runs()`は、K/V codes各256 BとK/V scales各8 Bを
  KV headごとに足す。2 headsで1056 B/cell、page size4で4224 B/pageとなる。
  報告の1040/4160はscale片側の計上漏れである。
- `Prefill::Impl`は単一の`m.stage`を持ち、`take_stage()`は一層分のfull logical KVを
  確保する。各QSA層に専用stageがあるという説明は誤りである。
  `run_layer_major()`ではlayer外側、`run_impl()`ではchunk外側になり、stage_lb_/le_で
  一層だけに絞るため、同一QSA層の連続chunk間の再利用は検討余地がある。
  通常chunk-majorでは層が切り替わる。owner/position/generationの証明なしに
  全経路のrestageを省略してはならない。R631でこの寿命・順序を独立監査する。

page-alignedな理想形状での、12 QSA層の過去chunk-prefix再転送の論理byte算術は
次の通り。実測PCIe trafficや削減可能wall時間ではない。

| Input | Chunk | Prefix cells summed per layer | Logical bytes across 12 layers |
| ---: | ---: | ---: | ---: |
| 32768 | 4096 | 114688 | 1453326336 |
| 65536 | 4096 | 491520 | 6228541440 |
| 65536 | 8192 | 229376 | 2906652672 |

## 次の実行

まず現在のclean full OFF比較を閉じ、成功時に同じbinary・同じrecipeでONを測る。
一部の有効なshapeで得た部品改善をモデル全体の改善としない。変動が重なれば別process、
逆順でも比較する。prefillとdecodeの採否は分離する。

次のIQ3_XXS subgroup候補にはMainの別build recipe `build_iq3xxs_subgroup_v2.py`を
準備した。product sourceを変更せず、選択的5形式版をbaseにしたfixtureとdevice IRを
別出力へ作る。未実行である。実build後もSG16/SG32 lowering・collective参加のMain
レビューを完了するまではGPU起動gateを通さない。

chunk4096/8192の比較は既存設定で可能だが、scratch増によるresidual GPU rowsやfit
fallbackの変化を記録する。これらが変わる場合、単なるchunk境界費の比較とは言わない。
KV stage再利用、cache容量、attention kernel変更を同じ初回armに混ぜない。
PR#6の645.85 tok/sは未再現、現在の1000/70目標は未達成、物理256K/RESTORE・task品質の
最終gateも未完了である。
