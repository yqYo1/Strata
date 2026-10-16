# Main review: agent cache / checkpoint settings, round 608

2026-10-15。LunaのFINAL受領後、Mainが現行SYCL入口のsourceを照合した。
このレビューは製品適用、build、test、GPU実行を含まない。Direct0/1の同一binary比較を継続する。

## 現在の比較で分かること

`check_prompt_direct_bench_v1.py`の実効設定はprompt-cache0 / turn-token-1 /
short-read0 / conversation-cache0。両armとも会話prefix再利用を無効にした計算比較であり、
agent sessionのREUSED0をcacheの不具合や最適化済みagent性能として扱わない。
途中で設定を変えず、32K・agent・thinking長生成の全反復を閉じる。

## Mainが確認したcheckpointの不整合経路

1. `src/prefill/layer_major.inc`は一つの層についてsegment全体を処理してから次層へ進む。
   `on_stage_chunk`のwrapperは最終層かつsegment終端だけを通知する。
2. 別の通常`on_chunk`は、最終層の途中を含む各chunkをserve callbackへ通知する。
   stage callbackの終端guardは、この経路には適用されない。
3. `src/program/generate.cpp`のserve callbackはMTP prompt処理後、周期閾値を超えると
   `checkpoint_at(done)`を呼ぶ。これはprefix `cur[0,done)`と現在のrunning stateを保存する。
4. `src/core/conversation_state.cpp`の`checkpoint_targets`はサイズ、pointer、token容量を
   検査するが、各層のrunning stateがprefixと同じ位置かは検査しない。
   `conversation_checkpoint_save`は現在のGDN/PLE/QSA tail/dead/block positionをcopyする。

従って、先行層がsegment終端、最終層が途中prefixという状態を保存する経路がsource上ある。
これは強い不整合候補だが、現在のcache0測定ではsaveはno-opであり、このレビューは
実GPUでのRESTORE再現や過去GPU障害との因果を示していない。

外側loopで全Prefill::runが戻ったroot/turn境界の保存は区別する。現行sourceではその位置の
checkpoint呼出しが外側loan `refill()`より前にあるため、R608の「cache-slot refill後」まで
完了しているという記述は採用しない。MTP wrapperのexpert再attachとloan refillも別処理である。

## 修正と性能評価の順序

周期保存機能を単に無効化する恒久修正にはしない。別Solが、整合するsegment境界まで
全層・MTPを処理し終えてから保存する提案を作成した。Mainのsource reviewと実行検証は未了。
segment分割は全層expert再読込を増やす可能性があるため、別のLuna round609が
単一layer-major passでの層別prefix state捕捉との費用差を調査する。

cache有効時の採用には、同じ実入力IDを用いたRESTORE対全再読、live/root/turn hit/miss、
MTP target/draft、取消・次request、近傍context境界とphysical262144の確認が必要。
`--max-context 262144`設定や小poolのID262143 fixtureだけでfull256K合格とはしない。
R608のevery0 / short-read0設定は機構を分ける初期screen案であり、最終機能の代替ではない。

## 数値補正

GDN payloadは1checkpointで117669888 B = **112.21875 MiB**、6個で706019328 B =
**673.3125 MiB**。R608原文の112.17 / 673.5 MiBは丸めを補正する。
PLE/QSA metadata、prefix IDs、vector capacityは別に必要で、host RAMの値である。
独立会話の全KV snapshot容量や解放済みVRAM量を意味しない。

原R608は変更せず、根拠source/researchのhashを`agent-cache-source-review-v1.json`へ記録する。
