# SYCLの未使用workspace削減と8Kチャンク

2026-10-15。`src/prefill/sycl/gemm.cpp`を全体確認した。
`init_external`と`rebind`はworkspaceのポインタを記録するだけで、GEMMはそのポインタを
oneMKLへ渡していない。standalone `init`も既にnullptr/0で呼んでいる。
従ってSYCLでは専用workspaceを0とし、init/relayout/bytes_neededの三箇所で
`Alloc::take`を省略した。payload33554432 Bとguard256 B、計33554688 Bを削減する。
GEMMの計算、FP32参照、重み形式、stream順序は変更しない。

`build_prefill_no_workspace_v1.py`でproduction prefill objectを再compileし、
直前のhead-release本体の全libraryへリンクした。古いprefill objectがarchiveから選ばれて
いないことをlink mapで確認した。元binary/archivesは不変。
`prefill-no-workspace-build-v1.json`にsource差分、全source/input/compiler hash、
実compile/link commandと正常終了を保存した。新binary SHA256は
`5de4d2695a3226f037ab8eb9330da0b33b5e9a4761598837c51bb5e93c0fbe99`。

`run_prefill_head_c8192_r0_v1.py`は正常完了した。head退避を有効にし、C8192/FP32/R_GPU0で
公式bench.shの実39270入力/readを二回処理し、両方24 token生成/finish=length/reuse0。
native/server全exit0、全owner退去、同じboot、新規fault0、dumpなし。
`prefill-head-lifecycle-c8192-r0-v1.json`に個別結果を保存した。
実carve2840054016 Bは元C8192 streamedの2873608704 Bから33554688 B減った値と一致した。
初回/二回目の両head+MTP expert退避後freeは2042560512/1990725632 Bで、
cache1363148800 Bと512 MiB余裕を残して実C8192を二回実行した。
headの復元、MTP full graph再capture、生成、通常終了まで成功した。
診断PP229.8/234.8 tok/sだが、parameter validationとSTRATA_TRACE付きなので
clean速度値にはしない。後続のclean三回用は
`run_prefill_clean_c8192_r0_v1.py`で、元cleanと同じ固定workload ID、65536以上、thinking ON。
この比較はchunk/head退避/保存配置の組合せであり、chunkだけの効果とは呼ばない。
実RESTORE、cancel/throw、物理262144、品質と速度採用は未合格である。
`retire_prefill_head_diagnostics_v1.py`で閉じた単体/C4096/C8192の成功API/chunk/journalログ
375210 Bを削除した。全層の順序・完了chunk数、必要なAPI/memory結果とlog hashを
`prefill-head-diagnostics-compact-v1.json`へ保存し、path/hash/理由は
`prefill-head-diagnostics-retention-v1.json`に記録した。元receiptのbytes/statusは不変。

## 元の容量試算との差

`prefill-capacity-plan.ja.md`の元表はworkspace32 MiBを含むsource版の計算である。
他の仮定を固定し、この33554688 Bだけを除いたfull-resident workspaceは
C4096で1352452096 B、C8192で2563228928 Bになる。
KV/hidden予算はそれぞれ7201798144 B、5991004928 B。

| chunk | KV | KVと全hiddenを同じCで保持する計算上限 | 256K KV保持後のFP16 hidden行数 |
| ---: | --- | ---: | ---: |
| 4096 | K8V8 | 206020 | 166355 |
| 4096 | K8V4 | 224520 | 203219 |
| 8192 | K8V8 | 171384 | 107234 |
| 8192 | K8V4 | 186772 | 144098 |

これは一層weight bank、decode/MTP重みなしという元の仮定を保った差分計算であり、
現serveがこの配置を実装済み・最大contextで物理確保済みという意味ではない。
driver/oneMKLの512 MiBと小owner16 MiBは引き続き仮置きの余裕である。

## 新しい調査のMain訂正

R686は全層HC normを常駐するなら3932160 Bと数えた。このpayload総数は正しい。
ただし「両halfで次層normが必要なので追加81920 B」とした部分は誤りである。
実sourceはhalf0が同じ層の`hc_ffn_norm`、half1が次層の`hc_attn_norm`を読む。
厳密なlookahead追加は一つ40960 Bである。all-layer poolへ切り出すなら、active bankに
含めていた二norm81920 Bを二重計上せず、差分payload3850240 Bとして扱う。
小owner予算の内訳を明示するか、この差分と実allocation guardを別途控除する必要がある。
R686原本/hashは保持する。prefill専用view/graph retirement/再公開は未実装である。

R687はCH16 attentionを提案したが、Mainはそのままの実装へ進めない。
現QK mappingは4 subgroupで二つの16-cell tileを扱い、CHだけを16へ変えると後半tileが
範囲外になる。実mode1/Indirectのaccessorを再集計するとCH32は60880 B、
CH16へ正しくmappingを変えた仮定で46928 B（差13952 B）である。
R687のks/vs/state内訳とCH16合計は誤記として扱う。実physical SLM/occupancyではない。
既存Direct1は同じ実model profileでattention44.066→29.805秒の証拠が既にある。
そのclean比較はDirect0側の過去停止で未完了だったため、別の未測定CH16案を先に採用せず、
現在の安定sourceで既存Directのclean/長context比較を完了する候補として残す。

Solの`prefill-scratch-alias-v1`は四つのN-wide bufferを一つにするsource-only候補である。
Mainは全読書き箇所、PLEのl==1境界、samequeue、half/BF16のGEMM入力分離と
count/carveの変更を確認した。R688の独立レビューも明確な衝突を見つけていない。
削減量はC8192で251659008 B（payload240 MiBとguard768 B）。
短入力も同じ配置で動く設計だが、まだ未適用・未build・未実行である。
mode1 Direct比較とともに、次のserial検証候補として保存する。
