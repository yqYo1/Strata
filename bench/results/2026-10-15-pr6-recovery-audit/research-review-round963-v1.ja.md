# R962 report review (R963)

- 判定: 補足付きで受理。R962は11,958 bytes/1,718 wordsで予算内、22 source pins全一致。旧workspace削除、FP32 alias、capacity記録とは異なるnative16 lifetime/byte-planを扱い、既存案を再提案していない。
- 数値と所有境界は妥当。`Alloc::take`のguard、8 MiB count slack、GDN/QSA/MoEのmax、owned `cudaMalloc`群と借用arena、16K/262144/RESTORE等の未認定範囲を区別。実測条件はalias=1/no-prefill-borrowで、alias-OFF `xn`の335,544,576 Bは現行の物理節約ではない。
- 補正: QSAの8,192 token guardは`qsa_native16.cpp:34-36`（報告の25-31は参照ずれ）。layer 1 PLEは`m.region`を一時scratchに使い、最終float callbackのMTP batch経路も同regionを使う。後者は使えないと通常`mtp.prefill`へfallbackするため、再配置時はscratch容量とcallback経路を別に評価する。
- Mainの限定参照: 将来arena案を具体化するときだけR962の49-63行、`prefill.cpp:1038-1041, 2211-2250, 1085-1155`、`generate.cpp:6771-6776`を読む。本文全体やcollection本文の追加閲覧は不要。
- 次scopeは`schedulable_now=false`。新規調査よりMainの実装判断と直列runtime evidenceを先行。別件のGDN output16-only案はR962の対象外で、未適用・native compile/test未実施・性能未認定。
