# Canonical bank案 R707のMain修正

2026-10-15。R707はread-only FINALで、次のcanonical weight bank案のsource inventoryとして受領。
Mainは実metadata全48層、pack indexのnon-layer行、generate.cppのPLE bindingを再確認した。
報告の数値/owner表現に以下の修正が必要。研究報告の原本は変更しない。

最大のcanonical+native denseはlayer1:95483776+46540800=142024576 B。
layer47の77012992 Bはその最大ではない。expert全層最大1363148800 Bと独立bankにするなら
合計1505173376 Bで、既存capacity表のseparate-max-expert/dense caseに一致する。
1440161792 Bというユーザーのcombined-layer仮定は、expert領域も実際の各層サイズに合わせて
一つの共有bankへ詰める場合の最大である。dense bankだけの共通化では達成しない。

non-layer canonicalは以下で、計13148160 B。

| tensor | bytes |
| --- | ---: |
| output_hc_down.weight | 6553600 |
| output_hc_norm.weight | 40960 |
| output_hc_up.weight | 6553600 |

output.weight/token_embd.weightは現在packではpayload0のGGUF metadata行。
NativeHead/NativeEmbedが実payloadを別ownerで供給する。MTPのtarget embedding読み取りもこれを保持する必要がある。

`generate.cpp`のss.ple.w.key_bf16/value_bf16/norm/conv fieldsはwt.find(...)->dataへの借用pointer。
PLE weightがsessionの独立allocationという解釈は誤りで、canonical arenaをfreeすれば失効する。
全PLE fieldをactive layer overlayからprivateに構成するか、明示して別allocationへ移し予算へ加算する。
96 HC norm3932160 Bのlookahead対応と、non-layer ownerも扱う。
entryのfull ownerを残してbankを確保する間は、確保bank分だけFullよりpeakが増す。
「bankが小さいからentryのpositive peakが無い」というR707の説明も採用しない。
全restore時はfull ownerとbankの重なりがあり、steady-stateだけではadmissionできない。

上記訂正とimmutable commit1025e0edを入力として、実装担当Solへ別dir
`prefill-canonical-dense-bank-v1`のcaller-connected案を依頼した。現在のnative-only product/build/GPUは変更しない。
元loaderの変換済みstageからstartupにRAM snapshotを作り、D2H snapshotを避ける方針。
graph退役/復帰を既存の一つのouter transactionへ統合し、default/native-only挙動を保持する。
これは未実装の依頼であり、容量や速度改善を達成した証拠ではない。
