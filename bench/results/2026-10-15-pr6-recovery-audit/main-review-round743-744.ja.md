# 出自再確認と実IGC dumpのMain補足

2026-10-16 JST。XeStrataは性能参考のみ。実装・helper・test・診断・保存sourceの
コピー・移植・翻案・そのsourceを下敷きにした再実装を行わない。規約とproject-only skillは
5942f185でこの制約を反映済み。既知由来・未確定由来の診断source86 fileの撤去記録を保持する。

R743が未実施としたselector専用化のqualificationは、proposal manifestの受領時点では正しい。
その後Mainのbuild/full3 process/clean18 processは実際に閉鎖してPASSした。
`own-prefill-selector-scope-v2.ja.md`と元receiptが現在の根拠であり、proposalの過去状態は書き換えない。
scoped binaryの実モデル・clean全体速度のqualificationは未実施のまま。

NLL v2のLGPL表記について、実装担当から次の出自説明を受領した。
担当はproject-only skillの先頭のSPDX二行を、新規fileの慣例と誤認して流用し、
root LICENSEを確認していなかった。rootはMITで、skillの表示は新規product codeの指定ではない。
担当の説明では、外部LGPL実装・template・XeStrata source・旧NLL実装を読んだり流用したりしていない。
実装の設計参照は現在の`src/program/generate.cpp`、`prefill_logits.hpp`の排他作成/公開、
`draft_uploads.hpp`のdrain lease、現prefill/native head/core layerとbuild定義。
したがって「独立」はXeStrata/旧NLLに依存しない意味であり、現project codeからの独立ではない。
担当は現在base全体の歴史的出自を独自に保証したとは述べていない。

元v2は未適用・未build・未検証のまま保持する。表記を消すだけで出自の確認に代えない。
新しいMIT整合案を作る場合は、この説明・元v2 hash・変更範囲と現source baseを記録し、
Mainが全diff、参照したproduction fileの出自、寿命、品質・機能を確認する。
LGPLという文字列の有無だけでXeStrata由来や非由来と判断しない。

R744はRA binaryをopaqueとして扱う読取り調査で、未発見のpublic schemaが存在しないとの
証明ではない。元capture v1は正常実行・出力byte一致・実loaded libraries一致・fault0だが、
取得物は11 binary RA metadataとSIPだけ。これからGRF/SLM/scratchの値を推定しない。
official IGC docsのShaderDumpRegexFilterはfile名を対象にする。v1の裸symbol完全一致regexが
ASMを得られなかったため、次はfile名にsymbolを含むregexと実libraryにあるZEInfo/ELF flagを
使う新しいbounded captureを別receiptで行う。元v1 recipe/receiptは変更しない。

公式compilerがXeStrata directory配下に配置されていることとXeStrata product sourceの採用は別。
実build resolverはこのforkのsourceと公式runtime/compilerを使い、path/hashを記録する。
この限定した監査を、全履歴・全外部依存の包括的な法的保証とは呼ばない。
