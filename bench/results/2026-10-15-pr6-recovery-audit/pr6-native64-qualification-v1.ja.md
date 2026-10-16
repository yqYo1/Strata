# PR#6復元: 実GPU上での64bit互換処理の確認

2026-10-16、Arc B570、driver1.17.39395+14、固定LLVM22/libsycl.so.9。
検証専用branch `verify/pr6-prefill-restore` の `0d659b126efc58763921a069a818f878259d54e4`。
本番の推論実装を置き換えた結果ではない。

前の全体ビルドは、歴史的DPCTのunsigned long long shuffle/shiftとLLVM22の
SPIR-V overload解決の不一致で失敗した。失敗結果・エラーの代表窓をそのまま保存した。
私有生成headerだけでnative OpenCL64型へbit_castする処置を加え、歴史的vendor原本や
kernelの数式・処理順・doorbell禁止は変更していない。新headerのSHA256は
`22709e0715e75621996b9eb289d507a11fcb2919770e161221cf74cd1cca9b21`。

モデルに入る前に、実際のpatched DPCT helperを呼ぶ独立targetをMainがビルド・実行した。
符号あり/なし2kernel、物理SG32、2bank、論理幅8/16/32、全maskで、9,088個の結果を
64bit全体で独立host参照と比較して全一致した。内訳はselect7,552、shift1,536、
境界self値356。物理mapping64個と実UR kernel提出2件も確認した。

device payload75,776bytes、upload2/readback1/明示retirement barrier2。
3つの明示holdでexec前環境・library maps・PID/startを照合し、exit0、所有process終了、
新規kernel fault0、dumpなし。成功判定はnative出力だけでなく外部receiptを含む。
環境は初期化後に変更していない。速度測定ではない。

`own-pr6-shuffle64-contract-v1.json` が実測の原本で、各ソース・ビルド・静的ELF確認、
環境、runtime、boot、journal境界、全比較数・終了結果を参照できる。
新target以外の112個のcompile argvと既存prefill最終linkはbuildroot正規化後に同じ。

この確認は固定SG32のhelper値の範囲に限る。歴史的WG128/1024の一般mapping、
非空のasyncエラー配信、モデル品質・全state・physical262144・RESTORE・600tok/s再現は
未認定のまま。次は全prefill専用targetのビルド/依存確認、その後に段階的なモデル検証。
