# R737/R738 Main review

両調査はGPT-6 Luna、読み取り専用、reportのみ保存。Mainは両報告を全て読み、
実source・compiler header・実測へ照合した。XeStrataの実装は参照していない。

R737はsource/API上の具体的な阻害要因を見つけず、decorated member pointerのcompileを残した。
Mainの実SYCL buildでそのcompile/link gateは閉じた。独立6processの全96ケースもexactで閉じた。
ただし資源使用量や速度は別で、pitch40は独立3sample中央値で61.40%遅く不採用。
reportのsizeof記述は、`sizeof(PaddedK)==10240*sizeof(half)`が単独でhalf2Bをassertする意味ではない。
半精度型の仕様と実compilerを別に確認する。原報告は改変しない。

R738のweighted4-bit MSD threshold案は、現行のNaN/zero順序・tail重量・同値の低ID優先を
保持する具体案。Mainはqsa_select.cppの全32bit threshold/gt/eq/scan/emissionへ照合した。
実装agentには次の訂正も含めたimmutable source-only依頼を渡した。

- Oracleはkey降順/ID昇順で選んだ後、選んだIDを昇順へsortしてdevice出力と比較する。
- 提案のprefix一致判定前に各keyを読むので、各roundの全B scanは残る。
  compactしない実装で1.0667Bへloadが減るという記述は誤り。
  8round+最終2scanの10Bという最悪値は、この提案の通常のsource load数でもある。
- 実long layer-majorは78452 token。qsa_step_fillはpos+1をNKvへ入れるので、
  その最終rowはn78452/tail0。原報告のn78453/tail1は有用なsyntheticケースだが実最終rowではない。
  両者B19614というslot数は同じ。
- APIのnは非負int32で、一般の重量和上限はINT_MAX。physical262144だけに狭めた新しい
  validationを入れず、既存domainのoverflowと判定を維持する。
- selectionの24399.6ms区間はscoresとtopkの両方を含み、非排他的なqueue観測。
  selector単独の実service時間や改善上限ではない。

source-only radix4 packageはこの時点で未受領・未適用・未build・未GPU。
次に全差分をレビューしてMainで部品測定する。原R738は訂正せずhashを保存する。
