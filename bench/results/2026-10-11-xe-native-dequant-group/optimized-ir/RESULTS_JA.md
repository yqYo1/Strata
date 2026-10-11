# Group pointer選択の最適化済みdevice IR確認

Mainがproduction compile DBのO3・数値設定・Werrorを使い、v1と定数添字APIv2の完全translation unitをdevice-only IRとして両モードで実際にcompileした。候補を別directoryから読むため、元sourceのローカルquoted header検索を同じ-iquoteで復元した。最初の試行はcuda_intrinsics.hppを探せずFAIL。原statusとerrorを保持し、修正controllerの4compileはPASS。kernel/math flagの緩和は行っていない。

|確認箇所|v1|APIv2|
|---|---|---|
|GU kernelのprivate alloca / memcpy|各1、776 byte copy|ともに0|
|Down kernelのprivate alloca / memcpy|各1、776 byte copy|ともに0|
|pointer選択|private captureへの動的GEP/load|引数の定数offset loadとswitch/phi|
|kernel引数のbyval構造体|776 byte payload|残る|

各候補のIRは、free/contrib-llvmの独立compileでSHA256まで一致。200の完全define bodyを比較し、変更は2つのgroup kernelだけだった。その他198はmetadata nodeの参照番号だけを正規化するとinstruction textが同一。metadata内容全体の意味的同一性を証明するものではない。decoderと旧個別関数のsource byte不変は別のMain source reviewで確認済み。

生成IRからprivate capture copyを取り除けたことは確認できたが、最終SPIR-V/IGC命令のcopy・scratch・spillやGPU時間への寄与は未測定。kernel ABIのbyval payloadも残る。APIv2のGPU改善、モデル品質や速度はまだ主張しない。v1のcomponent回帰と不採用判断は変更しない。production callerは未適用。

次は適用したv2で両モードのproduction release/host sanitizer/lintを確認し、新しいsource/binaryで18形式の全byte・guard・refusal・healthと同一componentを測る。有益なcomponent結果が出た場合にcaller lifetime・32K厳密出力・速度を扱う。full262144/RESTOREと本家単体の最適設定探索は未完了。

完全IRは最終lowering比較を行うMainの現在のconsumer用としてfreeの2つだけを外部stateに保持する。合計5MiB上限で、v2 component採用・拒否かlowering比較の終了時に再検討する。今回の実行ではGPU/サービス操作やsource編集は行っていない。新規agent報告はなく、索引v128の506件を参照する。
