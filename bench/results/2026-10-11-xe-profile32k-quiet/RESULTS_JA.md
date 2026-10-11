同じsource/binary/boot・32768 token入力・4096 chunk・MTP4・実144 cache slotでphase-on 1プロセスとquiet 3独立プロセスを測定した。個別の測定値、平均、標本標準偏差、元の結果はsummary.jsonと各record.json。合格した4件はQUITで正常終了し、新しいGPU fault/dumpはない。入力32768のbatch処理は32767位置、生成は16token。quietのprefill平均は216.164 tok/s、cold decode平均は8.605 tok/s。cold decodeには通常の初回graph capture等を含み、RESTORE/warm decodeではない。

phase-onのGPU timelineは248866msでhost grouping38.6%、dequant42.9%、wait copy0.4%。host groupingにはchecked stream_syncとtimer.foldのprofiling queryが入り、dequantにもhost/copy idleが入る。exclusiveなGPU演算/PCIe待ちの割合ではない。quietとの差にはparameter validation/loggingの差もあり、原因別の費用として扱わない。

最初のquiet v1はmodelの推論・QUITを完了したが、観測側が有効化していないvalidation layerの読み込みも必須にしてFAILEDになった。FAILEDを維持して平均から除外した。v2は必須runtime6個のpath/hashを全て要求し、任意diagnostic layerも実際に読み込まれたものはpath/hashを検査する。未知runtimeは拒否する。CPU側で3 positive/10 negativeと、元の実7-library mappingを検査し、別の3モデルプロセスで再測定した。model source/binary/argv/environmentは変えていない。

現在のring override8はSTAGE8と同じなのでstream_allはOFF、routed nonresidentだけをstagingする。全native arenaから3expert/layerを引いた約50GB/chunkは別のall-streaming policyの条件付き値で、現在の転送量には使えない。実際のper-layer transfer bytesが必要。6/6.3GB/sを仮定したtraffic-scopeの値は実測rooflineではない。

prefillとdecodeの採用は別に判断する。本家単体の最適設定探索は[TODO](../2026-10-11-upstream-parameter-audit/TODO.md)の通り未完了。本家の最新性能、性能採用、256K全context/RESTORE/global qualityの検証完了は主張しない。選択LP/top5と16tokenだけの一致は、広いモデル能力の評価ではない。
