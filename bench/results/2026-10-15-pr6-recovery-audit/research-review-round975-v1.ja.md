# Round 975 review digest

R974は対象内のstatic auditとして妥当で、10119 bytesのreport hashは指定値に一致した。新しいGPU fixtureを扱うため、engine統合を見たR971/R972や、補正を記録したR973とは重複が少ない。expected hash map記載の7ソースを再計算し、すべて一致した。

必須修正はfixtureのoracleにある。`moe_window.cpp:132-135`はgatherに使った同じ`src` mapから期待rowを作る。final oracleは元IDから計算するが、`native16.cpp:148-157`で重み付きFP32結果をBF16へ丸めた後なので、誤ったsource/expertの寄与が同じBF16値に隠れる場合がある。元IDだけから別方式で期待slot/sourceを作り、witness書き込み前にgather rowとexpertを照合する。combine前にはactual slotの範囲と期待slot一致を確認し、信頼できる期待slot上のDown値をoriginal token/expertのwitnessと直接比較する。これはfixtureのfalse-pass対策で、production routingの欠陥を示すものではない。

非同期寿命の懸念は確認できなかった。全ownerはleaseより先に作られ、upload・gather・combine・readbackは同一runtimeのin-order compute queueを使う。vector/route planとreadback先はchecked drainまで生存し、drain失敗時はowner解放前にfail-stopする。host pointerをqueue memcpyに使う点もKhronos SYCL 2020 rev.11 §4.9.4.3に合う。

hot8193とhot16384は各10 expertの8192+1および8192+8192 splitを直接assertする。balanced16384は式上512 expert各320 route、tail16383は上限より1行少ない。canaryはallocation端の書込みを検出するが、読込み範囲や内部破壊は検査しない。invalid source上限、実GEMM/router/state、物理262144 context、RESTORE、品質、速度はこのfixtureの対象外。

修正後、fixtureのsyntax確認とmanual targetのbuild/linkを行い、資源枠が空いたら traced smoke、single/unwind/negative-source/negative-slot、hot8193、最後にhot16384/balanced16384/tail16383を別processで実行する。foreign CPU benchmarkの終了とMainの直列resource gateが先。現状はsyntax-only PASSで、GPU runtime PASSではない。

Index登録案と全source/report hashは同じディレクトリの`research-review-round975-v1.json`にある。indexは変更していない。

