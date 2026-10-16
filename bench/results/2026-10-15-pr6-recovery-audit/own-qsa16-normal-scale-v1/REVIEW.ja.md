R844 は固定ベース `6354f0b4ff3731466795410e11ca9ef785c9d326` 用の提案。製品には未適用。コンパイル、syntax check、テスト、GPU、ISA、速度、品質、fault、physical262144、RESTORE はすべて **UNRUN**。Main の qualified source/binary は変更していない。

変更は typed BF16 attention の score rescale 選択、その共有 inline helper、既存 fixture への追加だけ。`STRATA_SYCL_PREFILL_NATIVE16_QSA_SCALE_FAST` が厳密に `1` の場合だけ ON。unset、`0`、その他は OFF。最初の有効な非空 typed 呼出しで host が一度だけ読み、submit 前に `launch_prompt_bf16<true/false>` を選ぶ。環境は親が exec 前に固定し、実行中に変更しない。OFF specialization は元の `sycl::ldexp(value * (ks * .0625f), -shift)` を保持する。

厳密域の証明：binary32 の正常数を `x = (-1)^s (1 + M/2^23) 2^(E-127)` と書く。`1 <= E <= 254` かつ `1 <= E+n <= 254` のとき、`x*2^n` は同じ符号・仮数を持つ正常数であり、丸めを要しない。従って指数欄だけを `E+n` に置換すれば厳密。全仮数を網羅する代数的証明であり、fixture の標本から全域を推定していない。実装は `1-E <= n <= 254-E` を先に検査し、加算は成立した枝だけで行うので、任意の int シフトでも判定中の加算 overflow はない。

正常数以外、または出力指数が域外なら、元の float 値と int シフトをそのまま `sycl::ldexp` に渡す。±0、非正規数、NaN/Inf、subnormal/zero への丸め、overflow の処理を置き換えない。score の二つの乗算は括弧も順序も保持し、その結果だけを helper に渡す。qdown の事前計算、乗算へのシフト移動、fast-math、FTZ、compiler flag 変更はない。例えば `2^100 * (2^-24 * 2^-4)` の後に `2^-126` を掛けると `2^-54` だが、shift を scale 積へ移すと途中で underflow し得る。

旧 FP32 実装の prefix は `base/src/kernels/sycl/prompt_attention.cpp` の先頭から typed 部分の直前まで byte-identical。typed launch は template 宣言と score の constexpr 分岐だけで、SLM、WG/SG、barrier、matrix collective、FP32 accumulation/softmax、query scale、K/V、selector、範囲検証、shared state、出力 narrowing は維持する。新しい Runtime/context/queue はない。既存 fixture の `run()`、FP64 oracle、許容値、全ケース、CLI と判定は byte-identical。追加された scalar fixture も同じ Runtime/queue を使い、所有 buffer より後に Drain を作る。

追加 fixture は正常入力の両符号 × 254 入力指数 × 8 仮数 × 254 出力指数 = **1,032,256** 組を、host の独立した FP64 `std::ldexp` と raw bits で比較する。仮数は全数列挙ではない。全指数の隣接域外・最小 subnormal/zero への丸め境界・さらに外側・INT_MIN/MAX と、±0、正負の最小/中間/最大 subnormal、Inf、quiet/signaling NaN と payload を加えた **36,880** 組は旧 device `sycl::ldexp` と raw bits を比較する。元と同じ乗算式、正負の FP16 scale の最小 subnormal/最小 normal/最大 finite/zero を含む **4,180** 組も追加。これは精度を緩める tolerance 変更ではない。各 assertion の PASS は実行時のみ出る。

Main 提示の背景値は BF16 bulk median 394.9 tok/s、typed QSA 384.8 tok/s（各独立 clean39K 3 processes）。診断では attention interval 36.2963 対 prior FP32 attention 33.9610 s、projection 1.20636 対 0.919497 s、host outside wait 19.1927 対 19.1238 s。worker の測定ではない。これらの非排他的窓を exclusive GPU service や ldexp 原因の証明とは扱わない。既存 compiler が ldexp を同等の正常域分岐へ lowering 済みなら、本案の比較・整数命令・分岐は重複して遅くなる可能性がある。normal 域への実際の命中率、register/occupancy、divergence、生成 ISA は未確認。改善の可能性を限定して調べる opt-in 案であり、性能採用の根拠はまだない。

Main の検証手順は `qualification-plan.json` に exact argv/environment を格納した。現在の freeze を解いた後、完全パッチと preimage hash を確認して merge する。`files/` から live file を一括置換しない。新しい private build は `/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-qsa16-normal-scale-v1/build`。元の qualified build を上書きしない。configure は現行 receipt と同じ options、build は既存 `sycl_prompt_native16_test`、FP32 prompt、typed QSA、engine の target。新規 build/test は Main が lock と所有 process/deadline/log budget/fault 記録の下で直列に行う。

実行予定コマンドの中心は次のとおり（**すべて未実行**、完全な child 環境と出力パスは plan が正）：

```sh
cmake --build /home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-qsa16-normal-scale-v1/build --target sycl_prompt_native16_test sycl_prompt_attention_test sycl_native16_qsa_test strata --parallel 2
/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-qsa16-normal-scale-v1/build/sycl_prompt_native16_test --smoke --dump /home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-qsa16-normal-scale-v1/runs/smoke-off/output.bf16
/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-qsa16-normal-scale-v1/build/sycl_prompt_native16_test --dump /home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-qsa16-normal-scale-v1/runs/full-off/output.bf16
/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-qsa16-normal-scale-v1/build/sycl_prompt_native16_test --dump /home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-qsa16-normal-scale-v1/runs/full-on/output.bf16
cmp /home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-qsa16-normal-scale-v1/runs/full-off/output.bf16 /home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-qsa16-normal-scale-v1/runs/full-on/output.bf16
```

`smoke-default` は unset、`smoke-off/full-off` は flag=0、`smoke-on/full-on` は flag=1 の別 child。traced smoke は combined validation/tracing、full は clean の exact 環境を exec 前に設定し、各 mode 間で環境を切り替えない。全 112 full cases の正常終了、追加 fixture、既存 oracle、guards/immutability、fault0 を確認してから dump 完全一致を要求する。scalar fixture の速度からモデル速度・品質を主張しない。次に同じ binary/configuration の独立 long-input arm を反復し、全依存経路で比較する。品質、physical262144、RESTORE と既存安全 gate は別に残る。

出自は既存 MIT Strata と今回の独立記述のみ。R787/R818 の review metadata と research-index の関連 entry を確認し、knowledge collection 本文は未読。外部 engine/helper/test のコピー・翻案なし。既存 receipt の compiler install パスに別 fork 名が含まれるが、その engine source は読んでいない。`provenance.json` と `manifest.json` に hash と出自を記録した。package 以外の書込、Git、ビルド、テスト、process inspection、profiler、cleanup、他 worker への介入は行っていない。
