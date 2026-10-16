# MoE GPU fixture: 独立oracleへの修正

R974のOrca gpt-6-luna調査をnative Codex gpt-6-luna maxのR975でレビューした。
Mainは元report本文を読まず、review digestを使用した。engineの誤りではなく、
新fixtureのgather検査が同じsrc mapを期待値にも使い、最終BF16丸めで
source/expertの誤りが隠れ得る、というfalse-passの指摘を採用した。

修正は検証用`tests/sycl/moe_window.cpp`だけで、engineの6ファイルは
2b4399a0のmanifestとbyte同一。production route builderのhistogram/prefix/scatterと
別方式で、元token-slot番号をexpert IDでstable-sortし、期待source/expert/slotを作る。
公開planと期待表、全jobの連続範囲/被覆/expertをhostで照合してからGPUへ渡す。
両upload元のownerはchecked leaseより先に宣言する。

GPUでactual source/slotを独立表と照合し、gatherのFP16 bitとexpert番号を
witness生成前に照合する。Down全要素はtrusted expected slotを使って元IDから
直接求めたwitnessとcombine前に照合する。最後のcombine照合も残す。
map、gather/expert、Down、combineの4 counterを分け、negative controlが
狙った検査で検出されたことを確認する。実モデルの完全一致を要求する変更ではない。

既存negative-source/negative-slotに、最初のjobのexpertだけを変える
`negative-expert`と、Downの1要素のBF16 bitを1つ反転する`negative-down`を追加した。
後者はDown counterが厳密に1にならなければfail。unwindでは未readback counterを
数値結果として表示せず、checks_completed=0とする。全Bufferの前後256B canaryは維持。
GPU oracleのscalar算術/converterを共用するため、モデル数値契約やGEMM精度の
独立oracleではない。対象はproduction gather/combineを通る索引・row/slot配置である。

最大16Kケースのfixture所有device payload+guardsは算術上1053235728B。
これはruntime/JITのVRAM消費や実fitを含まない。host referenceはfixture専用で、
productionの2MiB host admissionや性能を変更しない。

修正後の`run_moe_window_fixture_syntax_v2.py`をnice19/O0/fsyntax-onlyで実行しPASS。
wall45秒/process CPU20秒/RSS2GiB/artifact8MiBの上限内で10.179秒、
peak観測session RSS341450752B、全子process終了/reap、errorなし。
旧v1の構文PASSは当時の版にだけ有効で、旧report/receipt/commitを保存した。
現時点でGPUの11ケースは全て未実行。codegen/linkも未検証。

次の実行順（Main管理・全て別process、全体の速度値には使わない）：

1. foreign CPU llama-benchが閉じたことを確認し、既存のfrozen paired profileを完了する。
2. `python3 -B bench/results/2026-10-15-pr6-recovery-audit/build_moe_window_engine_v1.py`
   をこのchild worktreeで実行する。新private buildにstrata/manual fixtureを作る。
   同じPID/startがliveなら副作用なしで拒否する。初回の拒否検査はd4804843に記録済み。
3. qualified ownerでmanual binaryの既定smokeをcombined UR tracing+parameter validationで実行。
4. validationのみでsingle/unwind/negative-source/negative-slot/negative-expert/negative-down。
5. hot8193、その後hot16384/balanced16384/tail16383。各caseにsalt1/37の2回。
6. それぞれnormal exit、owner全終了、fresh kernel faultsなし、dumpなしを確認。
   その後に実modelのwindow 0/2、state/PLE/HC/MTP/nonzero-position/cancel、
   physical262144/RESTOREを確認し、公式bench.shの32K以上・thinking有効・複数回比較へ進む。

private buildの実出力は`~/.local/state/strata-sycl/post-reboot-tuning-20261007/main-moe-window-engine-build-v1/build/`。
現時点では未作成。新controllerの重い段階を実行したという意味ではない。
古いB配下のbuild出力を新buildの入力に要求せず、source/toolchainだけをpinする。
XeStrata project sourceは除外し、既存Xe-path内のcompiler/runtime依存だけを使用する。
performance adoptionなし、PP600/1000およびTG70は未達。
