# 長文NLL停止: 外部事例の照合

Mainが2026-10-16に一次報告を確認した。いずれも今回と同じ原因だという証拠ではなく、環境変数やドライバーの変更は行っていない。現在の対処は長文dense-NLL経路をquarantineし、別の小さいhealth確認から順に進めること。

- [Intel LLVM #18424](https://github.com/intel/llvm/issues/18424) はGROMACS/A770・UHD770のL0 **v1**で、experimental submit/discard eventを含む待機停止。報告者はv2で成功し、後のコメントではcompute-runtime26.05で解消したと述べている。今回のforced URv2/driver1.17の直接の根拠にはならない。host synchronizeのspinと終了後のkernel faultだけではGPU計算そのものが停止した順序を断定できない、という診断上の注意に留める。
- [llama.cpp #24810](https://github.com/ggml-org/llama.cpp/issues/24810) はB70/URv2/旧driver1.15で、engine reset後のSYCL待機spin等を報告している。issueはbug-unconfirmed/staleでcloseされている。投稿者が提案するUSM poolやxe module設定の対処は今回の原因や安全性を立証しない。今回のresetはOwner期限の近辺であり、待機停止より先にresetした証拠がない。driver/system変更や旧workaroundの無検証採用は行わない。

installed adapterとの比較を行う場合は、そのバイナリに対応する実source revisionを先に確定する。ローカル公式intel/llvm root HEADはfbccb466cc122c294a6c746dcf0d37e45ff70f3cで、v2はunified-runtime/source/adapters/level_zero/v2にある。queueFinishにzeCommandListHostSynchronizeがあることは確認したが、root HEADと今回使用したcompiler/runtimeバイナリとの一致をここでは主張しない。

原因識別に必要なのは今回の実停止phase/stackであり、外部事例の症状類似ではない。次の品質screenではdense callbackを起動せず、既存selected full-vocabulary captureとCPU側の独立FP64 NLLを使う。これも元停止の修正完了や全面的な品質承認を意味しない。
