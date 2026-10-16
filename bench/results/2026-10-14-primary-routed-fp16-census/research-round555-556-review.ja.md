# R555/R556のMain確認

2026-10-15 JST。両reportの全文とcurrent rootの該当sourceを読んだ。
同じthinkingON suiteが実行中であり、client/source/recipe、GPU設定を変えていない。
新しい実測、profiler、CPU fixture、build、cleanupは行っていない。

`STRATA_VERIFY_PROFILE`はSYCL buildでは明示的に無効にされる。
normal queue recordingでは既存compatのcaptured eventを個々のkernel clockとして
読むこともできない。MainはIntelの現行graph extensionのprofiling property、queueとの
組合せとgraph envelopeの定義を確認した。graph profilingは一部最適化を無効にするので、
その結果をquiet graphの通常速度として扱わない。

R555のUnitRace候補はdevice timeを得る方向として残すが、`--start-paused`の指定だけでは
one-window収集にならない。collectionをprocess内で開始・停止できる実際のAPIと
既存waitを変えずに置ける境界をR558へ読み取り専用で依頼した。
host observer、graph envelope、device kernel samplesをそれぞれ区別する。
profilerのlocal runtime対応やnode attributionは未確認である。

R556のresolver sourceでは、まず全queryのresident hitsをcurrent epochで保護し、
その後missesを置く。`ctl[2]`はneedではなく成功したplaced数である。
同じ未配置pageの再lookupは`-2`なので、lookup countとhit+miss countは一致しない。
四つのINT8 runのpage payloadは4224 B。8192-slot設定のcopy launchは8192 groupsで、
copyの前にdevice mapping nodeが同じqueueへ入る。

選択pageの出現数、query間で重複を除いたpage数、実misses、copy payloadを混ぜない。
clock scanはreference bitをclearingしながら進むため、全slot scan×missesを常時の仕事量と
仮定できない。reportのamortized式は保守的なsource boundとしてのみ保存する。
一つのgroupが複数のmiss pageを担当するfixed grid-stride案は、元のmap/epoch/countersを
保ち、dynamic `ctl[2]`をdevice上で読むなら検討できる。しかしlaunch extentが小さくなる
ことだけでは、現在のdecodeの時間短縮やボトルネックだとは判断しない。

mapped host-USMのremote accessは通常memcpyとは異なる経路である。
source address payloadをそのままPCIe trafficやCPU DRAM trafficと呼ばない。
host poolsはqueue/graphのconsumer完了まで生存し、CPU mutation/freeとの競合を避ける。
captured pointerの所有期間を維持した上で、後のdevice timelineと照合する。

今回作成した`../2026-10-14-long-generation-benchmark/finalize_upload_fix_suite.py`は
post-terminalの結果整理用sourceのみで、未実行・未検証である。shared measurement lock、
ownerのpre-reap closure receiptとPID/start identity、元のfailed status、個別のrequest hashを
確認し、元receiptをbyte-identicalに圧縮保存する設計。自動cleanupは含めない。
R557に独立したsource reviewを依頼した。Mainの直列検証はsuiteが閉じた後に行う。

次の順序は変更しない。thinkingONのbench検証を完了し、その後にPR #6と公開Codexログの
600 t/s超を調べ、実装のlifetime/bounds/queue契約を確認して高速な部品を回復する。
新しい微小最適化候補の実装でこの確認を先送りしない。

Source pins:

- `src/core/verify.cpp`: `640e0270be833ddff0218331a7db18043605af3ced2682787f6b06abdcf6316c`
- `src/sycl/compat.cpp`: `225e2021b5ce67b3bb58f116a4a15cd5d592a33dbdf2c9837940a3e89155dca7`
- `src/kernels/sycl/kv_stream.cpp`: `d935fcc1ba07f5e3982cd255e3ad53e0f375ec16425a92b049696f142575e76c`

Primary references:

- [Intel SYCL graph extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc), profiling property and queue interaction, Main checked 2026-10-15 JST.
- [Intel PTI UnitRace](https://github.com/intel/pti-gpu/blob/master/tools/unitrace/README.md), source collection plan remains unqualified locally.
