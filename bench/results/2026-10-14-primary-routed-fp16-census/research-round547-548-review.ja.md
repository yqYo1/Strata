# R547/R548: prefill中のdecode用VRAMと再構築の境界

2026-10-15 JST。Mainが現在のroot sourceを読み、研究reportの依存関係を確認した。
現在のthinking有効benchmarkには変更を加えていない。以下は未実装・未測定の候補であり、
再構築時間、実際のVRAM解放量、8K chunkのadmission、性能向上を認定するものではない。

R547の重要な判定は確認できた。MTPのfallback prompt captureは`record_forward(T, -1)`を
呼び、KV append後の`if (!full) return true`で終了する。canonical `Prefill::draft_kv`も
draft KVのみを作る。この二つのprompt経路はMTPのexpert計算とheadを実行しない。
従って、別allocationの`experts_`はprompt中に使われないdecode用weightである。
primaryの512 expertsとBLOB=1,382,400 bytesならsource-derived sizeは707,788,800 bytes
（675 MiB）。この計算は実際にdriverが返すfree VRAMの測定ではない。

immutable sourceはMTP packの`experts.bin`で、現loaderは64 MiBのhost chunkで読み戻せる。
decode weightの復元にVRAMからRAMへの保存は必須ではない。bounded RAM copyを保持するか、
元fileから読むかでdisk/read/H2D costが変わるため、その比較は別途必要になる。
head subsetとsamplerにもdecode専用allocationがあるが、まずexpert tableだけを評価する方が
変更範囲を絞れる。mixed `arena_`、mixed `dense_`、persistent draft KVとring metadataを
一括で解放する案は採用しない。

R548には二点の修正が必要だった。原reportはhashを保ったまま残す。

- 現在のSERVEでは`sp`をstartupで初期化し、request中に`sp.run()`して、貸したcache slotを
  refillした後そのままverifier decodeへ進む。正常request後の`sp.reset()`はない。
  `sp.reset()`はstartup allocation retryにあり、CLIの`prefill.reset()`とは異なる。
  従って「既存のPrefill reset後にrestoreする」というSERVE手順は成立しない。
  restoreがresident prompt workspaceと共存してfitするか、request毎に安全なreset/reinitを
  新たに設計する必要がある。後者には再allocationとcaptureのcostも含める。
- `MtpDrafter::bind()`は`window_R_`等を代入するだけで、cached decode graphを無効化しない。
  `capture_round()`は既存execがあれば直ちにreturnし、新規captureでは`window_R_`の現在の
  pointer値を`copy_from_mapped`へ渡す。SYCL lambdaはそのpointerをvalue captureする。
  `bind(..., nullptr)`や新pointerへのbindだけでは、旧verifier `R_`へのgraph leaseを解除
  できない。verifier workspaceを置換するなら、MTPの通常・coupled round graphも成功確認付きで
  retireし、新pointerからrecaptureするAPIが要る。R548の「既存bind protocolでdetach」は
  実装済みの安全なprotocolとして扱わない。

Verifier arenaはpersistent `SessionState`とは別のdecode workspaceである。ただし其の解放は
上記MTPへのborrowと、verifier window/work/commit graph、両queue、host callbackの完了を含む。
`wait_commit()`一つや既存destructorの呼出しだけでretirement成功とは判定しない。
sticky error時にはrequested wait自体が行われない可能性が既にR542で確認されている。
新APIはhealthy completion専用とし、失敗後に未知のgraphが参照するstorageをfreeしない。
これはGPU faultからの復帰処理ではない。

`SessionLoopScratch`の約100 KiBはpinned host storageであり、大きなVRAM回収とは扱わない。
其のfreeはserving threadのaffinityも戻すため、bufferだけの退避とaffinity変更を混同しない。
現在のMTP routeでは未使用のordinary NativeHitGraphs/SessionGraphsを省く候補もあるが、
graphのdriver resource sizeはC++ object sizeから推定できない。
target expert cache、SessionStateのGDN/QSA/PLE、native headとmodel weightsは保持する。

将来の比較はno-free control、expertのみ、Verifier workspaceのみ、必要なら組合せを分ける。
実free watermark、restore/read/H2D、graph recapture、prefillとdecodeの時間を別々に測る。
256Kの実入力境界を保持し、最後のprompt state、draft KV、最初のverify window、通常終了を
確認する。復元失敗はdecode handoff前に停止させる。この候補のGPU実験は現在のbenchmark
検証と、user指定のPR #6／公開Codexログの高速prefill監査・回復より先に実行しない。

Mainが照合したsourceのSHA-256:

| File | SHA-256 |
| --- | --- |
| `src/program/generate.cpp` | `e3ca2a964806270e1d72862f979b6e262fb17a925a6027e7534c81dc45418871` |
| `src/core/mtp.cpp` | `f5288a88c399d0611d13b7d418fa2d3268f05eabb9ff13a620fd317a5465a287` |
| `include/strata/core/mtp.hpp` | `6240600a0620805774473bfbec9cc1cf6071dd2fa4aaccfb8fb8d15c6351fc69` |

source anchors: `generate.cpp:4491,4549,4700,4861,5826,5853,6347`;
`mtp.cpp:416-451,455-531,664-694,826-896`;
`src/kernels/sycl/elementwise.cpp:86-90`と`include/strata/sycl/launch.hpp:85-96`。
