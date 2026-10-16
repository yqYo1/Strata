# K配置変更の実機コンパイラ証拠

同一binaryのscore33とKpitch36を、小さな別processでIGC ShaderDumpEnable/RAMetadata有効にして実行。両方oracle/guard/正常終了/faultなし。各runの生成物は約12MBで256MiB上限内。性能測定は別のclean6process。

選択score33とK36は最終metadataがSIMD16/128GRF/eu_thread_count8。GRF spill sizeのheaderは双方なし。ただしpredicate flag spill store22/load28は双方にある。「spill皆無」とは言わない。score32の初回attemptには1920B GRF spill、別の再compileもあるため、その初回値を選択score33の状態へ流用しない。

static ISAは基準4977命令・候補4970命令。両方に336個のload.slm.d32x8t命令があり、DPAS系命令数/主要memory opcode数も同じ。これは動的実行数・時間割合・実機SLM bank/帯域/occupancyではない。global Kを連続化しても全依存時間が短くならないというclean測定（+19.58%）を優先する。

次の有限仮説は、局所Q/K/P/Vを16単位のdense tileとBF16 packed-Bへ並べ、row-major sparse-stride matrix-loadの変換/読み出しを減らせるか。精度/演算順/入出力/総local bytesを保ち、仕様確認・小さなtraced実行・独立oracle・同一入力timingで判定する。現時点では未実装/未測定。全体prefill最速464.1tok/sと600未達の状態は変わらない。
