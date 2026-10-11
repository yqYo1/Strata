# 修正後lintとparity helperのホスト故障境界

source lint比較v4は通過した。freeのlint script exit 1はpristineにもある診断だけで、新規指摘0件。contrib-llvmのGemmは元ソース・現行ともexit 0、新規指摘0件。元のv1/v2/v3の失敗は書き換えていない。前のv3で実行したpristine freeを、同一compiler flagsと同じ元ソース・未変更CMake条件で再利用した。Optionsの余分なpaddingは136 bytesで同じ、Verifierは64から56 bytesへ減ったことを、完全な診断本文を残して確認した。

実production checked_transfer templateをhost C++としてビルドし、通常版とASan/UBSan版で各happy/standard/unknown/invalid、計8件を確認した。正常callbackは1回、caller vector全byte/checksumも一致。例外2種はbuffered stderrの前置・production診断をflushしてexit 1で終了し、caller destructor・atexit・after-call・caller catchは出なかった。正確な全stderrと終了statusを個別receiptに保存した。binaryはlibsycl/UR/Level Zero/UMF/oneMKLをリンクせず、Runtime/SYCLの操作、device query、GPUを使っていない。_Exit経路は終了時のleak検査を通らないため、leakの資格には使わない。

この確認はhost callbackの境界だけで、実submission/wait/async GPU、private oneMath queueやモデル数値を検証したことにはならない。最新source全体のhost ASan/UBSan free v6は閉じて通過し、CPU 10 PASS・ISA不足1 SKIP。contrib-llvm v6も閉じて通過し、同じくCPU 10 PASS・ISA不足1 SKIP。小GPU整数確認v3では64KiBのH2D/D2Hと16384個の整数値が一致した。続くモデルなしparity v2のdevice-selftest・BF16 GEMV・KV Q8・KV Q4・dense free・dense contrib-llvmの6件も、同一source/runtime・全owner閉鎖・新しいGPU障害なしの条件で通過した。GPUモデル性能、物理262144セル・保存復元、cached末尾、採用は未検証。

原本receipt・source・commands・hashと保持理由は[retention-review.json](retention-review.json)。全文の成功lint/compile/API logsやbinaryはGitにコピーしない。成功API履歴は個別結果、hash、結果token数、診断文の探索結果を検証・commitした後に整理する。

候補commitは`ea395598c8e967c632a955dc43b77a9291286d49`。release/host ASan/lintはcommit前のe32+overlayを測った元receiptのまま保持し、32件のsource pinsが候補commitのbytesと一致することを確認した。性能採用はしていない。[qualification-summary.json](qualification-summary.json)と[api-trace-summary.json](api-trace-summary.json)に確認範囲を残した。denseは各モード54回・34299出力値を独立したdouble oracleと比較した。oneMathのglobal path表示は型ごとのBLAS backend証明として使わない。

結果を436ec1e2でcommit・pushした後、閉じた7件の成功チェックから60件のraw text logを削除した。論理サイズ32,442,710 bytes、file allocation 33,382,400 bytes。直前・直後のfilesystem空きはともに1,022,114,791,424 bytesで、実際の空き増加は観測していない。元receipt・source・binary・fixtureは変更していない。各path/hash/理由は[deletion-manifest.json](deletion-manifest.json)。新規モデル測定用controllerはこの削除済みraw logに依存させない。
