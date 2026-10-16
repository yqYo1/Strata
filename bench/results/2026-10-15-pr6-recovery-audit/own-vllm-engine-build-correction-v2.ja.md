# Integrated native16 build boundary v2

Source5020c05eの初回全体buildはFAIL。通常のcompiler exit1、全owner回収済み、GPU実行前、fresh fault0/dumpなし。原receipt own-vllm-engine-build-v1.json(SHA55540948f8979db0c46a6d0cfad8a99a547300a6ea62d1e30e9ec99d1cb0a0e4)は変更しない。

原因は-fno-sycl-rdcがSYCL_EXTERNALを意図的に未定義にする一方、pinned TLAがSPIR-V builtin宣言にそれを使うこと。公式Intel LLVM UsersManual345-353とdefines_elementary.hppを確認した。macroの強制定義やTLA/mathの変更は行わない。

修正は通常RDC objectをnative TUだけ生成し、plain -fsycl-linkでBMG G21/GRF256 native registration objectを作る。fsycl-linkはdevice wrapperのみで元host methodsを含まないため、clang-offload-bundlerでhost sliceを別に取り出して、両plain host objectを最終linkへ渡す。公式driver test sycl-offload-old-model.cpp356-418、CompilerAndRuntimeDesign395-424に沿う。=image/earlyはFPGA用のため使わない。既存engineのgeneric/JIT targetへAOT/GRF指定を伝播させない。

新controllerは同じ失敗buildのobjectを明示的に再利用し、Ninjaで変わったflags/依存を再buildする。原FAIL receiptを改変せず、全source/tool/sourcecommit/commandを新recordへpinする。build-v1は正しさの合格gateには使わず、既に閉じたcompiler失敗からの手動レビュー済み継続根拠のみ。実compilerでのmixed link、image選択、GEMM/QSA/model/速度/256Kはこれから実行し、この記録からPASSとみなさない。
