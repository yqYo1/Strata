# ソース版LLVMとホスト検証

intel/llvm tag v7.1.1、commit `504366f4b82ff00bbac7b0c956635eed8bc599d8` にXeの17パッチを適用したコンパイラーを構築・設置した。表示バナーはDPC++ 7.1.0 pre-release / clang 22.1.0であり、タグ名と区別する。bootstrap v5は通常終了し、管理した全プロセスの終了と残存なしを記録した。過去v3/v4のhwloc失敗は前の記録に維持する。

ホストASan/UBSanの不足分を同じソースから構築した。runtime v1はCMakeの相対設置パスがレシピ所在を基準に解決され、実際のコンパイラーresourceディレクトリーにruntimeが存在しなかったため失敗。v2で絶対パスと対応するLLVM build-tree CMake packageを指定し成功した。v1を成功に書き換えていない。グローバルパッケージ・ドライバー変更は行っていない。

5つの実制御ヘルパー publication / commit transaction / physical full-context probe / MTP completion / context bounds を、ReleaseとASanUBSanで計10回実行し通過。GPUや private MTP の実計算ではない。

修正Xe e32をfree・Release・WERROR・native CPU設定で全体ビルドし通過。CPU CTestは10 PASS、AVX-512専用expert_multiはISA不足で1 SKIP。バイナリーSHA-256 `a07869d265c9214a3fba828fee5ed4b6c15bd42aa9fb89915fc3a729c9fb722d`。全依存・変更ソース・レシピ・コマンドと終了状態をreceiptに保存した。

contrib-llvm（oneMath/oneMKL ON）全体ビルドも同じ30ソースpinで通過し、CPU10 PASS・AVX-512専用1 SKIP。バイナリーSHA-256 `8e2b277acd1cce8f4283cb52d1c1fae0be4a2ccf904e58b4496117328c5ec0f1`。全体ASanUBSanビルドは開始済みで稼働中の結果をコピーしていない。新runtime能力照会のレシピは準備のみ。lint、GPU parity、通常末尾・cached再利用、262144セル実使用と保存復元、長入力比較が未完了で、実装の採用・新規速度はまだ主張しない。

原本の閉じた構造化receiptをそのまま保持し、不要な成功ログ・巨大ビルド中間物・重複runtimeバイナリーをGitへコピーしない。削除はこの境界で行っていない。設置LLVMと対応build-treeのsanitizer CMake packageには現行consumerがあり保持する。
