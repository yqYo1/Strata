# 私設 hwloc 依存の修正とコンパイラの再開

ソース製 Intel LLVM 7.1.1 の host ビルドv3は、hwlocの生成ヘッダーの探索不足でexit1となった。v4はヘッダーの私設参照先を追加して先へ進んだが、libumfのリンクで-lhwlocを発見できずexit1となった。両方とも全子プロセスの終了を確認済みで、元の失敗ステータスを維持する。GPUは実行していない。

v5ではDebian開発パッケージの私設libhwloc.soリンクの欠けた参照先を、既存の同じlibhwloc.so.15.7.0へ結び、GCCのLIBRARY_PATHに私設ライブラリディレクトリを追加した。ヘッダーの正確なhashと参照先も再確認する。システム領域には書き込まない。先行ビルドのソース・patch・構成を確認し、残りのコンパイルを再開した。作成時点では進行中で、インストール成功はまだ主張しない。

[manifest.json](manifest.json)にclosed失敗のcompact receipt、各診断、再現controllerのhashを記録した。activeなv5のreceipt/logはコピーせず、削除も行っていない。現在のconsumerはcompiler installおよびXe e32修正版のfree/contrib-llvmビルド。メモリ安全性・境界・新runtimeの確認が終わるまでモデルの性能測定には進まない。
