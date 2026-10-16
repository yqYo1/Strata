# 次の候補のMainレビュー

## CPU GU再訪observer

`cpu-gu-reuse-observer-v1/source.diff`とheader/mockを読んだ。
CPU jobが実際に作られる時点のexpert IDを保存し、job順でissuerだけが更新する。
workerの新しいclock/atomicや、追加GPU readbackはない。
request-coldで48×512のnt別回数と4容量のpayload-only LRUを計算する。
これはモデルを跨ぐwarm cacheでも、実RAM admissionでもない。

Mainのlossless GU部品結果は4独立processで再現したので、再訪とfill回収を判断する
このobserverには現在のconsumerがある。次は未適用のpatchを照合し、独立vector LRU、
base64/ULEB、上限、job/nt完了のmockを通常・sanitizerで実行する。
その後、実際の製品OFF/ONで既存joined ledgerと照合する。
mockのOFF wrapperだけでは製品OFFの同等性を証明できない。
8GiB仮想budgetはlive MemAvailable/cgroup、metadata、8GiB RAM floorの合格を意味しない。

## prefill scratchのchecked retirement

`checked-prefill-retire-v1/source.diff`とREADMEの所有関係を初回レビューした。
既存の単なる`reset`では、releaseのエラーと古いImpl内のポインター/ownedを
再利用する問題が残る。候補は全prompt・callback・借用cache refill後にchecked drainし、
所有allocationを解放してfresh Implへ交換し、次のbatchの前に再初期化する。
Verifier/MTPのRとgraph、永続SessionState、KV、checkpointsは保持する設計。
compat freeのregistryからAllocationを先に消す現動作に対し、成功するまでownerを残す
専用free APIも含むため、通常のprefill編集より広いレビュー・検証が必要。
まだ適用・build・testしていない。1.55GiBの解放は未測定。

最初はsingleGPU/static cacheの実験だが、物理256K対応を短い入力で代替しない。
256K入力をcontext262144で使い切った場合、残り生成枠がなくなること自体は正常である。
full conditioning、末端状態、範囲外アクセスや解放後使用がないこと、正常な上限終了を検証し、
出力するために検証contextを縮めたり「256K設定だけ」で合格にしない。
借用あり/なし、複数request、RESTORE、checkpoint、cancel、normal/coupled MTPは別の未完gate。

## profilerの限定窓

`external-session-profile-smoke-v1`は未実行。
今回、無変更nativeへの公式`--device-timing`でグラフ再生756回と実モデルの660回を
実際に取得できたため、現時点のカーネル優先順位はこの実測を使う。
限定窓のhook/CLI handshakeは、共有カーネルのphase分離が必要になった時に再検討する。
未検証の限定窓proposalを完了したprofiler機能として扱わない。

各提案のunified diffとgenerator、manifest、README、独立mockをGitで保持する。
同じ製品ファイルの全コピーはpatchから復元できるためGitへの重複追加を避け、
今後のapply consumerが終わった時に保持manifestで整理する。
