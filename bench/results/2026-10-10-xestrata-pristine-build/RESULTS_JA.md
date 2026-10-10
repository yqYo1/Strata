# XeStrataの無改変ビルドと測定条件

ghqで独立して取得したXeStrataの39bdadcc9e2b89b1e3c8be7bb2a603b04fa0e197
（xe0.1.40.2.1、fork元0.1.40.2/e8ca9af）をworktreeで無改変ビルドした。
IntelLLVM2026.1.1、contrib-icpx、Release、GGML3cf03257を使用した。
configure/buildは正常終了し、所有した全プロセスを回収した。モデルとGPUの実行はこの時点では未実施。
これはビルド成功であり、推論・品質・速度の検証結果ではない。

## 目標と比較条件

ユーザーの2026-10-10の指定により、prefill1000 tok/sの達成判定は約64Kの実入力
（予定65536 tokens）でもよい。32Kを固定目標長にしない。長い入力で初期読み込み等の
固定費を償却する条件を採用できる。双方で同じtoken列・入力長を使い、比較は最低32768 tokens。
入力fixtureのハッシュ、実際に読み込んだ全token数、再利用数0を確認する。
計時範囲を示し、起動・ロードを含む時間とrequestのprefill時間を分けて記録する。
1000 tok/sで65536 tokensを処理するrequest時間の予算は65.536秒。
prefix cacheで読み込まなかったtokenをfresh-prefillの分子に含めない。

decode70 tok/sは別に評価し、複数回の個別サンプルと分布を保存する。
prefillの利益だけでdecodeの変更を採用しない。採用前にはphysical262144位置を
実際に消費するcontext境界・状態復元・容量の検証を別途行う。

初回は32768入力の診断で安全性とprotocolを確認し、別のcleanプロセスで64K比較へ進む。
同じ設定値でもforkのring override8はsourceで16にclampされる。現在実装のeffective8との
差を明記し、設定文字列だけを見て同一のメモリ配置とは扱わない。

## CPU側の確認

source-onlyの初版controllerにはPP総数を32767と判定する誤りと、fault-filterの
閉じ括弧欠落があった。実装sourceでは総数32768、batch最終位置32767である。
失敗したCPU準備と原文を保存し、rootのv3で修正した。現行のcanonical32768入力に
対する受理、欠けたDONE/PP/LP、再利用、非有限値、範囲外token等、20ケースを確認した。
ビルドreceipt、binary、source、compiler、compile inputs、GGMLのidentityも照合した。
初版・失敗v2を推論に使っていない。

forkにはこちらの独立した全head/live-state dumpがない。初回のID/LP比較だけを
数学的同値や品質同等と呼ばない。grouped GEMMの小さなqualifierを別途準備している。

並行調査は[registry106](../2026-10-10-parallel-round83/report-registry-v106.json)に保存した。
R288のGPU種別の誤りは原文と別の訂正を保存した。Qwen開発元の残差FP8保存の記述は
低精度保存の候補を検討する根拠だが、この実装での劣化率はまだ測定していない。
