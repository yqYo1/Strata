# R569/R570 と layer-major 34K検証のMain判断

2026-10-15。目標1000 prefill / 70 decode token/sは未達。
直前のユーザー向け進捗回答は状態説明であり、目標に対する新しい検証ではない。
本境界では既に終了した34K実行とCPU事後変換の証拠を保存し、次の実装を絞る。

## 処理順変更

`layer-major-long-v1.json` は34,603入力、34,602 batched、4096行×9チャンク。
新しい処理順の全248,320 first logitsと541地点×10,240要素の最終層残差は
`natural-layout-long-pair-v2.json` のnatural0とビット一致。所有プロセスは正常終了、
fresh GPU faultなし。サンプルした最終層と初回logitsの比較であり、全層の状態、
長文decode品質、実入力262,144、RESTOREを検証したものではない。

専門家転送回数は参照の91,356から24,576へ減った。新候補は残差40,960 B/行のうち
4096行/167,772,160 BをGPU、30,506行/1,249,525,760 BをRAMへ保存した。
診断ログは196,135.9 ms / 176.4 token/s。validation、R readback、traceを含む単発のため
正式な速度採用判定に使わない。転送回数減少だけで性能回復を主張しない。

VRAM表示の文脈を訂正する。2343 MiBはstandalone prompt buffers確保後、1960 MiBは
standalone prompt実行後。後者を新候補の確保前の空き容量として扱わない。
実際のresidual planとその時点のadmissionを優先し、表示の差から回収可能容量を推定しない。

短い入力では512専門家×48層のロードが利益を生まないため、オプション有効でも
`n <= chunk`を従来経路へ送るdispatcher修正と、各層2行の進捗表示を追加した。
この修正のビルド/実行証拠は34Kの旧バイナリ検証と分ける。

## R569: 保存表現のみFP16

モデル著者の残差FP8保存報告は候補の優先度を上げる一次情報だが、format/scale、
当forkのIQ3重み、層間cast、評価条件の同一性を証明しない。FP16候補は保存/復元だけを
変更し、演算、再帰状態、KV、最終出力はFP32経路を維持する。既存device castの
丸め・飽和仕様はNumPyの理想IEEE変換と別途検証する。

`saved-residual-precision-screen-v1.json` は保存済み最終層5,539,840値に対するCPU事後変換。
元データ最大絶対値41.5853、FP16相対L2誤差0.00021125、非有限0、ゼロへのunderflow1。
BF16相対L2誤差0.00167661。この数値は層をまたぐ誤差累積、モデル品質劣化率、
全ての中間値の範囲を表さない。FP16の実装・数値・品質・速度検証はこれから行う。

R569の299件/1ppやPPL+0.5%は研究担当の条件例であり、ユーザー合意済みの必須基準ではない。
同一出力列を低精度採用条件にしない。logit分布/NLLと長文タスクの実用品質を比較し、
誤差と限界を記録して判断する。

## R570: decode計測

過去の未適用observerを現在の8ファイルへ適応する別の実装担当を起動した。
prompt全体の追加FNV走査と巨大histogramを削り、実際のT、W/R/D/A/C/E、
CPU job rows、issuer/workerの仕事量とCPU時間、正常な最終wait/commitを記録する。
observer寿命をpool reader終了より短くしない。CPU時間や重なるwall区間を
排他的なcritical pathの割合とみなさない。

Mainがソースレビュー、ビルド、CPU fixture、32K以上のcontextでの有限decode診断を
順次管理する。速度採用測定は計測無効の別実行で行う。

## 並行作業と保持

6.1 SolのFP16保存、decode observer候補は製品ソースを編集せず専用成果物へ出力中。
Direct比較ツール候補はFINAL受領、未適用/未ビルド/未測定。
Luna R571は過去レポート所在を共有した上で、RAM tailの順序・転送削減にスコープを
ずらして調査中。Mainの実行と並行して進め、途中介入しない。

`closed-artifact-retention-v2.json` は終了済み100ファイル、32,107,458 logical B /
33,120,256 allocated Bの削除記録。元の結果・失敗状態・fixture hashは変更していない。
実際の空き容量増加は未測定。natural0を参照とし、閉じた重複tensorと成功ログを退役した。
