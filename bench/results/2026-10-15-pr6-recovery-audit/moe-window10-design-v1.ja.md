# MoE window10の次実験

Mainが既存の自作Strataコードから独立に作成した変更案。XeStrata由来のコードは使わない。
現時点ではパッチは未適用で、速度・品質・物理262144コンテキスト・RESTOREは未評価。
進行中のwindow2/8各3回比較が閉じるまで、測定対象ソースを変更しない。

mode8は最大65536行、operator/jobは最大8192行で、今回の実入力78453 tokenの
batched78452行を65536+12916へ分割している。mode10を別のopt-inとして追加し、
最大81920行、今回なら78452行を一つのMoE windowへまとめる。
各expertの重み展開をwindow間で繰り返す回数を減らす仮説であり、実速度は未測定。
Attention/PLEは従来通り8192行以下の各chunkで進み、FFNとHC書込みをまとめた後に
各chunkのcallbackを元の順に呼ぶ。mode0/2/8、短入力のdecode配置によるprefill、
GPU prefixを超えるhost tailとcheckpoint要求時の従来処理を残す。

|実際のwindow行数|別所有FP16 mixed|共有arena必要量|
|---:|---:|---:|
|65536|335544320 B (320 MiB)|888410624 B|
|78452|401674240 B (383.0664 MiB)|1022996480 B|
|81920|419430400 B (400 MiB)|1059131904 B|

共有arenaは全17planeそれぞれにround_up(payload,256)+256の予約を加算した値。
一部windowは単純な行比例式と一致しない。最大mixedの増分は80 MiB、今回の
78452行では66129920 B。上限81920であっても実際の予約は
min(cap,GPU-resident rows)*2560*2 Bとし、固定400 MiBを常に予約しない。
既存residual領域のaliasやFP32への保存復元を追加しない。

既知の空きVRAMからlayer cache、BF16 KV prefix、residual、staging、512 MiBの
headroomを引いた残りにmixedが収まる場合だけ有効化する。従来どおり、この
予算不足ならmode2を明示する。共有arenaの不足や後続malloc失敗はエラーであり、
mode2再試行とはしない。新しい一行のadmissionログに実際のfree/arena/prefix/
fixed/mixed容量を残し、机上計算だけで実機に収まると主張しない。

変更案はmoe-window10-candidate-v1.patchと同名JSONのsource hashへ固定した。
完全なパッチをMainが確認済み。適用後はRelease/UBSanのroute検査、traced GPUの
smoke、hot81920/balanced81920/partial78452で全出力・guardを検証し、新しいengineで
mode8/10の同一binary診断を行う。有望ならログなしの反復測定へ進む。

R1016のreviewにより、401674240 Bは81920行ではなく78452行であること、partialの
arena alignment、予算不足と割当失敗の異なる扱いを再確認した。
R1015の知識収集本文はMainでは読まず、selector overlapを測る具体的な実装判断まで保留する。
