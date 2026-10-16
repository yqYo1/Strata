# Hardware resources and the next measurement

2026-10-16 UTC。利用者の追加指示を受け、機能別時間だけでなくPCIe・RAM・VRAM・GPU演算・CPUの
どの資源が不足し、どの資源が空いているかを測る。処理の重なりと依存を含む同じ区間で比較する。
この報告で新しいGPUカウンターやモデル測定を実行したとは主張しない。

## 現在確認できること

`resource-topology-observation-v1.json`は今回のread-only sysfs観測。
00:01.2→01:00.0→02:02.0→03:00.0の外部経路は16 GT/s×4。
04:01.0とGPU function自身の2.5 GT/s×1表示を外部リンク帯域として使わない。
これはネゴシエートされたリンクの表示で、payload帯域・利用率・律速の測定ではない。
同時点のxe observation_paranoidは1。通常GPU実行の権限とPMU取得の権限は別である。

以前の現スタックのcopy実測は`own-host-transfer-component-v1.ja.md`にある。
320MiBのordinary H2Dは約6.015 GB/s、D2Hは約5.540 GB/s。host-USMのH2Dは約5.095 GB/s。
サイズ・メモリ種別・方向で差があり、一つの数値を全転送へ当てはめない。
10月10日のRAM約36.87 GB/s、VRAM copy約330.07 GB/s、FP16入力の大きなGU GEMM約52.69 TFLOP/sは
`../2026-10-14-pre-Xe-model-compare/historical-hardware/ATTAINED_CAPACITY_AND_DECISION_BUDGET.md`の
別スタック・別演算の到達値。絶対上限や現在の利用率にはしない。

既存の39K BF16固定32768残差の診断receiptを再集計した。48層のexpert upload要求量は
50,292,326,400 B、host残差はH2D/D2Hそれぞれ6,249,902,080 B。
これは要求payloadで、実PCIe wire bytesではない。全残差GPU設定はこのhost往復を除く。
今回のunitrace全process累計M2D/H2Dを、この量やprefill区間だけのDMAと同一視しない。

現sourceのfresh pos0=0、39261 batched、幅min(pos+1,2051)、12QSA層×2KV headから、
32-cell tileは59,649,720個、joint_matrix_madは15,270,328,320回、
padding/hi-loを含むsource-issued算術は62,547,264,798,720 FLOP。
valid pageとしたcode/scaleのlogical payload上限は993,764,040,192 B。
これはsource会計で、実DRAM/cache trafficや実行命令数のカウンターではない。
全processのAttention約33.966秒と機械的に割り、XMXやVRAM帯域の利用率を宣言しない。
計算式は`sum(ceil(min(i+1,2051)/32), i=0..39260)*12*2`、
mad/tile=256、FLOP/mad=4096、payload/cell/KV head=528B。
元receipt/hash、集計、source pinsは`resource-measurement-preparation-v1.json`。

## ソフトマックス比較の閉じ方

現行候補のclean正常標本はOFF403.4、ON404.2/404.8 tok/s。
次のOFFは推論・生成・runner/normal native closure・request検証を正常終了して402.4 tok/sを記録したが、
Mainが途中でAGENTS.mdへ測定方針を追記し、最後の固定ファイル検査でFAILになった。
元`own-prompt-softmax-long-clean-0-r1-v1.json` SHA
`d45130fd43d88c2d17a5081df7657f1f817ecd83c6bc7ba8074b959341fc8d67`はそのまま保持する。
新規fault0、dumpなし、全所有process回収済み。GPUハングとして分類しない。

AGENTS.mdは元SHA `7f750320bef4146a2872086922d84fc9fa7a8351ae655b7876c912856e3747fe`へ戻した。
guardを無効化せず、FAILをPASSや採用可能標本へ書き換えない。3標本/armの比較は未完了であり、
診断403.3→404.9だけでも採用しない。利用者の追加指示に合わせ、次の優先事項はhardware資源の測定。
候補のdefault OFF、既存FP64 oracle結果、広い品質/physical262144/RESTORE未了は維持する。
この方針を固定対象のAGENTSへ再追記せず、このcommitted reportに保持する。

## 実行する測定と並行作業

Orcaのsource-only担当を3件開始し、実際のmodel/turn_started/liveを確認した。

- gpt-6.1-sol / task_00abc6a80d11 / ctx_a9c4d296a850：PR6の忠実復元と具体的バグ修正。
  feature/syclの08874a80から`verify/pr6-prefill-restore`を一つ作った。
  当時の実行source closureと条件を特定し、復元と修正を別commitへ残す。Mainが全実行を担当する。
  子の報告先は`bench/results/2026-10-16-pr6-faithful-restore/report.ja.md`。
  検証・成果統合後にこの子を削除する。現時点で600超を再現したとは主張しない。
- gpt-6.1-sol / task_f981af7ebe67 / ctx_b2fdf7587fd8：既存qualified unitrace collectorから、
  nativeだけのdevice timelineと厳密なoffline集計を実装する。copyのbyte/type、service合計、
  busy union、copy/compute overlap、どちらも動かない区間を分ける。runtime/model/owner/fault/pin条件を維持し、
  Mainによる全差分レビューと小診断を通してから39Kを測る。全prefillの境界が証明できなければ
  whole-processや特定subwindowの結果と明記する。timelineをPMU/occupancy/barrier counterと呼ばない。
- gpt-6-luna / R875 / task_e8a697e1d7df / ctx_36199a0607bb：現Attention sourceの
  memory/synchronization/computeを区別する観測設計を調査。既存失敗VTuneや拒否済みK配置を繰り返さない。
  完了後にnative Luna-maxがreviewし、Mainは必要なdecision-support部分だけ読む。

R875初回launchはOrcaがeffort=max非対応としてpreflight拒否した。request確認に加えTask一覧で該当Taskなし、
worker一覧で復元担当のみliveを確認してから、effort指定を外した新しいlaunchを行った。
元拒否はlaunch-v1、実際の開始証拠はlaunch-v2。重複workerを作っていない。
R876/877のknowledge scopeはeligibleだが、現在の3worker枠が空くまで開始しない。

次の診断ではactual byte/FLOPとcapacityを同じscopeで対応付ける。CPU演算/提出とGPU待ちpolling、
PCIe payloadとDRAM bytes、非排他的service合計とcritical pathを分ける。
不足counterは明記し、source上のSLM量や長い機能時間だけでhardware律速を決めない。
同期/帯域/演算のcontrolled比較を加え、資源の制約を変えた時のlatency感度を見てから次のkernelを選ぶ。
成功raw timelineは抽出・検証後にlock下で退役し、全個別標本・fault結果・判断をGitへ保持する。
