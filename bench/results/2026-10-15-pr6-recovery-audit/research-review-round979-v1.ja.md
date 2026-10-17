# Round 979 review digest

R976–978の3 reportはすべて指定SHA-256とbyte数が一致し、wave manifest内の6入力hashも一致した。settlementでは各workerの最終dispatchが成功して解放済み。manifestのschedulable_now=trueはこの研究waveの前提を示すもので、次の実装・GPU測定まで今すぐ開始できるという意味ではない。

R976の停止位置はQSAの行列計算中ではなく、layer17/chunk16384で、成功したchunk drain後にnative16 callbackがdevice-to-device residual copyを出し、その後もう一度行うqueue drain。これが最も近い境界だがcopyやdriverが原因とは確定しない。optimized armの1回完了も免疫や速度を示さない。Mainのwindow0/2 matched runをoptimized QSAで進める。再発時だけ、chunk drain、callback copyの種類・範囲・submit結果、callback drainをhost記録する。現時点でdriver全体の調査は要らない。

R977のsmall-M提案は実測censusで支持される候補だが採用結果ではない。1–15行は30,543/59,614 calls、135,562/18,845,280 rows。過去のR627 small-ne packed-XMX、R728 direct-IQ/grouped、R754 groupingと考え方は重なる。新しさは7つの実GGUF format pair、M境界、FP16 weight decode・FP32 accumulation・BF16出力契約を絞ったscreenにある。現行window比較の後、slotが空いてから7 formatのcomponent screenへ進む。

R978のreturned-event案はinclusive markerの代わりにcommand intervalを測れる。eventは既存のsuccessful chunk retirement後に読むので新しいwaitは不要。command_end-startはbackend報告のaction時間、command_start-submitはruntime/queueのsubmit-to-start待ちで、どちらも利用率やEU/DRAM/PCIe量を示さない。host submit CPUは別に測る。修正点は、空stratumをeligible=0/selected=0として扱うこと、upload capの「16/request」と「16/chunk」の矛盾を直すこと。matched window結果に具体的な未解決質問が残るまでcollectorは保留し、新規researchは勧めない。

再利用する過去index metadataと未変更のindex hash、検証したsource pin、報告ごとのretrieve条件は、同じフォルダのresearch-review-round979-v1.jsonに記録した。index編集、build、test、GPU実行、source変更は行っていない。
