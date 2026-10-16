# MoE窓source一覧の一括転送を選ぶ

2026-10-17、Mainの接続設計変更。R970はC++のsource/slot/job対応を支持し、GPU接続は未了とした。
元の8K jobごとにsource indexをdevice_src[0]へ転送する案は、安全だが小さいH2D投入をjob数だけ増やす。
新しい窓領域には算術上の余裕があるため、source一覧をW×10 int32の本当の全窓ownerに変更する。
最大655,360 Bを窓ごとに一括転送し、各gatherは`window_src + job.global_row`から読む。
出力jobの容量は8Kのまま。古い32,768 Bのjob領域へglobal offsetを足す方法ではない。

実行した`calculate_moe_window_bulk_source_v1.py`は、固定した旧plannerのview配置を用い、最後の
source viewだけを拡張した。追加622,592 B、guard込み使用998,443,520 B、既存regionとの差182,255,104 B。
他viewのoffset/payload/reservationは同一。これは実allocation/contextや物理fitの証明ではない。
CPU/PCIeの仕事を減らす意図でsource H2DのAPI投入を窓ごと1回へ減らす。合計のsource bytesは同じ。
実driver command数、staging、所要時間、速度効果はまだ測定していない。

C++ planは元から全窓のhost source配列とglobal job offsetsを所有し、R970で対応方向を照合済み。
40ケースのC++検証はそのmetadataの証拠であり、このGPU全窓uploadを実行した証拠ではない。
実装側で実際の全窓device allocationのextent/context/deviceを確認し、全job spanの境界を検証する。
同queueへsourceとslot mapのuploadを各一回投入し、host planをchecked drainまで保持する。
その後gather/GU/swiglu/Downをjob単位で投入し、全Downの後にcombineと両HC writeを行う。

旧32KiB/rebase方式の契約・テスト・receiptは不変で保持し、この決定で過去の範囲を変更しない。
GPU常駐2pieceのみ窓化、8K operator/GEMM job、PLE/GDN/QSA/両write/MTP/checkpointのgateは維持する。
