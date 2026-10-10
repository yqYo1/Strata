XeStrata 39bdadccの未変更ソースで、Arc B570のgrouped XMX GEMMを14条件検証した。GU/Downそれぞれ1/127/128/129/257行、空expertを含む387行、全expert空を検証。独立FP64参照と3,951,360出力を比較し、全ケースPASS、非有限値・ガード破壊・新規GPU障害0。GU最大絶対誤差6.6121e-5、Down9.1661e-6。モデル品質、速度、262144セルの検証ではない。

小さなallocation selftestと独立H2D/kernel/D2H healthもPASS。qualifierの最初のビルドは相反するFPオプションのWerrorで失敗し、未変更ソースを -fno-fast-math -ffp-contract=off で再ビルド、CPU契約24条件PASS。元の失敗receiptを保持した。

成功APIログは構造化結果・入力/出力ハッシュ・所有プロセスの閉鎖・faultチェックを確認後に削除する。削除は元のreceiptを変更しない。次は未変更forkの32K診断、続いて実入力64Kの独立clean比較。1000tok/s目標は64K、decodeは別に3回。
