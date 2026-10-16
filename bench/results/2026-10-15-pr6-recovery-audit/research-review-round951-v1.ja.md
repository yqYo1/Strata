R950は、`alignas(4) MatrixTile`の型寿命・BF16 matrix alignmentを扱う有用な知識収集です。1,551語/11,664 bytes、hash一致。R948のscore pitchとは別の問いで、現在のoptimized callerに沿っています。

SYCL rev.12のlocal_accessor、C++ aggregate lifetime、固定Intel matrix仕様の読み分けは妥当です。load offsetは16 BF16要素=32 bytes、strideは64/512 bytes。4-byte baseには `values` member offsetの確認も必要です。現行assertはQ中心でKV/P未確認、LocalPayloadは `*2` 固定。全型のlayout/offset/sizeと `sizeof(BFloat)==2` をassert推奨。現状不良の証拠ではありません。

base/optimized各11 componentケースがPASSし、262144 cellsを含みfaultなし。ただしreceiptはmodel/full-context/RESTORE qualification・速度を未認定。runtime PASSは命令列を示さず、MainのNLL比較と6-sample official benchは継続中です。性能・採用結論はありません。

worker_done成功とreport hash一致。terminal previewはLuna xhighですがprojectionのmodel欄はnull。releaseはretained/external_terminal・processActionなし、Main作成terminalは別close記録。報告作成・ACK・端末終了を分けました。新規dispatch不要。具体的なstorage/compiler選択が出た時に該当節を取得します。
