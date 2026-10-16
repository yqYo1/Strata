# R875 レビュー運用メモ

R875は1,468語、一次資料4件で依頼予算内です。R748のカウンター取得制約を繰り返すだけでなく、現在のQSA sourceの各段階、unitraceの時系列、転送量とPCIeの区別を結び付けています。R746の演算数・roofline計算や新しいkernel案は再掲していません。

参照前に次の2点を補正してください。R875-C1: 19行目の「V scales were loaded with K scales」は誤りです。K scaleはQK前に`ks`へ、V scaleはsoftmax後に別配列`vs`へ読みます。R875-C2: 17行目の表にはK/Vではなく、K code/scaleをロードした後、QKの前にあるbarrierが抜けています。22行目の7 barrierという合計は正しいです。

Mainが必要なら、source段階とcounter/timeline対応は13〜22行、unitrace能力と現在のカウンター上限は26〜32行、時間・payload・PCIe・DRAM等の区別は34〜38行、測定手順と判断境界は40〜44行を参照してください。参照は進行中のnative unitrace timeline作業のレビューまたは後続測定の判断に限り、別のprofiler調査や実装を始める根拠にはしません。

R861の2診断はclean sampleではなく、現在のclean cohortはAGENTS.mdの固定確認失敗で未完了です。計画はcohort完了を条件としているので、未完了armをclean比較として扱わないでください。別記録の外部PCIe 16 GT/s ×4はネゴシエートされたtopologyであり、達成payload帯域や利用率ではありません。R876/R877のknowledge-collection本文は未読のまま保持し、新規dispatchは行っていません。
