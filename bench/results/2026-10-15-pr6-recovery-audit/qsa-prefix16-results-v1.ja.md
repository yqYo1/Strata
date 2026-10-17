# QSA16セル候補と262144セルの追加検証

同一binary/input、pack込み、各arm3process×10時間標本。Arc B570。

|論理prefix|32セル中央値(ms)|16セル中央値(ms)|16セルの遅延短縮|
|---|---:|---:|---:|
|32768|333.254|360.328|-8.124%|
|131072|444.515|437.850|1.499%|

32K領域では16セルが悪化、128K領域では小さな改善だった。本体は32セルを維持し、
16セルは大きいprefix向けの候補として保持する。model PPへの効果は未測定。
16セルの18部品caseは独立16セルoracle、guard、入力不変、不正範囲拒否、NaN/mask、
物理262144セルの1queryケースを含めてPASS。一般品質・実モデル256K消費・RESTOREの保証ではない。

8192query/262144prefixの追加速度試験は既存32セル側の数値oracleで停止し、時間標本はない。
Mainが全PID/start ticks消滅・session空・子reapを再確認した。新規GPUfault/dumpなし。
別のhost-only observerを同じGPUobjectへlinkし、同じ入力fingerprintで同じ失敗を再現した。
最初の差はquery2/head13/d31、GPU=-0.000736236572265625、CPU=-0.000728607177734375、
絶対差7.62939453125e-06、基準5.8146972656249995e-06。元のFAILは変更していない。
これは過去に調査した入力・最初の座標・値と一致する。過去の係数replayではFP32 QKとFP64 dotの
順序差がBF16確率の丸め境界をまたぐためと説明済み。ただし今回全出力一致の比較は追加していない。
epsilonを失敗に合わせて緩める変更は行わない。数値停止をGPUハングとして扱わない。

observer receiptのtraced=falseは速度modeから引き継いだ記載ミス。実環境・proc snapshotと
UR2kernelの記録によりtracing+parameter validation有効を確認している。速度標本には使わない。

次はselectionのFP32 scoreとtop-kを分離して測る。参照精度の違いは段階別契約・品質と区別し、
1000 PP/70 TG、全context、RESTOREの未達・未検証を維持して最適化を続ける。

[全個別結果と判断](qsa-prefix16-decision-v1.json)
