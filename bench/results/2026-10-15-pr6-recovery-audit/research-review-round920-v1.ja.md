# R920: R918知識収集レビュー

R918は、保留する知識コレクションとして受理する。新規点は、OA query/poolと独立したSysmanのPCI・memory snapshot、およびXe debugfsとXPU-SMIの集計・帰属範囲を比較したこと。R875の広いresource計測限界、R911の停止済みOA経路、R914のコンポーネント比較とは役割が異なる。

公式Level Zero v1.17.24の仕様で主要APIとPCI timestamp/counterの説明を照合した。memory帯域構造体にはbytesというfield説明と、deltaを32倍するtransaction説明が併存するため、換算は未解決として保持する。Linux `next` のXe統計は現行ホストの実装証明ではない。どの資料もB570上の対応を実証していない。

今は新しい研究スコープもcounter実験も勧めない。将来は、Mainの既存作業で実機のBDF・driver/runtime/kernelとSysman capability/errorが記録され、さらに転送に関する具体的な未解決測定問が残った場合だけ、対象を絞ってeligibilityを再評価する。権限変更やOA再試行を提案しない。

**本文の取得条件:** 実測済みの具体的なAPI・counter問では9–14行と17–31行だけを参照する。許可されたprefill観測設計を検討する場合に限り37–41行を加える。R918-C1〜C4と、R916の「アイドルPCIe階層は速度差の説明にならない」という訂正を適用する。Mainの本文読了はfalse。
