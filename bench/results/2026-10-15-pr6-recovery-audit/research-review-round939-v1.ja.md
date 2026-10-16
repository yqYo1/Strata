# R939 調査レポートレビュー

- R937は指定SHAと一致する2,038語の判断支援レポート。Mode1ではhost KVが正本で、layer-major promptはresident slotではなく`m.stage`を読みます。全物理slotを読む現行packの初期化前提は一般のMode1では満たされず、現在のtyped QSA16経路もMode1を拒否します。
- R937の推奨は将来のMode1統合向けです。現行のstrict-oracle数値停止の原因説明、速度判断、採用判断には使いません。ローカル9ファイルのSHAは一致し、SGLang比較元はApache-2.0の固定commitです。
- R938は実ファイルSHA `dde882…d2637` の2,245語。8ソース以内で仕様、lowering、counterevidence、測定限界を扱っており、知識収集として本文を保留します。
- R938を読む候補は、将来診断がexp2/正規化/BF16確率丸めを示した場合だけです。本文8–13、21–23、29–31、37–40行を参照します。
- R938は実際のhost `std::exp2` 対SYCL `exp2` 経路やBF16確率境界を調べておらず、現行失敗の説明や許容誤差拡大を支えません。
- 両レポートをindexへ追加する案はJSONに記載。現時点で新しい研究スコープはschedulableではありません。
