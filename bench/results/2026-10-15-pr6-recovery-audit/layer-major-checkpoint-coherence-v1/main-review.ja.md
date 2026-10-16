# Main review: coherent caller-segment checkpoint proposal

2026-10-15。FINAL後、2file source.diff、planner、独立event-oracle fixture、Main検証計画を確認。
未適用・未build・未test・未GPU実行。Direct0/1比較中の製品source/binaryは変更していない。

callbackからの不整合saveを対象モードだけ抑止し、全層を処理したrunの終端でMTP expertsの
復元、loan refillを確認してから保存する構造は、位置不整合を避ける具体的な安全策候補である。
root/turn境界、short-readを選ぶ元outer長、取消時のpartial LIVE非公開、reread診断の無分割を
保つ。refill/save失敗を同時cancelで成功に変えない`boundary_failed`も確認した。

`checkpoint_next`の正値加算は飽和し、`checkpoint_segment_end`は差分の範囲でround-upする。
負intervalをthrowするhelperへの置換について、CLI parserが`max(0,atoll(...))`にすることを
照合した。CPU fixtureは実stateやcopy完了を証明せず、実行結果もまだ存在しない。
loan relayoutでactual chunkが変わる点は実engine境界の記録が必要である。

**性能改善の標準経路としては、現時点でこの方式を採用しない。** 周期16Kのために
whole Prefill::runを分割すると全層expert sweep、MTP expert restore、loan返却・再lendを繰り返す。
R609の論理expert payload増加はMainが既存model metadataから再計算した。
ただしR609のbulk帯域からの時間を厳密なlower boundや目標不可能の証明には使わない。
詳細は`main-review-round609-610.ja.md`と`checkpoint-traffic-scenarios-v1.json`に記録した。

本案は安全性fallbackと、別方式のRESTORE比較用の候補として保存する。次の実装案は、
一回のlayer-major pass中に各層の同じprefix位置のstateだけを捕捉し、全pieceとMTPがそろい、
run/復元/refillが正常終了してから公開する。periodic機能とexpert再利用の両立を目指す。
Mainが別Solにその提案を依頼し、review/CPU/実GPU数値・RESTORE・physical256Kは未了。
