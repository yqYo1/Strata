# R911 レビュー digest（R912）

**判定:** decision support として受理。R911 は、記録された Metrics Library/TBS 初期化失敗が `QueryCreate` より前の境界で起きたことを、条件付きのソース説明として示している。権限拒否、版の不整合、全カウンター利用不能までは証明していない。

Main の v3 syscall receipt には renderD128 上の `EACCES` ioctl が2件、固定 GUID の `id` パスに対する `ENOENT` が2件ある。strace の名前は `DRM_IOCTL_I915_CMDBUFFER` だが、該当する Metrics Library header では Xe observation と i915 command buffer の ioctl 番号・x86_64 構造体サイズが一致する。この表示だけでは実際の Xe OA sub-operation を特定できず、permission denial の根拠として扱わない。

`zetContextActivateMetricGroups` の成功は遅延された API bookkeeping と区別する。Metrics Library の TBS bootstrap と、選択した Metrics Discovery event group の登録も別段階である。固定 GUID ファイルの `ENOENT` はその読み取り時点での不在を示すだけで、全 group や全 counter route の不在を示さない。Paranoid warning は pinned source で常に success を返す検査から出ており、`observation_paranoid=1` と `CapEff=0` は原因を確定しない。

ロード済み版・SONAME・hash は報告されているが、Metrics Discovery/Library の実行バイナリが参照 source commit からビルドされた証明はない。compute-runtime manifest の依存 pin も互換範囲や原因を確定しない。Linux v7.0 の権限動作は比較資料で、稼働中の Ubuntu kernel source ではない。

R911 の次の syscall trace 提案は Main の v3 記録で実施済み。Main は counter 診断を止めて component 比較へ戻るため、新しい調査 scope は今は不要。詳細が必要になった場合だけ [レビュー JSON](research-scope-review-round912-v1.json) の指定行を参照し、上記 caveat を維持する。
