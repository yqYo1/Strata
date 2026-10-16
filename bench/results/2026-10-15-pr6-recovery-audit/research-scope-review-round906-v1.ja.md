# R905 レビュー digest（R906）

**判定:** decision support として受理。R905-C1（Metrics Discovery / Library の source provenance と互換性の限界）と R905-C2（記録された descriptor では到達しない分岐）を適用する。元レポートは変更していない。

## 次の判断で読める範囲

[research-round905-v1.txt](research-round905-v1.txt) の次の行だけを参照する。

- 8–20: 実測 API 結果、ロード済みライブラリ名・ハッシュ、パッケージ版とソース pin。R905-C1 を適用。
- 24–33: INVALID_ARGUMENT へ集約されるドライバ経路と count=64 の意味。R905-C2 を適用。
- 37–39: activation context と pool-call context の差。調べた compute-runtime 版の範囲を保つ。
- 43–55: 依存版、OA 権限、次の限定診断。原因を断定しない。

## 実測から言えること

記録では ComputeBasic の event group 列挙と activation は成功し、最初の performance query-pool 作成が `0x78000004 / ZE_RESULT_ERROR_INVALID_ARGUMENT` を返した。query begin、kernel の数値結果、counter sample は得ていない。所有プロセスは通常終了し、同じ記録は fresh driver fault 0、dump なし、collection 未認定を示す。したがって、この失敗だけで OA 権限、context mismatch、GPU hang、Strata kernel defect は言えない。

pin した [compute-runtime 26.31.39395.14 の OA 実装](https://github.com/intel/compute-runtime/blob/26.31.39395.14/level_zero/tools/source/metrics/metric_oa_query_imp.cpp) では、Metrics Library の query-create 失敗と report-size が 0 になる allocation 経路が、どちらも同じ INVALID_ARGUMENT に集約される。implicit scaling の subdevice create failure も別経路になり得るが、当該記録はその分岐を確定していない。Level Zero [v1.32.0 header](https://github.com/oneapi-src/level-zero/blob/v1.32.0/include/zet_api.h) の count=64 は internal slots を指し、メトリック数ではない。[PTI の pinned collector](https://github.com/intel/pti-gpu/blob/6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033/tools/unitrace/src/levelzero/ze_collector.h) も performance pool に 64 slots を使う。ただし同じ値は Metrics Library QueryCreate に渡され、report allocation の乗数にもなる。

## 適用する修正

- **R905-C1:** 記録はロード済み libigdmd / libigdml の SONAME、ファイルハッシュ、パッケージ版を示し、compute-runtime パッケージの source version を upstream tag に対応付けている。しかし Metrics Discovery と Metrics Library のロード済み image を正確な source commit/build に対応付けていない。compute-runtime manifest の依存 pin は build input であり、それだけで実行時の互換範囲や不互換を証明しない。主張は「version/SONAME の差がある候補」までに留める。
- **R905-C2:** 記録された type=PERFORMANCE (0) は `createMetricQueryPool()` の有効分岐に入る。unsupported descriptor の default 分岐はこの呼び出しでは起きないため、候補原因から外す。
- context の差は記録上実在するが、同版 OA pool method は hContext を読まず device から metric source を取る。この出典だけではその差を原因にできず、pool 作成後の query behavior も証明しない。

## 次の限定診断

Main 所有の小さな harness を一回だけ使い、Metrics Library の初期化状態、`GetParameter(QueryHwCountersReportGpuSize)` status/size、`QueryCreate` status/slot count、allocation-size 判定を同一環境で記録する。64 slot の QueryCreate だけ失敗した場合に限り count=1 を一回比較する。既存の package version と xe observation policy、実効 identity は読み取りだけで確認し、権限変更、昇格、service 再起動、reset はしない。結果が出るまでは time-based sampling も未認定の候補に留める。

新しい研究 scope は今は不要。Main の限定診断結果か正確な package/source provenance が揃い、それでも具体的な upstream 問いが残った時だけ再検討する。
