# R956 — MoE scheduling 研究ライブラリレビュー

R601はexpert重みのchunk間再利用について、順序付きroute traceと実VRAM入場がない限りLRU hitを推定できないと既に整理している。R628は旧32K censusがlayer集約で、chunkごとのexpert順・group機会を復元できないと指摘。R633は2/4 chunk waveを示したが、route identity、全live memory、正しいslot順のFP32出力が必要である。R701のgrouped GEMMもcommand削減や現行shapeでの利益を証明していない。R702の64K案は保持payloadだけで約7.22 GiB、別置きresidual込み約9.72 GiBとなり、現行10 GiB級GPUの具体案にできない。

今回のB570測定はM=160のFP16-input/BF16-output例でwide 128×128がcompact 16×64より速く、dense M=8192でも差が出た。しかしこれは現行MoEのexpert別service timeではなく、実測ne分布もまだない。既存R814のscope reviewは、実route counts、live memory map、R810 source-review結果を前提にしており、現測定では埋まらない。新しいOrca研究は **schedulable_now=false** とし、dispatchしない。

Mainが現在の39K/C8192実行について、layer/chunk/roleごとのneと順序付きexpert IDs、dequant/GEMM/transfer/host grouping・waitの分離、GPU/RAM owner mapを記録し、R810のsource/provenance判定を確定した後に再判定する。先にR813の必要箇所をR814の条件に沿って取得し、残った具体的なtile-dispatch判断だけを新規scopeにする。物理262144、BF16 live state、numerical/task gate、10 GiB以内の実入場を維持する。

参照対象はreview JSON内の行・節マップとSHA-256を参照。`research-index.json`は変更していない。書き込みはR956のJSONとこのdigestのみ。
