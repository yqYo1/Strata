# R557/R558とpost-terminal整理の確認

2026-10-15 JST。Mainは両reportと該当sourceを読み、UnitRaceのprimary READMEも確認した。
現在のsuiteは同じrunner69843/native1031613/start34056195で継続中。
実source/client/launch recipe、GPUの設定は変えていない。

R557のclosure hashの欠落はpost-terminal collectorに反映した。
suite bytes/hashとclosure receipt内のsuite、journalのpath/size/hash/state/verification、
runner resultとclosure内のresultを照合する。agentとnominal promptの個別samplesも
suiteとjournalの間でjoinする。元のPASS/failed statusを書き換えない。
collectorは未実行・未検証で、original runにもarchiveにもまだ作用していない。

R557が指摘したnominal32Kのactual inputについては、既にcommitted observationで
実値32089を保存してある。32768以上の入力だとは扱わない。
collectorのgate名を`nominal_prompt_usage_valid`に変更し、positive actual usageを確認する。
別の`nominal_prompt_meets_32768`に厳密なthreshold判定を保存するため、この三samplesでは
falseになる見込みである。このupstream標準phaseの24-token capはprefill測定だけで、
long decode、回答本文完成、32768以上の入力の証拠には使わない。
二種類のlong-generationについてはactual input>=32768を維持し、各3runを必須にする。
生成cap16384、thinkingON、body>=512charsも維持する。後付けで元結果を合格に変えない。

R558で、UnitRaceの`--start-paused`とprocess内の`PTI_ENABLE_COLLECTION`変更が
公式に対応していることを確認した。別のSDK linkなしでも収集を制御できる仕組みはある。
ただし、このmulti-threaded engineでprocess environmentを変更する案をそのまま採用は
しない。runtime/observerの同時environment accessとの契約、または公式のITT制御経路を
確認する必要がある。現時点で当該変更を追加したり実行したりしていない。

既存のcapture後・最初のgraph replay前から、既存のcompute/copy drainsまでなら、
新しいwaitを入れずに観測区間を選べる。layer splitのnested stages、samplingとcommitの
別launchは範囲を明記する。early failureではincomplete diagnosticとし、traceを止めたことを
GPU drainの成功と混同しない。normal graph replayの個々のdevice kernelが実際に取れるかは
まだ未確認なので、bounded smoke checkなしではkernel別の時間を報告しない。

今回のsource-review後も、測定を妨げるtest/build/profiler/cleanup/compressionをしない。
suiteのterminal検証・整理後に、PR #6の600 t/s超の実装と公開Codexログの確認を優先する。
次の並行研究はそこで判明する具体的な差分を対象にする。過去report registryは共有済み。

References:

- [Intel UnitRace in-application control](https://github.com/intel/pti-gpu/blob/6c0d6b0b80c6dbac4b5902d9d5ade1c75b9e9033/tools/unitrace/README.md), Main checked the current primary README as well.
- R557/R558 original reports are registered with their SHA256 in `research-index.json`.
