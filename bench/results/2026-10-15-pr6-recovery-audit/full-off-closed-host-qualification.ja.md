# Full OFF完了と比較toolの実適用

2026-10-15。exec16956はexit0で終了した。元native3892403/start38567673とwrapper3892400/
start38567660を含む所有identityの退去を確認した。元receiptの三commandはexit0、直接child回収、
session empty、survivorsなし。native/serverのexitも0、same boot、fresh GPU faults0、dumpなし。
`selective-iq-clean-bench-full-0-v1.json`のpassed/completeはtrue、activeはfalseである。
これは公式suiteの完了と退去・障害checkの結果で、task品質や物理256Kの資格ではない。

## 全個別標本

32K区分の実入力は三回とも32089 tokenで、32768以上ではない。PPは200.5/199.8/199.5 tok/s、
中央値199.8、TTFT160.376/160.944/161.178秒、reuse0だった。nominal32Kであることを明記し、
long-generation側の実33214/33288 tokenとは混同しない。
agent sessionは六turn×三回が完了した。

| workload/run | 入力token | 生成token（thinking含む） | decode tok/s | draft accepted/offered |
| --- | ---: | ---: | ---: | ---: |
| regular-code/1 | 33214 | 3346 | 9.8 | 2382/2895 |
| regular-code/2 | 33214 | 4164 | 9.7 | 2961/3612 |
| regular-code/3 | 33214 | 3198 | 9.8 | 2277/2766 |
| varied-prose/1 | 33288 | 8631 | 6.2 | 4551/12243 |
| varied-prose/2 | 33288 | 9056 | 6.4 | 4926/12387 |
| varied-prose/3 | 33288 | 8869 | 4.6 | 4774/12282 |

六件すべてeligibleで、MTPのstrong/weak分類は両方成立した。code中央値9.8、prose中央値6.2だが、
prose最後の4.6を除外しない。acceptanceだけからこの速度差の原因を決めない。三回は同じserver内の
異なる要求で、独立processの分散測定ではない。PR6約646 PPの回復、1000 PP/70 TGは未達である。

## 実行したhost検証と一件の失敗

元clientを変える前にrecover_bench_nonce_v1.pyを実行した。元要求の二つのcalibration body SHAを
両方一致させる唯一のnonce1792036964を、実commandの開始/終了時刻の有限範囲から回復した。
`selective-iq-full0-workload-provenance-v1.json`はpassed/complete true、GPU_executed false。
これ自体をON/OFFの実入力一致の証明とは扱わない。

qualify_bench_input_tools_v1.pyではfixed-ID六testが通ったが、comparator十一testのうち一件が
OSError27で失敗した。境界拒否testが8MiB+1-byte fileを作るところで、Ownerの8MiB file capが
先に拒否した。元bench-input-tools-host-v1.jsonを失敗のまま保存した。test/comparatorの入力上限・
判定を変更せず、実行側file capだけ16MiBとしたqualify_bench_input_tools_v2.pyを別rootで実行し、
六＋十一testが全件PASSした。両runとも所有commandは終了・回収済みである。

qualify_wave_census_host_v2.pyも実行しPASS。実proposal headerのcontract/serializationをnormalと
ASan/UBSanで実行し、独立Python照合とvalidator四testが通った。CPU合成routeでtoken262144/
offset17+262127まで確認したが、実model/GPUで物理256Kを検証した証拠ではない。
qualify_checked_pcie_fetch_host_v1.pyも実行しPASS。実production helperへfake APIを渡すfaultケースが
normalとASan/UBSanで通り、両fixtureは無出力・exit0・退去済みだった。実Verifier/SYCL DMA、
integration build、正のPCIe fraction、速度や品質は引き続き未確認である。

## 実際に変えたclient

host PASS後、レビュー済みbench-fixed-workload-v1/source.diffをtools/strata_bench.pyへ適用した。
適用前c2cb3aed...、適用後7be3ab66b9afaf911c80c0fc7869fa440886841c5168564df09202199a2e3645は
実host検証したproposalと一致する。任意の--workload-id指定と実nonce/policy記録を追加し、
既存timed/request buildersはASTで不変を確認した。旧結果のstatus/source/hashは変更しない。
run_matched_iq_on_v2.pyは新しい実PASS host-v2 receiptを参照するよう変更した。

次のONは元b1cc9cb4... native、engine args、safe runtime envを維持し、IQ選択だけを1にする。
clientには回復nonceを指定する。実fixed request bodiesと順序・parameterが一致した場合にだけ
入力を対応づける。agent後続turnの生成履歴差は非pairedの観測として残す。ONはまだ未実行。

原full OFFのrequest journalとbounded logsは次のON照合および品質確認のconsumerがあるため残す。
64MiB text/job上限を引き継ぎ、ONの終了・比較抽出後にretentionを再確認する。元失敗host receiptと
必要なerror stackはGitへ保存し、失敗をPASSへ書き換えない。新native/GPU比較、task品質、
物理262144/RESTORE、upstream/defaultの未完了をhost合格で置き換えない。
