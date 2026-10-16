# IQ materialization: actual CPU and B570 correctness gates

2026-10-15。推論本体は変更していない。元のstrata binary92acf71eと既存archiveを保持したまま、
candidate iq.cppを別のGPU fixtureにcompile/linkした。nativeと同じcustom clang22/oneMKL世代の
runtimeを使い、実行前の固定environmentでOFF/ONを切り替えた。runtime中のsetenvはない。

Mainが実行したCPU fixtureは6形式18/22/21/23/20/42について503808 blocks、94044160 elements。
normalとASan+UBSanの双方で元scalarとhalf全bitが一致した。6920016個のNaN比較も含む。
有限値とInfはFP32中間値もbit比較している。FP32 NaN payloadは除外し、halfのcanonical
payload/signは比較している。scale/codeの全直積や専用halfway corpusを検証したとは言わない。

同じCPU jobの後続checkpoint fixture compileはunused helperをWerrorで拒否した。
元exit1/receiptをcandidate-cpu-contracts-v1.jsonに保存し、未使用read_tagの削除だけを行った。
候補本体とassertionは変更せず、checkpointだけ新root-v2で再build/runした。
normal/ASan+UBSanとも通った。これは実candidate headerのposition/owner/LRU/RAII/reread契約を
host stubで検証したもので、実tensorのRESTORE、queue、MTP KV、物理256K成功ではない。

GPU fixtureは小さい42case、その後実際のGU640x2560/Down2560x640を加えた48caseで、各々
新しいOFF/ON processを順番に実行した。48caseには先の42caseが含まれる。
全half配列（NaN bitを含む）、guard、全input、CPU oracle、6つのF32配列が一致した。
各armの20 invalid-entry checksも通った。全process正常exit0/reap/session空、fresh fault0、
dumpなし、boot不変。IQ新経路だけでなくunsupported typeの旧経路とF32非変更も確認した。

quickのUR API traceをkernel-create handleと実launchで対応づけた。
両armのlaunchは132回で、そのうち新parallel kernelはOFF0、ON18。
ONでは6形式それぞれgeneric2回/GU1回を実際に実行した。正しい出力だけから新経路の実行を
推測していない。marker interval、host submit、completed wallは個別に保存したが、
一回だけの診断sampleであり、exclusive kernel durationや速度採用の証拠にはしない。

正式Direct0は失敗のまま、Direct1は未実行。今回の別診断では、元と同じ32089-token入力3件を
直接所有するnativeに渡し、全48layerを3回通過して各24tokenを生成した。
prompt msは183280.3/161970.2/162591.3、decode msは8802.1/2533.9/2557.0。
3件の生成IDsは同じで、draftは各17/21。validation/trace付きで先行chat/agentを省略している。
したがって速度比較でも元の停止原因の解決でもない。停止は観測せず、GDB attachは不要だった。

close_iq_diagnostics_v1.pyが、正常終了/所有process不在、全比較fileのhash、commit予定の
compact比較copy、実launchと全layer進捗を検証した後、成功raw配列と繰返しAPI/layer trace
866 files/221980654 logical bytesを削除した。physical空き容量の増分は主張しない。
path/size/hash/reasonはiq-diagnostics-retention-v1.json、実routeと進捗は
iq-diagnostic-evidence-v1.jsonに記録した。元の未解決停止の代表証拠と3つの再現token入力は保持した。

次はdevice-residentな同じweight/X/scratchで、展開単体と展開+既存oneMKL GEMMを分けて
3つ以上の独立process/arm、交互順で測る。timing-v1は毎trial後に全shared-USMをCPUで検証して
移動/evictionを混入させ得るため実行せず、device USM、timing外の前後だけの検証に修正する。
実model/official bench、prefill/decode別、RESTORE/cancel/full256Kは以後の別gateである。
新経路のdefault有効化やモデル速度改善の主張はまだ行わない。
