# per-layer cache fillの修正と検証

source [e188dd60](https://github.com/yqYo1/Strata/commit/e188dd609b800ea78fb4afc0cc6dcf6ef7e02328)は、全ランキングを有限回走査してレイヤー別quotaを埋める修正と、重複profileの明示的拒否を含む。[Mainレビュー](xe-per-layer-profile-fill-main-review-v1.json)と[source publication](source-publication.json)に、元の報告・patchとの同一性、両モードの検証と公開先を残した。

release・host ASan/UBSanはfree/contrib-llvmの各モードで12件PASS、native ISAの1件skip。実際のproduction profile readerは、sparse・先頭nonzero-layer・重複・過剰header count・範囲外pair・truncation・emptyを検査した。[lint](lint-v6.json)は各83件の既存diagnosticに対して新規0件。generateの修正前diagnosticはsource・compile flags一致のclosed v5から行番号を変換して再利用し、今回の全3ファイルは両モードで実際にrun.shを実行した。失敗の非zero exitを消していない。

[変更前のモデルv4](model-v4.json)は8192入力で16出力、MTP22提案・7採用、QUIT exit0、終了後のinteger health PASS、新規GPU障害0。入力はcanonical先頭8143と検証済み49-token task/chat末尾で、前回の末尾を欠いた入力では停止トークン1個だった。これは入力全体の変更による動作確認で、末尾のどの部分が原因かの切り分け・長文性能・全vocabulary品質ではない。

次の[モデルv5](prepared-model-v5/run_root.py)は同じ入力で144/144 fillとresidency144、MTP・正常終了・fresh faultを検査する。まだ実行前であり、sourceのcommitを144/144の実測や性能の採用と扱わない。

[32K区間profileの準備](prepared-profile32k/run_root.py)は、過去R205/R207の不正timestamp警告により、現在は新しいraw marker資格receiptを必須にして起動を拒否する。既存foldが取得失敗区間を黙って除外するため、別のSol実装taskがタイミングの失敗検知を作業中。Mainはモデルv5の確認を続ける。イベント区間はCPU/copy待ちを含み、exclusive kernel service timeではない。

[メモリ配分候補](xe-per-layer-profile-fill-capacity-followup-v1.json)は観測した2133MiBのprefill領域、MTPをprefill時に保持するコスト、FP32のGU/Dm一時領域を記録する。精度変更は未実装で、全vocabulary likelihood・task品質・physical262144全域の検証を必要とする。[PCIe補足](xe-per-layer-profile-fill-pcie-metadata-correction-v1.json)はleafの2.5GT/s x1を帯域の上限に使わず、外部祖先のGen4 x4と区別する。新しい転送速度は測っていない。

[モデルv5](model-v5.json)は実行済みPASS。144/144 fill、residency144、最初のprofile pairはlayer28で実slot84を読み戻して検証した。16出力とMTP22提案・7採用、QUIT exit0、後段integer health PASS、新規GPU障害0。[個別解析](model-v5-analysis.json)では変更前と16 greedy IDsは一致したがlogprobは差がある。8Kの診断時間は採用判断に使わず、decodeが速くなったと扱わない。品質とphysical262144全域は未検証のまま。
