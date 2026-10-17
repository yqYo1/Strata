# 実測censusとCPU/waitの基準データ

2026-10-17、外部CPUベンチPID3431768/start52544839の消失を確認し測定再開。
既存frozen binary af043aad...、公式bench.sh経由、実入力39262/再利用0、
context65536/chunk8192/GPU residual32768/native16 BF16/vLLM/optimized QSAで
診断付きprefill459.5tok/s、prompt85449ms、wrapper84109.9msで完了した。
thinking有効、24出力は動作確認用でdecode速度比較には使わない。
1診断であり、cleanな反復速度の採用や品質/256K/RESTOREの資格付けではない。

主thread CPU78.694sのうち58.052sは待機中CPU。待機外CPUは20.642s。
その内訳にはsource-copy4.895s、upload-submit4.955s、GEMM-submit2.398s、
host-group0.0826s、unclassified6.542sがある。1coreの高使用率から
全てがCPU計算律速とは判断できず、これらCPU値も重なるGPU時間へ単純加算しない。

MoE nonzero expert-piece59614回、GU/Down GEMM119228回。
1..15行の30543回が呼び出しの51.235%、routed rowでは135562/18845280=0.719%。
小さいMの経路は新しい具体的な調査対象である。ただしlogical FP16 dequant
書込み586.03GBはphysical VRAM trafficではなく、cache作用も測っていない。
8K histogramだけから隣接pieceの同一expertの重複も求められない。

coarse queue区間はQSA24.714s、GDN9.243s、MoE20.186s、upload8.469s。
QSA内attention18.750s、selection3.888s。これらはhost投入の隙間を含む
非排他的区間であり、GPU service/utilizationやGPU compute77%の根拠には使わない。
50292326400 requested upload bytes / 8.46862s = 約5.939GB/sだが、
これもwire帯域や飽和率ではない。真のkernel start/endの収集を別に検討する。

sysfs末端05:00.0と04:01.0はcurrent/maxとも2.5GT/s x1を報告するが、
その上流03:00.0・02:02.0・01:00.0・00:01.2はcurrent16GT/s x4。
末端値をPC全体の外部PCIeリンク帯域と同一視しない。経路snapshotはanalysis JSONに保存。

同じ二本の比較のうちpacked armはlayer17/chunk16384で数分進まず、
host stackはURv2 queueFinish -> cudaStreamSynchronize -> layer-major bank callback。
Mainがそのnative PID/startをpidfdで照合してSIGTERMし、全ownerの終了を確認した。
新しいkernel fault/dumpなし。pair全体はFAILのまま、packed速度値なし。
このstackだけで原因kernelやpacked算術の不具合を断定できない。
停止後、再起動せずtraced runtime64 handoffとnative16 smokeが両方PASSした。
原profile/stop/stack/healthのreceiptは改変せず保存する。

その後、別child verify/moe-window source53c759cbの完全build/linkと
production route/gather/combineのGPU11case（各2salt）がPASSした。
主な証拠はchild commit b8806476のbuild/GPU receiptとreport。
実FFN/GEMM/state/physical262144/RESTORE/速度はまだ未検証で、Draft PR32のまま。
次は新binaryのwindow0/2をoptimized QSAで比較する。未採用packedを再試行しない。
研究R976/977/978は新実測を入力に終了し、native Luna-max R979でreview中。
Mainはその本文を読み込まず、review digestを利用する。
