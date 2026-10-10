# Local effective hardware ceilings and default decode phase evidence

Ryzen5600X6cores/12threads,128GBRAM, ArcB57005:00, boot0a908c16. Safe production runtime: direct submission disabled, LevelZero-v2 copy offload disabled. All measurement processes exited normally; GPU stages had no new visible xe fault or devcoredump. Seven samples per case in one process, qualified full outputs; see validated-summary.json for every sample distribution. These are observed effective rates, not absolute maxima or memory-controller counters.

512MiB arrays: RAM read median37.689GB/s with6distinctphysicalcores,36.609with2cores; cachedcopy22.949GB/s vsnon-temporal37.575 with6cores (logicalread+write convention excludesRFO). GPU256MiB host-USM H2D6.449GB/s,D2H5.643; pageableH2D4.590,D2H5.295. GPUqueuecopy330.073GB/s logicalread+write, kernelcopy308.300, read-XOR266.511, triad-XOR316.676. Sources include arithmetic/checksum effects; no cacheeviction orphysicalDRAM proof.

oneMKL FP16-input/FP32-output geometry: GU1280x2560, Down2560x640, M1..8192, hot and8rotatingweights. LargeM8192 GUmedian52.787TFLOP/s, Down41.768; M8hotGU2.363,Down1.480. Synthetic +/-0.125 weights andonesactivations validate every output exactly but are not production-value distributions or a proof of coldweights. Smallmatrix shape matters as well as operation count.

Actual32768batchedpositions baseline+Tpool0 diagnostic passed firsthead/live66states and64IDs/logprobs/MTP parity, includingtwo matched32832checkpoint-restored continuations. Within the same restored requests CPUdispatch145.04/145.81ms perwindow joinsGU104.159/104.009,FFquant0.363/0.330,Down40.493/41.448. Phases reconcile but remain hostelapsed intervals includingdispatch/drain/barriers andGPUoverlap. No cleanspeed claim.

Scheduling vs architecture: a fixed~190GBprefill H2Dpayload would require~29.5seconds at the measured host-USM rate, leaving~3.3seconds against32768positions/1000tok/s if allotherwork overlaps. This is a conditional serialization estimate using a historical same-route32K queuedbyte receipt, not the new32769request's measuredtransfercount. The actual perexpertM histogram, transfercommand service, dequant andother phases must be joined before deciding the critical resource. Decode submits~3.18GB logical expert-blob extents/window; they are not DRAM reads. Its GU+Down dominances prioritize those phases, while physicaltraffic andworker-service remainunmeasured. No change is adopted from these ceilings alone.

Root holds new optimization work. A separate diagnostic-only Prefill timer guard is prepared to reject omitted/backward timestamps before collecting current-prefill phase data. Full262144 capacity gates remain mandatory before any production change.

Raw logs are small; retain currentreceipts/controllers/samples/failurestatus. No pack, activefixture or rawfailure deletion. Registryv76 records316 full-read researchreports androot corrections.
