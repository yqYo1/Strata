# IQ2_S index-only CPU qualification

The first root-owned IntelLLVM build failed at compilation before any fixture ran. Its original incomplete/failed status and compiler excerpt are retained. The private table header first included cstdint inside isolated_iq2s, declaring isolated_iq2s::std; global pre-inclusion fixes this without changing the dot arithmetic. Earlier read-only source reviews did not discover this compile defect.

Root added bounded mixed-byte/alignment checks and MXCSR endpoint logging without setting an FP mode. Release and ASan/UBSan execution, linked-operation/codegen proof, real-weight/live-input comparisons and performance adoption remain pending at this source boundary.

Root v2 fresh release and IntelLLVM ASan/UBSan builds both passed normal exit0 with no stderr, cleanup or owned survivors. Each checked 262144 uniform indices, 4194304 mixed active indices, 1048576 mixed decoded vectors across16 alignment offsets, and96 synthetic profiles/1536 GU pairs/9216 dot calls. All finite Gate/Up/finish results were bitwise equal across traits/control/index arms.

MXCSR controls were logged and unchanged from post-initialization to final checks; exception status may change. This does not establish production thread FP state. Both artifacts are CPU-only by direct imports; no GPU, model or timing ran. Complete flags, command/environment/source/compiler/binary identities and individual CSV results are retained. Exact release symbols/callers/quantizer are captured for future offline interpretation; no emitted-mechanism or speed claim yet.

Observed FP entry lines: {"release": "fp_environment,00001f80,00001fbb,0000ffc0", "asan-ubsan": "fp_environment,00001f80,00001fbb,0000ffc0"}
