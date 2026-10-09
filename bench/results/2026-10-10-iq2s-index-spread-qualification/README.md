# IQ2_S index-only CPU qualification

The first root-owned IntelLLVM build failed at compilation before any fixture ran. Its original incomplete/failed status and compiler excerpt are retained. The private table header first included cstdint inside isolated_iq2s, declaring isolated_iq2s::std; global pre-inclusion fixes this without changing the dot arithmetic. Earlier read-only source reviews did not discover this compile defect.

Root added bounded mixed-byte/alignment checks and MXCSR endpoint logging without setting an FP mode. Release and ASan/UBSan execution, linked-operation/codegen proof, real-weight/live-input comparisons and performance adoption remain pending at this source boundary.
