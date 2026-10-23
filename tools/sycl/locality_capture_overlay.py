"""Main-authored bounded locality/step capture overlay; repository LICENSE.

Only a separate diagnostic TU changes. Original6e40 engine/decode stays selected.
The legacy unchecked dump is removed,not enabled. Build recipe pins generated
source and the exact original quoted-include directory.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

BASE='6e40afc1beddd059ee518617f6c1efce1da14aca'
SOURCE='src/prefill/prefill.cpp'

HOOK = r'''#if defined(STRATA_ENABLE_SYCL)
                    if (std::getenv("STRATA_QSA_DUMP")) {
                        if (cudaStreamSynchronize(m.cs) != cudaSuccess)
                            strata::sycl_backend::cleanup_fail_stop("legacy QSA dump refusal: completion uncertain");
                        err = "legacy unchecked QSA dump is disabled in this diagnostic"; return false;
                    }
                    static const char* const locality_dump = std::getenv("STRATA_QSA_LOCALITY_CAPTURE");
                    if (locality_dump && m.layer_major_active) {
                        if (!typed_qsa || !*locality_dump || pos0 < 0 || n <= 0 || n > INT32_MAX ||
                            pos0 > INT32_MAX - n || p0 < pos0 || p0 > pos0 + n ||
                            T <= 0 || T > pos0 + n - p0 || T > m.T) {
                            if (cudaStreamSynchronize(m.cs) != cudaSuccess)
                                strata::sycl_backend::cleanup_fail_stop("QSA locality geometry refusal: completion uncertain");
                            err = "QSA locality capture requires a bounded native16 layer-major chunk"; return false;
                        }
                        if (p0 + T == pos0 + n && !detail::locality_capture(locality_dump,
                                qsa_index, p0, T, m.T, m.cap, strata::kernels::kStepCount,
                                m.sel_ids, m.steps_dev,
                                [&](void* dst, const void* src, size_t bytes) {
                                    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, m.cs) == cudaSuccess;
                                },
                                [&] { return cudaStreamSynchronize(m.cs) == cudaSuccess; },
                                []() noexcept {
                                    strata::sycl_backend::cleanup_fail_stop("QSA locality capture: completion uncertain");
                                }, err)) return false;
                    }
#endif
'''
CALL = r'''                        else if (arithmetic == native16::qsa::AttentionMath::bf16_prefix_ordered_exp)
                            matrix_used = strata::kernels::qsa_bf16_prefix_ordered_exp_batch(
                                m.qsa_q16, pools, m.sel_ids, m.steps_dev, m.cap, s, m.qsa_attn16, T,
                                {st.n_pages, st.n_slots}, p0 + T, m.qsa_prefix, m.cs);'''
NEW_CALL = r'''                        else if (arithmetic == native16::qsa::AttentionMath::bf16_prefix_ordered_exp) {
                            matrix_used = strata::kernels::qsa_bf16_prefix_ordered_exp_batch(
                                m.qsa_q16, pools, m.sel_ids, m.steps_dev, m.cap, s, m.qsa_attn16, T,
                                {st.n_pages, st.n_slots}, p0 + T, m.qsa_prefix, m.cs);
                            if (locality_dump && m.layer_major_active && matrix_used) {
                                const uint64_t page = uint64_t(s.page_size);
                                const uint64_t live = uint64_t(p0 + T);
                                const uint64_t packed_rows = ((live + page - 1) / page) * 2 * page;
                                if (std::fprintf(stderr, "strata QSA locality prefix: layer=%lld first=%lld queries=%lld live=%llu page=%llu packed_logical_rows=%llu submitted=1\n",
                                        (long long)qsa_index, (long long)p0, (long long)T,
                                        (unsigned long long)live, (unsigned long long)page,
                                        (unsigned long long)packed_rows) < 0 || std::fflush(stderr) != 0) {
                                    if (cudaStreamSynchronize(m.cs) != cudaSuccess)
                                        strata::sycl_backend::cleanup_fail_stop("QSA locality log failure: completion uncertain");
                                    err = "QSA locality prefix log failed"; return false;
                                }
                            }
                        }'''

def generated(project):
    project=Path(project)
    original=subprocess.check_output(['git','-C',str(project),'show',BASE+':'+SOURCE],timeout=20)
    if original!=(project/SOURCE).read_bytes():
        raise ValueError('selected production prefill source changed')
    text=original.decode()
    start=text.index('                    // STRATA_QSA_DUMP=<file>:')
    end=text.index('                    const strata::kernels::QsaAttnPools pools =',start)
    before=text[start:end]
    include='#include "strata/prefill/draft_uploads.hpp"'
    added=include+'\n#include "strata/prefill/locality_capture.hpp"'
    if text.count(before)!=1 or text.count(include)!=1 or text.count(CALL)!=1:
        raise ValueError('locality overlay anchors not unique')
    fixed=text.replace(before,HOOK,1).replace(include,added,1).replace(CALL,NEW_CALL,1)
    if fixed.replace(NEW_CALL,CALL,1).replace(added,include,1).replace(HOOK,before,1)!=text:
        raise ValueError('locality overlay changes another source byte')
    return fixed.encode()


def compile_source(plan_path, argv):
    """CMake launcher: substitute exactly one TU in the diagnostic build only."""
    plan = json.loads(Path(plan_path).read_text())
    project = Path(plan['project']['root'])
    compiler = plan['compiler']
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if not argv or argv[0] != compiler['path'] or digest(argv[0]) != compiler['sha256']:
        raise ValueError('diagnostic compiler identity differs')
    overlay = plan['overlay']
    if digest(__file__) != overlay['generator_sha256']:
        raise ValueError('overlay launcher changed')
    original = str(project / SOURCE)
    if original in argv:
        if argv.count(original) != 1 or '-c' not in argv:
            raise ValueError('unexpected prefill compilation command')
        target = Path(overlay['path'])
        if target.read_bytes() != generated(project) or digest(target) != overlay['sha256']:
            raise ValueError('generated prefill source changed')
        argv = [str(target) if a == original else a for a in argv]
        argv += ['-iquote', str(project / 'src/prefill')]
        receipt = Path(plan['root']) / 'prefill-compile.json'
        with receipt.open('x') as stream:
            json.dump(dict(argv=argv, source_sha256=overlay['sha256']), stream, indent=2)
    os.execv(argv[0], argv)


if __name__ == '__main__':
    if len(sys.argv) < 5 or sys.argv[1] != 'compile':
        raise SystemExit('CMake launcher only: compile PLAN COMPILER ARG...')
    compile_source(sys.argv[2], sys.argv[3:])
