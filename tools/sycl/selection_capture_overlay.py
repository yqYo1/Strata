"""Main-authored checked-dump source for a separate diagnostic native build.

Repository LICENSE applies. The production prefill/decode files and selected
engine stay unchanged. Compile this source with -iquote <project>/src/prefill;
the future pinned build recipe must record this substitution explicitly.
"""
from pathlib import Path
import subprocess

BASE = '6e40afc1beddd059ee518617f6c1efce1da14aca'
SOURCE = 'src/prefill/prefill.cpp'


def generated(project):
    project = Path(project)
    original = subprocess.check_output(['git', '-C', str(project), 'show', BASE + ':' + SOURCE], timeout=20)
    if original != (project / SOURCE).read_bytes():
        raise ValueError('selected production prefill source changed')
    text = original.decode()
    before = '''                            std::vector<int32_t> h((size_t) (T * m.cap));
                            cudaMemcpyAsync(h.data(), m.sel_ids, h.size() * 4, cudaMemcpyDeviceToHost, m.cs);
                            cudaStreamSynchronize(m.cs);
                            if (std::FILE* f = std::fopen(dump, "ab")) {
                                const int32_t hdr[4] = {(int32_t) qsa_index, (int32_t) p0, (int32_t) T, (int32_t) m.cap};
                                std::fwrite(hdr, 4, 4, f);
                                std::fwrite(h.data(), 4, h.size(), f);
                                std::fclose(f);
                            }'''
    after = '''                            if (!detail::selection_dump(dump, qsa_index, p0, T, m.cap, m.sel_ids,
                                    [&](void* dst, const void* src, size_t bytes) {
                                        return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, m.cs) == cudaSuccess;
                                    },
                                    [&] { return cudaStreamSynchronize(m.cs) == cudaSuccess; },
                                    []() noexcept {
                                        strata::sycl_backend::cleanup_fail_stop("QSA selection dump: readback completion uncertain");
                                    }, err)) return false;'''
    include = '#include "strata/prefill/draft_uploads.hpp"'
    added = include + '\n#include "strata/prefill/selection_dump.hpp"'
    if text.count(before) != 1 or text.count(include) != 1:
        raise ValueError('dump overlay anchor is not unique')
    fixed = text.replace(before, after, 1).replace(include, added, 1)
    if fixed.replace(added, include, 1).replace(after, before, 1) != text:
        raise ValueError('overlay changes another native source byte')
    return fixed.encode()
