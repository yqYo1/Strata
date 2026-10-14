from pathlib import Path
import difflib, hashlib, json
root=Path('.')
out=root/'bench/results/2026-10-14-pre-Xe-model-compare/separate-copy-proposal'
p=root/'src/prefill/prefill.cpp'; s=p.read_text()
a='''#if defined(STRATA_ENABLE_SYCL)
    // Keep prefill transfers and GEMMs on one in-order queue. On the B570 with
    // oneAPI 2026.1, the two-queue ring stalls at realistic expert matrix sizes.
    // Host staging remains parallel; the compute queue belongs to the caller.
    m.copy = m.cs;
#else
'''
b='''#if defined(STRATA_ENABLE_SYCL)
    // Default stays on the caller's in-order queue: the B570/oneAPI 2026.1
    // two-queue ring stall is unresolved. A private queue is experimental.
    m.copy = m.cs;
    const char* separate_copy = std::getenv("STRATA_SYCL_PREFILL_SEPARATE_COPY");
    if (separate_copy && std::strcmp(separate_copy, "1") == 0) {
        // The compat shim owns construction/registration, using the runtime's
        // same context/device and error handler. Publish ownership only on success.
        cudaStream_t private_copy = nullptr;
        const cudaError_t copy_status = cudaStreamCreateWithFlags(&private_copy, cudaStreamNonBlocking);
        if (copy_status != cudaSuccess) {
            err = std::string("prefill: separate copy stream: ") + cudaGetErrorString(copy_status);
            std::fprintf(stderr, "strata prefill copy: requested=separate mode=failed compute=%p error=%s\\n",
                         (void*) m.cs, cudaGetErrorString(copy_status));
            return false;
        }
        m.copy = private_copy; // release drains it and destroys only copy != cs
        std::fprintf(stderr, "strata prefill copy: requested=separate mode=separate compute=%p copy=%p cross-queue-events=required\\n",
                     (void*) m.cs, (void*) m.copy);
    } else if (separate_copy) {
        std::fprintf(stderr, "strata prefill copy: requested=non-exact mode=same reason=requires-exact-1 compute=%p copy=%p events=existing-policy\\n",
                     (void*) m.cs, (void*) m.copy);
    }
#else
'''
assert s.count(a)==1
candidate=s.replace(a,b,1)
patch=''.join(difflib.unified_diff(s.splitlines(True),candidate.splitlines(True),fromfile='a/src/prefill/prefill.cpp',tofile='b/src/prefill/prefill.cpp'))
(out/'separate-copy.patch').write_text(patch)
sha=lambda b:hashlib.sha256(b).hexdigest()
x={'status':'proposal-only-unapplied-unbuilt-untested-unmeasured','base_head_supplied':'4808d87e','source_path':str(p),'source_sha256':sha(p.read_bytes()),'derived_source_sha256':sha(candidate.encode()),'patch_sha256':sha(patch.encode()),'contracts':{str(q):sha(q.read_bytes()) for q in [root/'src/sycl/compat.cpp',root/'src/sycl/runtime.cpp',root/'include/strata/sycl/launch.hpp',root/'include/strata/prefill/prefill.hpp']}}
(out/'hashes.json').write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps(x,indent=2))
