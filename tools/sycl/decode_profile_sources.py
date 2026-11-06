"""Attach optional drained decode gates to the exact measured generator.

Main-authored; repository LICENSE. Pure source construction only. It preserves
all567 selected files except generate.cpp and adds the host-qualified header.
The four gates are diagnostic; their waits are not clean throughput evidence.
"""
import hashlib
import textwrap

GENERATOR = 'src/program/generate.cpp'
GENERATOR_SHA = 'b4423da4a95f09544599ea20fc82fd830717397fec3cc6f8e28ebc7e83a3e17d'
HEADER = 'include/strata/sycl/decode_profile_gate.hpp'
HEADER_SOURCE = '1a0c9c3c3d690c6da6a94212833c067c30053e0d'
HEADER_SHA = '5bc9b648948631c5dbb5946a0c38b0cf47b6071eebf80f13736652c7917f1642'


def one(value, before, after):
    if value.count(before) != 1:
        raise RuntimeError('one exact decode-profile source anchor required: ' + before[:100])
    return value.replace(before, after, 1)


def block(value, indent=0):
    return textwrap.indent(textwrap.dedent(value).strip() + '\n', ' ' * indent)


def generator(original):
    if hashlib.sha256(original).hexdigest() != GENERATOR_SHA:
        raise RuntimeError('recover the actual567-file native generator;do not refresh its hash')
    value = original.decode()
    anchor = '#ifdef STRATA_ENABLE_SYCL\n#include "strata/sycl/decode_service_profile.hpp"'
    value = one(value, anchor, anchor.replace('#include', '#include "strata/sycl/decode_profile_gate.hpp"\n#include', 1))
    anchor = 'int main(int argc, char** argv) {\n'
    startup = block(r'''
        #ifdef STRATA_ENABLE_SYCL
            // Read once before runtime setup; never mutate the process environment.
            const auto decode_profile_config = strata::sycl_backend::decode_profile_gate::environment();
            if (!decode_profile_config.valid) {
                std::fprintf(stderr, "strata: invalid decode-profile startup pair\n");
                return 2;
            }
        #endif
    ''')
    value = one(value, anchor, anchor + startup)
    anchor = '    if (!have_tokens && o.serve) {   // plan v0.3 P8: requests bring their own tokens\n'
    guard = block(r'''
        #ifdef STRATA_ENABLE_SYCL
            if (decode_profile_config.request != 0 && !o.serve) {
                std::fprintf(stderr, "strata: decode-profile gates require the serving entrypoint\n");
                return 2;
            }
        #endif
    ''')
    value = one(value, anchor, guard + anchor)
    anchor = '            if (cancelled) finish = "cancel";\n            while (!cancelled && produced_n < max_new) {'
    setup = block(r'''
        #ifdef STRATA_ENABLE_SYCL
            strata::sycl_backend::decode_profile_gate::Gate decode_profile_gate(
                decode_profile_config, census_request);
            auto decode_profile_checkpoint = [&](uint64_t completed_rounds) {
                std::string why;
                const bool ok = decode_profile_gate.checkpoint(completed_rounds, [&] {
                    // The caller has joined the adaptive worker. Drain the pending
                    // commit and every registered/runtime queue,including draft/copy.
                    return ver.wait_commit(why) && cudaDeviceSynchronize() == cudaSuccess;
                }, why);
                if (!ok) {
                    std::fprintf(stderr, "strata serve: %s\n", why.c_str());
                    std::printf("ERR decode-profile checkpoint failed\n");
                    std::fflush(stdout); std::fflush(stderr);
                    std::_Exit(1); // A failed completion must not unwind GPU owners.
                }
            };
            decode_profile_checkpoint(0);
        #endif
    ''')
    value = one(value, anchor, setup + anchor)
    anchor = '                if (eos) { finish = "stop"; break; }\n                if (stop_req.load()) { finish = "cancel"; break; }'
    checkpoint = block(r'''
        #ifdef STRATA_ENABLE_SYCL
                decode_profile_checkpoint(uint64_t(dec_windows));
        #endif
    ''')
    value = one(value, anchor, checkpoint + anchor)
    anchor = '            const double decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();'
    complete = block(r'''
        #ifdef STRATA_ENABLE_SYCL
            if (!decode_profile_gate.complete()) {
                std::fprintf(stderr, "strata serve: selected decode-profile request ended before all24rounds\n");
                std::printf("ERR incomplete decode-profile extent\n");
                std::fflush(stdout); std::fflush(stderr);
                std::_Exit(1);
            }
        #endif
    ''')
    value = one(value, anchor, complete + anchor)
    return value.encode()


def projection(selected, header):
    if len(selected) != 567 or HEADER in selected or hashlib.sha256(header).hexdigest() != HEADER_SHA:
        raise RuntimeError('recover exact567-source basis and host-qualified decode gate header')
    result = dict(selected)
    result[GENERATOR] = generator(selected[GENERATOR])
    result[HEADER] = header
    if len(result) != 568 or {n for n in result if result[n] != selected.get(n)} != {GENERATOR, HEADER}:
        raise RuntimeError('unplanned decode-profile source change')
    return result
