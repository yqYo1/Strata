"""Prepare a source-only typed oneMKL call for known host FP32 GEMM scalars."""
from pathlib import Path
import datetime, difflib, hashlib, json

base = Path(__file__).parent
root = Path('/home/yayoi/ghq/github.com/yqYo1/Strata/.worktree/sync-upstream-2026-10-05')
source = root/'build-sycl-event-ack-registered-copy-v3-20261008/source/sycl/src/prefill/gemm.dp.cpp'
header = root/'build-sycl-dpct-profile-definition-v3-20261008/include/dpct'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(source) == '11a2c943ccc323c328b58d99119d77b3b4ed95c92a0f680480bf5ede96bcb705'
original = source.read_text()
s = original
guard = '#if defined(__INTEL_MKL__) && !defined(DPCT_USM_LEVEL_NONE) && !defined(__HIPCC__)'
helper = '''// These two callers own ordinary host FP32 alpha/beta values. Keep the
// DPCT descriptor's queue, compute mode and oneMKL call; skip scalar USM lookup.
namespace {
template <typename Input>
void gemm_host_scalars(dpct::blas::descriptor_ptr desc,
                       oneapi::mkl::transpose a_trans,
                       oneapi::mkl::transpose b_trans, int m, int n, int k,
                       float alpha, const void* a, int lda, const void* b,
                       int ldb, float beta, float* c, int ldc) {
    sycl::queue q = desc->get_queue();
    const auto cm = dpct::blas::deduce_compute_mode(
        dpct::compute_type::f32, desc->get_math_mode(), false);
    oneapi::mkl::blas::column_major::gemm(
        q, a_trans, b_trans, m, n, k, alpha,
        reinterpret_cast<const Input*>(a), lda,
        reinterpret_cast<const Input*>(b), ldb, beta, c, ldc, cm);
}
}  // namespace
'''
marker = 'void ck(int s, const char *what) {'
assert s.count(marker) == 1
s = s.replace(marker,guard+'\n'+helper+'#endif\n\n'+marker)
for kind, ty, ldx, what in [('bfloat16','oneapi::mkl::bfloat16','ldx','cublasGemmEx'),('half','sycl::half','K','cublasGemmEx f16')]:
    old = f'''    ck(DPCT_CHECK_ERROR(dpct::blas::gemm(
           (dpct::blas::descriptor_ptr)handle_, oneapi::mkl::transpose::trans,
           oneapi::mkl::transpose::nontrans, (int)N, (int)T, (int)K, &alpha, W,
           dpct::library_data_t::real_{kind}, (int)K, X,
           dpct::library_data_t::real_{kind}, (int){ldx}, &beta, Y,
           dpct::library_data_t::real_float, (int)ldy,
           dpct::compute_type::f32)),
       "{what}");'''
    new = f'''    ck(DPCT_CHECK_ERROR(gemm_host_scalars<{ty}>(
           (dpct::blas::descriptor_ptr)handle_, oneapi::mkl::transpose::trans,
           oneapi::mkl::transpose::nontrans, (int)N, (int)T, (int)K, alpha, W,
           (int)K, X, (int){ldx}, beta, Y, (int)ldy)),
       "{what}");'''
    assert s.count(old) == 1, kind
    s = s.replace(old,guard+'\n'+new+'\n#else\n'+old+'\n#endif')
assert s.count('ck(DPCT_CHECK_ERROR(gemm_host_scalars<') == 2
assert s.count('desc->get_queue()') == 1 and s.count('desc->get_math_mode()') == 1
assert s.count('DPCT_PROFILING_ENABLED') == original.count('DPCT_PROFILING_ENABLED')
inputs = {str(p):sha(p) for p in [source,header/'blas_utils.hpp',header/'detail/blas_utils_detail.hpp',
                                  header/'detail/lib_common_utils_detail.hpp',header/'detail/memory_detail.hpp']}
out = base/'gemm-host-scalars-v01402-source-v1'
out.mkdir()
target = out/'gemm.dp.cpp'
target.write_text(s)
(out/'candidate.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),
                                                               fromfile=str(source),tofile=str(target))))
record = {'active':False,'prepared':True,'compiled':False,'gpu_tested':False,'adopted':False,
          'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preparer_sha256':sha(__file__),
          'source_inputs_sha256':inputs,'candidate_source':str(target),'candidate_source_sha256':sha(target),
          'intended_baseline':'Qualified DD5; separate from the QSA candidate currently executing its full lifecycle.',
          'changed_default_call_sites':2,'minimum_performance_input_tokens':32768,
          'source_contract_review':{
              'same_oneMKL_column_major_GEMM':True,'same_descriptor_queue_copy':True,
              'same_math_mode_getter_and_f32_deduction':True,'same_int_dimensions_and_leading_dimensions':True,
              'same_BF16_or_F16_inputs_and_FP32_output':True,'same_host_FP32_alpha_beta_values':True,
              'same_transposed_W_nontransposed_X':True,'same_DPCT_CHECK_ERROR_and_ck_boundary':True,
              'same_default_event_dependency_arguments':True,'no_barrier_wait_event_or_allocation_added':True,
              'queue_event_storage_lifetimes_unchanged':True,'unsupported_library_buffer_or_HIP_falls_back':True},
          'observations':[
              'The original detail::get_value checks sycl::get_pointer_type through get_pointer_attribute. Ordinary stack alpha/beta are host-only and are dereferenced; this path does not wait or transfer scalars.',
              'Only device-only scalar pointers use memcpy(...).wait() in that helper; the two changed default callers do not supply device scalar pointers.',
              'This source candidate removes two scalar pointer-type lookups and dynamic type dispatch, with the same typed oneMKL operation. Compiler elimination and runtime cost are not measured.',
              'All matrix allocations, owned queue completion, graph/staging/error machinery and launch ordering are outside the changed blocks.'],
          'limitations':[
              'Source review is not a compile, device numerical proof or measured speed gain.',
              'No claim that this wrapper caused earlier hangs or dominant transfer wait.',
              'Wait for the current owned GPU job to terminate before any engine build or GPU execution.',
              'Before adoption require logged fresh>=32768 math/disk checks, matched clean>=32768 timing with first/later reads separate, and this binary own complete physical256K lifecycle.']}
(out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'prepared':True,'compiled':False,'candidate_sha256':sha(target),'changed_default_call_sites':2}))
