"""GPU-dispatch provenance regressions from oneMKL's actual UR interop trace. MIT."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/sycl'))
from measure_gdn_storage import launch_census

NATIVE = '<--- urKernelCreateWithNativeHandle(.hNativeKernel = 0x11, .hContext = 0x12, .hProgram = 0x13, .phKernel = 0x14 (0x15)) -> UR_RESULT_SUCCESS;'
NAMED = '<--- urKernelCreate(.hProgram = 0x13, .pKernelName = 0x16 (_ZTS_convert), .phKernel = 0x14 (0x17)) -> UR_RESULT_SUCCESS;'
LAUNCH = '<--- urEnqueueKernelLaunchWithArgsExp(.hQueue = 0x18, .hKernel = {}, .workDim = 3) -> UR_RESULT_SUCCESS;'
QUERY = '<--- urKernelGetInfo(.hKernel = 0x15, .propName = UR_KERNEL_INFO_FUNCTION_NAME, .propSize = 37, .pPropValue = 0x19 (gemm_f16f16f32_nocopy_tn_4x16_1x1x16), .pPropSizeRet = nullptr) -> UR_RESULT_SUCCESS;'


class TraceProvenance(unittest.TestCase):
    def test_native_import_has_no_invented_name(self):
        row = launch_census(NATIVE+'\n'+LAUNCH.format('0x15'))
        self.assertEqual(row['launches'],1)
        self.assertEqual(row['per_kernel'],{'native_import_1':{'creation':'native_import','name':None,'launches':1}})

    def test_named_conversion_and_native_import_coexist(self):
        row = launch_census('\n'.join((NATIVE,NAMED,LAUNCH.format('0x15'),LAUNCH.format('0x17'))))
        self.assertEqual(row['launches'],2)
        self.assertEqual(row['per_kernel']['_ZTS_convert']['name'],'_ZTS_convert')

    def test_reused_native_handle_keeps_distinct_imports(self):
        row = launch_census('\n'.join((NATIVE,LAUNCH.format('0x15'),
            NATIVE.replace('0x11','0x21'),LAUNCH.format('0x15'),LAUNCH.format('0x15'))))
        self.assertEqual([v['launches'] for v in row['per_kernel'].values()],[1,2])

    def test_existing_function_query_names_prior_launch(self):
        row = launch_census('\n'.join((NATIVE,LAUNCH.format('0x15'),QUERY)))
        self.assertEqual(row['per_kernel']['native_import_1']['name'],'gemm_f16f16f32_nocopy_tn_4x16_1x1x16')

    def test_size_only_or_failed_query_does_not_invent_name(self):
        size_only = QUERY.replace('.pPropValue = 0x19 (gemm_f16f16f32_nocopy_tn_4x16_1x1x16)', '.pPropValue = nullptr')
        for query in (size_only,QUERY.replace('UR_RESULT_SUCCESS','UR_RESULT_ERROR_UNKNOWN')):
            row = launch_census('\n'.join((NATIVE,query,LAUNCH.format('0x15'))))
            self.assertIsNone(row['per_kernel']['native_import_1']['name'])

    def test_unknown_and_failed_provenance_refuse(self):
        for text in (LAUNCH.format('0x15'),
                     NATIVE.replace('UR_RESULT_SUCCESS','UR_RESULT_ERROR_UNKNOWN')+'\n'+LAUNCH.format('0x15'),
                     NATIVE+'\n'+LAUNCH.format('0x15').replace('UR_RESULT_SUCCESS','UR_RESULT_ERROR_UNKNOWN'),
                     NATIVE+'\n'+LAUNCH.format('0x99')):
            with self.subTest(text=text),self.assertRaises(RuntimeError): launch_census(text)


if __name__ == '__main__': unittest.main()
