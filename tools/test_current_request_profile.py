"""Host controls for production request/collector separation and clock bounds."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).parent/'sycl'))
import current_request_profile as profile
import engine_once


class CurrentRequestProfileTests(unittest.TestCase):
    def test_actual_fast_32k_route_uses_zero_byte_packed_draft_contract(self):
        text=(profile.P/profile.FIXTURE).read_text();plan=dict(read=32769,heads=1)
        result=profile.request_route(text,plan)
        self.assertEqual(result['rows'],32768)
        self.assertEqual(result['draft_requested_D2H'],0)
        self.assertEqual(result['fast_window']['prefill_slots'],384)
        self.assertEqual(result['MTP_expert_restore_bytes'],707788800)
        for changed in (text.replace('draft detached bytes=0','draft detached bytes=223227900'),
                        text.replace('tokens=32768 chunk=8192','tokens=32768 chunk=4096')):
            with self.assertRaises(RuntimeError):profile.request_route(changed,plan)

    def test_whole_request_keeps_device_capture_from_exec_and_legacy_modes(self):
        config=dict(mode='whole_request',launcher=dict(path='/qualified/unitrace'),session='unused')
        argv=engine_once.profile_command(config,['/qualified/strata','--serve'])
        self.assertEqual(argv[-2:],['/qualified/strata','--serve'])
        self.assertNotIn('--session',argv);self.assertNotIn('--start-paused',argv)
        self.assertNotIn('--host-timing',argv);self.assertNotIn('--chrome-call-logging',argv)
        self.assertIn('--chrome-device-logging',argv)
        self.assertIn('--start-paused',engine_once.profile_command(dict(config,mode='trace'),['exe']))
        self.assertNotIn('--start-paused',engine_once.profile_command(dict(config,mode='aggregate'),['exe']))
        with self.assertRaises(RuntimeError):engine_once.profile_command(dict(config,mode='unknown'),['exe'])

    def test_actual_native_environment_allows_only_declared_collector_changes(self):
        expected=dict(PATH='/usr/bin',LD_PRELOAD='/qualified/shim',ZES_ENABLE_SYSMAN='0')
        with tempfile.TemporaryDirectory() as root:
            plan=dict(arm='trace',root=root)
            prof=dict(collector=dict(path='/qualified/collector'),environment=dict(UNITRACE_SystemTime='1',
                UNITRACE_FollowChildProcess='1',UNITRACE_TraceOutputDir=root))
            Path(root,'plan.json').write_text(json.dumps(dict(native_profile=prof)))
            actual=dict(expected,**prof['environment'],STRATA_UNITRACE_TARGET_PRELOAD='/qualified/shim',
                LD_PRELOAD='/qualified/collector:/qualified/shim',UNITRACE_DeviceTiming='1',
                UNITRACE_ChromeDeviceLogging='1',UNITRACE_ChromeEventBufferSize='128',
                UNITRACE_LD_PRELOAD_OLD='/qualified/shim',ZE_ENABLE_TRACING_LAYER='1',
                INTEL_LIBITTNOTIFY64='libunitrace_tool.so',
                UNITRACE_VERSION='2.4.1 (4b6d74294cdc72766c00a9211fe2e4261ee94244)')
            def wire(env):return {k.encode():v.encode() for k,v in env.items()}
            profile.check_native_environment(wire(actual),expected,plan)
            for key,value in [('ZET_ENABLE_METRICS','1'),('UNITRACE_StartPaused','1'),
                              ('ZES_ENABLE_SYSMAN','1'),('LD_PRELOAD','/qualified/shim:/qualified/collector'),
                              ('INTEL_LIBITTNOTIFY64','/unexpected/libunitrace_tool.so'),
                              ('UNITRACE_VERSION','unqualified')]:
                with self.assertRaises(ValueError):profile.check_native_environment(wire(dict(actual,**{key:value})),expected,plan)
            profile.check_native_environment(wire(expected),expected,dict(arm='clean'))
            with self.assertRaises(ValueError):profile.check_native_environment(wire(actual),expected,dict(arm='clean'))
        self.assertEqual(expected['LD_PRELOAD'],'/qualified/shim')

    def test_request_gate_order_duplicate_and_incomplete_windows_are_rejected(self):
        out={};profile.observe_gate(out,0,'begin',100);profile.observe_gate(out,0,'end',200)
        profile.observe_gate(out,1,'begin',300)
        for request,stage,stamp in [(1,'end',299),(0,'end',400),(2,'begin',400),(1,'begin',400)]:
            with self.assertRaises(ValueError):profile.observe_gate(out,request,stage,stamp)
        profile.observe_gate(out,1,'end',400)
        self.assertEqual(out['external_request_raw_windows'],[dict(begin=100,end=200),dict(begin=300,end=400)])

    def test_enclosing_clock_cannot_reverse_or_be_overwritten(self):
        out={};profile.observe_lifetime(out,'begin',100)
        for stage,stamp in [('begin',101),('end',99),('unexpected',200)]:
            with self.assertRaises(ValueError):profile.observe_lifetime(out,stage,stamp)
        profile.observe_lifetime(out,'end',200)
        with self.assertRaises(ValueError):profile.observe_lifetime(out,'end',201)

    def test_overhead_requires_both_pairs_in_both_first_and_warmed_scopes(self):
        sample=dict(prompt_ms=40000.,ttft_s=40.2,total_s=42.)
        processes=[dict(passed=True,complete=True,samples=[dict(sample) for _ in range(3)]) for _ in range(4)]
        self.assertTrue(profile.decision(processes)['observer_overhead_qualified'])
        for index,request in [(1,0),(2,1),(2,2)]:
            changed=copy.deepcopy(processes);changed[index]['samples'][request]['prompt_ms']*=1.3
            self.assertFalse(profile.decision(changed)['observer_overhead_qualified'])
        for value in (float('nan'),0.,-1.):
            changed=copy.deepcopy(processes);changed[1]['samples'][0]['prompt_ms']=value
            with self.assertRaises(ValueError):profile.decision(changed)
        self.assertFalse(profile.decision(processes)['phase_attribution_qualified'])

    def test_no_postprocessing_without_complete_enclosing_request_windows(self):
        out=dict(clean_speed=True,enclosing_owned_run_raw_ns=dict(begin=10,end=100),
            external_request_raw_windows=[dict(begin=20,end=50),dict(begin=51,end=60),dict(begin=61,end=110)])
        with self.assertRaises(ValueError):profile.after_execution(out,{},dict(arm='clean'))
        out['external_request_raw_windows'][-1]['end']=90
        profile.after_execution(out,{},dict(arm='clean'))
        with self.assertRaises(ValueError):profile.native_context({},dict(index=0,arm='trace',read=32769,heads=1))


if __name__=='__main__':unittest.main()
