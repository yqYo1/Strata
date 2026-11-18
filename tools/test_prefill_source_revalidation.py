"""Host-only source/upload comparison admission regressions;repository LICENSE.

Main-authored synthetic gate fixtures. No engine execution,model result or
native numerical qualification. Preserve failed units despite a faster net arm.
"""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import prefill_64k_revalidate as campaign
import prefill_workspace_admission as admission


def upload_line(layer):
    return (f'strata prefill fullfit upload: policy=eager layer={layer} first_row=0 '
        'logical_experts=512 accepted_copy_commands=512 requested_H2D_bytes=773324800 '
        'publication=accepted_same_queue native_completion=unchecked_until_outer_drain')


def cohort():
    values=[]
    # A coherent-source loss and a larger eager win must remain separate.
    times={'original':64000.,'jit':65000.,'eager':63000.}
    for i,arm in enumerate(campaign.ARMS):
        upload=campaign.selection(i)['upload']
        samples=[dict(actual_read_tokens=65537,actual_reused_tokens=0,generated_tokens=24,
            request_sha256=f'{j:064x}',prompt_ms=times[arm],ttft_s=times[arm]/1000+.1)
            for j in range(3)]
        values.append(dict(index=i,kernel=arm,read=65537,heads=1,clean_speed=i>=3,
            passed=True,complete=True,ownership_closed=True,native_exit_code=0,fresh_faults=[],
            boot_unchanged=True,native_identity=dict(pid=i+100,start_ticks=i+1000),
            samples=samples,subsequent_median_ms=times[arm],
            routes=[dict(layout=dict(free_gpu_bytes=4000000000-i),jobs=12000,
                fullfit_upload=dict(policy=upload)) for _ in samples]))
    return values


class UploadAdmission(unittest.TestCase):
    def test_jit_and_actual_individual_copies(self):
        self.assertEqual(admission.eager_route('',dict(upload='jit'))['fullfit_layers'],0)
        text='\n'.join(upload_line(i) for i in range(24))
        result=admission.eager_route(text,dict(upload='eager'))
        self.assertEqual(result['accepted_copy_commands'],24*512)
        self.assertEqual(result['requested_fullfit_H2D_bytes'],24*773324800)
        self.assertFalse(result['physical_bus_traffic_or_overlap_measured'])

    def test_unknown_malformed_missing_and_duplicate_dispatch_fail(self):
        text='\n'.join(upload_line(i) for i in range(24))
        invalid=[text+'\n'+upload_line(0),text.replace('layer=23 ','layer=22 '),
            text.replace('first_row=0','first_row=1',1),
            text.replace('accepted_copy_commands=512','accepted_copy_commands=1',1),
            text.replace('requested_H2D_bytes=773324800','requested_H2D_bytes=773324801',1),
            text.replace('native_completion=unchecked_until_outer_drain','native_completion=checked',1),
            '\n'.join(text.splitlines()[:-1])]
        for value in invalid:
            with self.subTest(value=value[-90:]):
                with self.assertRaises(RuntimeError):admission.eager_route(value,dict(upload='eager'))
        with self.assertRaises(RuntimeError):admission.eager_route(text,dict(upload='jit'))
        with self.assertRaises(RuntimeError):admission.eager_route(text,dict(upload='unknown'))


class HistoricalControlAdmission(unittest.TestCase):
    def test_only_the_exact_archived_cli_may_be_absent(self):
        old=admission.RETIRED_CONTROL;live='src/live.hpp'
        originals={old:b'closed CLI',live:b'unchanged native'}
        pins={p:hashlib.sha256(raw).hexdigest() for p,raw in originals.items()}
        source=admission.device.build.source
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/live).parent.mkdir();(root/live).write_bytes(originals[live])
            def current(*args):
                return dict(source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest()
                    for p in args[1] if (root/p).is_file()})
            with patch.object(admission,'P',root),patch.object(source,'blobs',return_value=originals.items()),\
                    patch.object(source,'git',return_value=b''),patch.object(source,'pin',side_effect=current):
                self.assertEqual(admission.historical_controls(pins,'old')['unchanged_live_files'],1)
                (root/live).write_bytes(b'changed native')
                with self.assertRaises(RuntimeError):admission.historical_controls(pins,'old')
                (root/live).unlink()
                with self.assertRaises(RuntimeError):admission.historical_controls(pins,'old')
                bad=dict(pins);bad[old]='0'*64
                with self.assertRaises(RuntimeError):admission.historical_controls(bad,'old')


class SourceComparisonAdmission(unittest.TestCase):
    def test_unit_loss_survives_net_gain_and_first_use_loss(self):
        values=cohort()
        for v in values:
            if v['kernel']=='eager':v['samples'][0]['prompt_ms']=80000.
        result=campaign.decision(values)
        self.assertFalse(result['units']['coherence']['performance_eligible'])
        self.assertTrue(result['units']['eager']['performance_eligible'])
        self.assertTrue(result['units']['combined']['performance_eligible'])
        self.assertTrue(all(v>1 for v in result['units']['eager']['first_time_ratios']))
        self.assertTrue(result['arms']['eager']['all_warmed_processes_at_least1000'])
        self.assertFalse(result['inference_adopted'])
        self.assertFalse(result['current9849decode_preserved'])

    def test_one_slow_pair_or_ttft_loss_cannot_hide_in_aggregate(self):
        values=cohort();values[11]['subsequent_median_ms']=66000.
        for s in values[11]['samples'][1:]:s['prompt_ms']=66000.
        result=campaign.decision(values)
        self.assertLess(result['units']['eager']['geometric_mean_time_ratio'],1)
        self.assertFalse(result['units']['eager']['performance_eligible'])
        values=cohort()
        for v in values:
            if v['kernel']=='eager':
                for s in v['samples'][1:]:s['ttft_s']=70.
        self.assertFalse(campaign.decision(values)['units']['eager']['performance_eligible'])

    def test_order_identity_work_length_and_median_changes_fail(self):
        mutations=[lambda v:v[4].update(kernel='eager'),
            lambda v:v[4].update(native_identity=v[3]['native_identity']),
            lambda v:v[4]['routes'][0].update(jobs=12001),
            lambda v:v[4]['routes'][0]['fullfit_upload'].update(policy='eager'),
            lambda v:v[4]['samples'][0].update(request_sha256='f'*64),
            lambda v:v[4]['samples'][0].update(actual_reused_tokens=1),
            lambda v:v[4]['samples'][0].update(generated_tokens=23),
            lambda v:v[4].update(subsequent_median_ms=64999.),
            lambda v:v[4].update(clean_speed=False),
            lambda v:v[4].update(boot_unchanged=False)]
        for mutate in mutations:
            values=cohort();mutate(values)
            with self.assertRaises(RuntimeError):campaign.decision(values)
        with self.assertRaises(RuntimeError):campaign.decision(cohort()[:-1])

    def test_declared_shared_controls_cover_both_orders(self):
        expected={'coherence':('original','jit'),'eager':('jit','eager'),
            'combined':('original','eager')}
        for name,pairs in campaign.PAIRS.items():
            self.assertEqual(len(pairs),3)
            self.assertEqual({campaign.ARMS[a] for a,b in pairs},{expected[name][0]})
            self.assertEqual({campaign.ARMS[b] for a,b in pairs},{expected[name][1]})
            self.assertEqual({a<b for a,b in pairs},{True,False})


if __name__=='__main__':unittest.main()
