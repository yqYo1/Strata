"""Reusable host-only prefill upload/control admission regressions.

Main-authored,repository LICENSE. No engine execution,model result or native
numerical qualification. The closed64Kcampaign and its decision tests live in Git.
"""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'sycl'))
import prefill_workspace_admission as admission


def upload_line(layer):
    return (f'strata prefill fullfit upload: policy=eager layer={layer} first_row=0 '
        'logical_experts=512 accepted_copy_commands=512 requested_H2D_bytes=773324800 '
        'publication=accepted_same_queue native_completion=unchecked_until_outer_drain')


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


if __name__=='__main__':unittest.main()
