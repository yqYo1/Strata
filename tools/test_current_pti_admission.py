"""Direct source/manifest recovery without loading PTI or requiring old ELFs."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / 'sycl'))
import current_pti_admission as admission


def sha(value): return hashlib.sha256(value).hexdigest()


class CurrentPTIAdmissionTests(unittest.TestCase):
    def test_original_launcher_is_reconstructed_without_the_retired_file(self):
        recipe = b"PATCHES=(('alpha','a'),('beta','b'),('gamma','c'))\n"
        owner = dict(source_commit='1'*40, input_sha256={'tools/sycl/unitrace_launch.py':sha(recipe)},
            generated_source_sha256=sha(b'a b c'))
        def blob(repo, revision, name):
            return recipe if name == 'tools/sycl/unitrace_launch.py' else b'alpha beta gamma'
        with patch.object(admission, 'blob', side_effect=blob):
            self.assertEqual(admission.recover_launcher(owner), sha(b'a b c'))
            for field, value in [('generated_source_sha256', '0'*64),
                                 ('input_sha256', {'tools/sycl/unitrace_launch.py':'0'*64})]:
                with self.assertRaises(ValueError): admission.recover_launcher(dict(owner, **{field:value}))
        with patch.object(admission, 'blob', side_effect=lambda r,v,n:
                recipe if n == 'tools/sycl/unitrace_launch.py' else b'alpha alpha beta gamma'):
            with self.assertRaises(ValueError): admission.recover_launcher(owner)

    def test_launcher_recipe_must_be_literal_and_have_the_exact_three_fixes(self):
        for recipe in (b"PATCHES=tuple()", b"PATCHES=(('alpha','a'),)"):
            owner = dict(source_commit='1'*40, input_sha256={'tools/sycl/unitrace_launch.py':sha(recipe)},
                generated_source_sha256=sha(b'a'))
            with patch.object(admission, 'blob', return_value=recipe):
                with self.assertRaises(ValueError): admission.recover_launcher(owner)

    def test_manifest_refuses_missing_changed_duplicate_and_truncated_inputs(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / 'input'; source.write_bytes(b'original')
            manifest = Path(root) / 'inputs.json'
            rows = [dict(path=str(source), sha256=sha(b'original'))]
            def pin(values, count=1):
                manifest.write_text(json.dumps(values))
                return dict(path=str(manifest), sha256=admission.digest(manifest), files=count)
            self.assertEqual(admission.verify_manifest(pin(rows)), rows)
            for values, count in [(rows+rows, 2), ([], 1), ([dict(rows[0],sha256='0'*64)], 1),
                                  ([dict(rows[0],path=str(source)+'-absent')], 1)]:
                with self.assertRaises(ValueError): admission.verify_manifest(pin(values,count))
            valid = pin(rows); source.write_bytes(b'changed')
            with self.assertRaises(ValueError): admission.verify_manifest(valid)
            with self.assertRaises(ValueError): admission.verify_manifest(dict(valid,sha256='0'*64))

    def test_history_source_must_match_recorded_revision_hashes(self):
        record = dict(source_commit='1'*40,input_sha256={'one.py':sha(b'old')})
        with patch.object(admission, 'blob', return_value=b'old') as call:
            self.assertEqual(admission.recover_inputs(record), 1)
            self.assertEqual(call.call_args.args[1], '1'*40)
        with patch.object(admission, 'blob', return_value=b'current'):
            with self.assertRaises(ValueError): admission.recover_inputs(record)


if __name__ == '__main__': unittest.main()
