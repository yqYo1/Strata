"""Offline, no-GPU checks for the native setup adapter. Run with Python 3."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("native_setup", ROOT / "tools/setup_intel.py")
I = importlib.util.module_from_spec(spec)
spec.loader.exec_module(I)


class NativeSetupTests(unittest.TestCase):
    def test_host_paths_outside_checkout_and_spaces(self):
        with tempfile.TemporaryDirectory(prefix="model path ") as folder:
            self.assertEqual(I.host_path(folder), str(Path(folder).resolve()))
            build, exe = I.native_paths({"STRATA_SYCL_BUILD_DIR": folder})
            self.assertEqual(exe, build / I.S.EXE)
            self.assertNotIn("/work/", str(exe))

    def test_config_preserves_engine_options_and_vision(self):
        cfg = dict(args=["--pack", "/models/my pack", "--max-context", "8192"],
                   env={"USER_TUNING": "1"}, sampling={"temperature": 0.7})
        vision = {"exe": "/encoder", "mmproj": "/models/custom.gguf", "gpu": True}
        out = I.to_native(cfg, Path("/host/strata"), {**cfg, "vision": vision})
        self.assertEqual(out["args"], cfg["args"])
        self.assertEqual(out["vision"], vision)
        self.assertEqual(out["env"], cfg["env"])
        self.assertEqual(out["sampling"], cfg["sampling"])
        self.assertEqual(out["backend"], "sycl")
        self.assertNotIn("--stream-experts", out["args"])

    def test_pending_flag_refuses_config(self):
        with self.assertRaisesRegex(ValueError, "does not support"):
            I.validate_config({"args": ["--stream-experts"]}, {"--pack"})
        with self.assertRaisesRegex(ValueError, "would be lost"):
            I.to_native({"args": ["--pack", "/pack"]}, Path("/strata"),
                        {"args": ["--pack", "/pack", "--pool-workers", "6"]})

    def test_remote_host_requires_key(self):
        with self.assertRaisesRegex(ValueError, "requires --api-key"):
            I.validate_config({"args": [], "host": "0.0.0.0"}, set())
        I.validate_config({"args": [], "host": "0.0.0.0", "api_key": "test-only"}, set())

    def test_delegation_preserves_arguments(self):
        args = ["--backend=sycl", "--data-dir", "/host/data with spaces", "--vision", "gpu"]
        self.assertEqual(I.strip_backend(args), args[1:])
        with patch.object(I.S.subprocess, "call", return_value=7) as call:
            self.assertEqual(I.S.sycl_setup(args), 7)
            self.assertEqual(call.call_args.args[0][2:], args)
            self.assertEqual(Path(call.call_args.args[0][1]), ROOT / "tools/setup_intel.py")

    def test_actual_ram_remains_upstream_function(self):
        ram = I.S.ram_gb
        with patch.object(I, "intel_gpus", return_value=[dict(index=0, name="Intel test", vram_gb=8, arch="xe")]), \
             patch.object(I.S, "main", return_value=0), patch.object(I.sys, "argv", []), \
             patch.multiple(I.S, say=I.S.say, gpus=I.S.gpus, amd_gpus=I.S.amd_gpus,
                            amd_problem=I.S.amd_problem, hip_vision=I.S.hip_vision,
                            build_engine_hip=I.S.build_engine_hip, cuda_lib_dirs=I.S.cuda_lib_dirs,
                            hipblaslt_table=I.S.hipblaslt_table, write_setup_config=I.S.write_setup_config,
                            write_config=I.S.write_config,
                            upgrade_config=I.S.upgrade_config, update_installed_engine=I.S.update_installed_engine,
                            setup_calibration=I.S.setup_calibration, installed_configs=I.S.installed_configs):
            self.assertEqual(I.install(["--backend", "sycl", "--check"]), 0)
            self.assertIs(I.S.ram_gb, ram)
            self.assertEqual(I.sys.argv[-2:], ["--backend", "hip"])

    def test_build_commands_use_native_tree(self):
        commands = I.build_commands(Path("/tmp/build"), "/compiler/icpx", Path("/llama"))
        self.assertIn("-DSTRATA_ENABLE_SYCL=ON", commands[0])
        self.assertIn("-DSTRATA_GGML_DIR=/llama", commands[0])
        self.assertNotIn("docker", str(commands))
        self.assertEqual(commands[1][3:5], ["--target", "strata"])

    def test_normal_native_start_and_update_delegate_before_cuda_updater(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = root / "strata-native.json"
            cfg.write_text(json.dumps({"backend": "sycl"}))
            for args in (["--no-browser"], ["--update"]):
                with self.subTest(args=args), \
                     patch.object(I.sys, "argv", ["setup.py", *args]), \
                     patch.object(I.S, "say"), patch.object(I.S, "data_folder", return_value=(root, [])), \
                     patch.object(I.S, "installed_configs", return_value=[cfg]), \
                     patch.object(I.S, "sycl_setup", return_value=17) as delegate, \
                     patch.object(I.S, "update_install") as updater, \
                     patch.object(I.S, "start") as start:
                    self.assertEqual(I.S.main(), 17)
                    delegate.assert_called_once_with(args)
                    updater.assert_not_called()
                    start.assert_not_called()

    def test_previous_native_install_delegates_before_adoption(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = root / "strata-native.json"
            cfg.write_text(json.dumps({"backend": "sycl"}))
            with patch.object(I.sys, "argv", ["setup.py"]), patch.object(I.S, "say"), \
                 patch.object(I.S, "data_folder", return_value=(root, [])), \
                 patch.object(I.S, "installed_configs", return_value=[]), \
                 patch.object(I.S, "load_settings", return_value={}), \
                 patch.object(I.S, "previous_config", return_value=cfg), \
                 patch.object(I.S, "sycl_setup", return_value=19) as delegate, \
                 patch.object(I.S, "choices_from_config") as choices:
                self.assertEqual(I.S.main(), 19)
                delegate.assert_called_once_with([])
                choices.assert_not_called()


if __name__ == "__main__":
    unittest.main()
