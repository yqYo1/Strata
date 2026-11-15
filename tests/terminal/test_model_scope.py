"""The fork's explicit model deferral survives shared upstream control updates."""
import unittest

from strata_control.model_scope import deferred, require_config


class ModelScopeTests(unittest.TestCase):
    def test_deferred_families_and_saved_launch_paths(self):
        for name in ['deepseek41', 'DeepSeek-V4.1-Flash', 'qwen36', 'qwen35moe',
                     '/models/Qwen3.6-35B-A3B', r'C:\models\Qwen-35B-UD-Q4.gguf']:
            with self.subTest(name=name):
                self.assertTrue(deferred(name))
                for config in [dict(arch=name), dict(family=name), dict(model=name),
                               dict(args=['--native', name]), dict(args=['--pack=' + name])]:
                    with self.assertRaisesRegex(ValueError, 'execution is deferred'):
                        require_config(config)

    def test_current_flash_and_quantization_choices_remain_available(self):
        for name in ['qwen4exp', 'qwen3.8-flash-next-iq3_s', 'UD-IQ1_M', 'UD-IQ4_XS-v31',
                     'Qwen3.8-Flash-Next-DFlash2', '/models/flash-135B/pack']:
            with self.subTest(name=name):
                self.assertFalse(deferred(name))
                require_config(dict(model=name, args=['--native', name]))

    def test_setup_catalog_suppresses_execution_families(self):
        import setup
        self.assertFalse(any(deferred(name) or deferred(value.get('arch'))
                             for name, value in setup.OWN_PATH_FAMILIES.items()))
        self.assertFalse(any(deferred(name) or deferred(value.get('arch'))
                             for name, value in setup.MULTI_FILE_FAMILIES.items()))


if __name__ == '__main__':
    unittest.main()
