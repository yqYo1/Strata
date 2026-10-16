"""CPU-only native startup protocol fixtures; no provider or GPU imports."""
import importlib.util
import json
from pathlib import Path

A = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prefill_window_v2', A/'run_prefill_window_profile_v2.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Input:
    def __init__(self, lines):
        self.lines = list(lines)

    def line(self, seconds):
        assert 0 < seconds <= 300
        if not self.lines:
            raise TimeoutError('fixture EOF before READY')
        return self.lines.pop(0)


def main():
    path = module.B/'main-prefill-window-profile-component-root-v1/owned/request-profile.stdout'
    real = path.read_text().splitlines()
    assert len(real) == 1 and real[0].startswith('INFO ')
    info = real[0]
    out = {}
    module.read_ready(Input([info, 'READY 262144 stop']), out, 30)
    assert out['native_INFO']['context'] == '262144'
    assert out['native_INFO']['engine'] == '0.1.38'
    cases = 1
    bad = [[info], ['READY 262144 stop'], [info, 'ERROR startup'],
           [info, info], [info+' context=262144', 'READY 262144 stop'],
           [info+' badfield', 'READY 262144 stop'],
           [info+' empty=', 'READY 262144 stop'],
           [info, 'READY 65536 stop']]
    for key, value in dict(context='65536', kv='q4_0', kv_resident='20480',
                           expert_slots='64', spec='1', pool_workers='4', pcie_frac='1.00').items():
        changed = ' '.join(k+'='+value if k == key else item
                           for item in info.split() for k in [item.split('=', 1)[0]])
        bad.append([changed, 'READY 262144 stop'])
    for lines in bad:
        out = {}
        try:
            module.read_ready(Input(lines), out, 30)
        except (RuntimeError, TimeoutError):
            assert not out
        else:
            raise AssertionError('invalid effective configuration accepted')
        cases += 1
    print(json.dumps(dict(passed=True, cases=cases, actual_failed_v1_INFO_used=True,
                         GPU_executed=False, provider_executed=False)), flush=True)


if __name__ == '__main__':
    main()
