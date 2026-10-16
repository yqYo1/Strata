#!/home/yayoi/.local/share/strata-sycl/venv/bin/python
"""Use the existing bounded wrapper with an explicitly pinned legacy native."""
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

D = Path('/home/yayoi/orca/workspaces/Strata/sycl/bench/results/2026-10-14-primary-model-baseline')
ENGINE = Path('/home/yayoi/.local/state/strata-sycl/post-reboot-tuning-20261007/main-pre-Xe-1eb-build-root-v2/build/strata')
EXPECTED = '4a7e9eeeb5ddea917c8dc678f0df0ad4b5a181f8492c875bdcda5cb3026079bc'


def main():
    with ENGINE.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == EXPECTED
    filename = 'engine_wrapper_v5.py' if os.environ.get('STRATA_MEASUREMENT_WRAPPER_DEBUG') == '1' else 'engine_wrapper_v3.py'
    sys.path.insert(0, str(D))
    spec = importlib.util.spec_from_file_location('bounded_reference_wrapper', D / filename)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    wrapper.ENGINE = str(ENGINE)
    wrapper.main()


if __name__ == '__main__':
    main()
