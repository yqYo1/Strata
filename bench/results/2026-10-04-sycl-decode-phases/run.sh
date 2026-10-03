set -e
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -uo pipefail
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0 NEO_CACHE_PERSISTENT=1
python3 /tmp/strata-sycl-goal-decode-phases.py
