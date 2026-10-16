set -e
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -uo pipefail
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0 NEO_CACHE_PERSISTENT=1
for value in 0 1; do
 export STRATA_SYCL_MMQ_XMX_PACK="$value"
 python3 /tmp/strata-sycl-goal-xmx-pack-profile.py 4k 4096
done
