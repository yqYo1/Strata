set -e
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -uo pipefail
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0 NEO_CACHE_PERSISTENT=1
for trial in off1 on1 on2 off2 off3 on3; do
 export PACK_TRIAL="$trial"
 export STRATA_SYCL_MMQ_XMX_PACK=0
 if [[ "$trial" == on* ]]; then export STRATA_SYCL_MMQ_XMX_PACK=1; fi
 python3 /tmp/strata-sycl-goal-xmx-pack-pairs-profile.py 4k 4096
done
