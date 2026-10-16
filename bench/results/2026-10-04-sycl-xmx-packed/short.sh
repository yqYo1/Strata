set -e
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -uo pipefail
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0 NEO_CACHE_PERSISTENT=1
export STRATA_SYCL_MMQ_XMX=1 STRATA_SYCL_MMQ_XMX_TILE=8 STRATA_SYCL_MMQ_XMX_EXACT=1
export PERF_EXE="$PWD/build-sycl-aot/strata" PERF_PREFILL=1024 PERF_WORKERS=5
for trial in off on; do
 export STRATA_SYCL_MMQ_XMX_PACK=0
 if [ "$trial" = on ]; then export STRATA_SYCL_MMQ_XMX_PACK=1; fi
 export PERF_NAME="goal-xmx-pack-short-$trial" PERF_TIMING=0
 python3 /tmp/strata-sycl-perf-profile.py
 export PERF_NAME="goal-xmx-pack-prof-$trial" PERF_TIMING=1
 python3 /tmp/strata-sycl-perf-profile.py
done
