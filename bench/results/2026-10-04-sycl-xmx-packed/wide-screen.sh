set -e
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -uo pipefail
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0 NEO_CACHE_PERSISTENT=1
export STRATA_SYCL_MMQ_XMX=1 STRATA_SYCL_MMQ_XMX_TILE=8 STRATA_SYCL_MMQ_XMX_EXACT=1
export PERF_EXE="$PWD/build-sycl-aot/strata" PERF_PREFILL=1024 PERF_WORKERS=5 PERF_TIMING=0
for trial in off both wide; do
 export STRATA_SYCL_MMQ_XMX_PACK=1 STRATA_SYCL_MMQ_XMX_PACK_MIN_COLS=0
 if [ "$trial" = off ]; then export STRATA_SYCL_MMQ_XMX_PACK=0; fi
 if [ "$trial" = wide ]; then export STRATA_SYCL_MMQ_XMX_PACK_MIN_COLS=1024; fi
 export PERF_NAME="goal-xmx-wide-short-$trial"
 python3 /tmp/strata-sycl-perf-profile.py
done
for trial in both wide; do
 export STRATA_SYCL_MMQ_XMX_PACK=1 STRATA_SYCL_MMQ_XMX_PACK_MIN_COLS=0
 if [ "$trial" = wide ]; then export STRATA_SYCL_MMQ_XMX_PACK_MIN_COLS=1024; fi
 export PACK_TRIAL="4k-$trial"
 python3 /tmp/strata-sycl-goal-xmx-wide-profile.py 4k 4096
done
