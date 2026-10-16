set -e
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
set -uo pipefail
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu SYCL_CACHE_PERSISTENT=0 NEO_CACHE_PERSISTENT=1
export STRATA_SYCL_MMQ_XMX=1 STRATA_SYCL_MMQ_XMX_TILE=8 STRATA_SYCL_MMQ_XMX_EXACT=1
for trial in off on; do
 export STRATA_SYCL_MMQ_XMX_PACK=0
 if [ "$trial" = on ]; then export STRATA_SYCL_MMQ_XMX_PACK=1; fi
 python3 tools/sycl/serve_bench.py --config "$HOME/.local/share/strata-sycl/serve-config-iq3_s.json" --prompt writing=bench/results/2026-10-03-sycl-mtp-floor/strata-sycl-writing-tokens.txt --prompt coding=bench/results/2026-10-03-sycl-mtp-floor/strata-sycl-profile-coding-tokens.txt --cancel-during-prefill --timeout 300 --compare /tmp/strata-sycl-goal-xmx-exact8-serve.json --output "/tmp/strata-sycl-goal-xmx-pack-serve-$trial.json"
done
