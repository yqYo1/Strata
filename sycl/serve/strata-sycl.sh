#!/usr/bin/env bash
# strata-sycl.sh: the SYCL-built engine as a drop-in `exe` for serve/server.py (--engine strata). The port's binary
# needs the oneAPI runtime of the dev image, so this runs it there with stdin/stdout attached (the serve protocol is
# lines on those pipes) and stderr going to the server's log. Paths in the config's args are the container's:
# the data root is mounted at /work.
#   STRATA_SYCL_ROOT   host directory mounted at /work        (default: two levels above this script's repo)
#   STRATA_SYCL_IMAGE  the runtime image                      (default: strata-sycl-dev)
#   STRATA_SYCL_BIN    the engine binary, relative to the repo (default: build-sycl-aot/strata)
#   STRATA_SYCL_NAME   the container's name                   (default: strata-sycl-serve)
# Passed into the container: every variable of this script's environment that starts with STRATA_ (the config's
# "env" block arrives that way: the server puts it in the engine's environment), ONEAPI_ (level_zero:* for a
# two-card split, #423), UR_, IGC_, SYCL_ or ZES_. The port's own defaults (below) are overridden the same way,
# e.g. "env": {"STRATA_PREFILL_FIRST": "0"} in the config.
# EnableDirectSubmission and NEOReadDebugKeys are forwarded too; driver/runtime values of 0 are retained.
set -euo pipefail
# Reject the retired path before removing a container or opening the GPU.
if [ -n "${STRATA_VERIFY_NO_HOST+x}" ] ||
   { [ -n "${STRATA_SYCL_HOST_BOUNDARY+x}" ] && [ "$STRATA_SYCL_HOST_BOUNDARY" != "1" ]; }; then
    echo 'SYCL device-spin verification is disabled; unset STRATA_VERIFY_NO_HOST and use the default host boundaries.' >&2
    exit 2
fi
here=$(cd "$(dirname "$0")/../.." && pwd)                 # the repo
root=${STRATA_SYCL_ROOT:-$(dirname "$here")}
repo_in=/work/$(basename "$here")
name=${STRATA_SYCL_NAME:-strata-sycl-serve}
docker rm -f "$name" >/dev/null 2>&1 || true              # a container left behind by a killed server
args=""
for a in "$@"; do args+=" $(printf '%q' "$a")"; done
# Runtime defaults for this fork's host-completion verifier.
declare -A setting=([STRATA_VERIFY_DEVICE_PLAN]=1 [STRATA_STAGER_THREADS]=12)
while IFS= read -r k; do
    case "$k" in
        STRATA_SYCL_*) ;;                                 # this script's own settings
        STRATA_*|ONEAPI_*|UR_*|IGC_*|SYCL_*|ZES_*|NEOReadDebugKeys|EnableDirectSubmission|OverrideDefaultFP64Settings) setting[$k]=${!k} ;;
    esac
done < <(compgen -e)
envs=()
for k in "${!setting[@]}"; do
    case "$k" in
        STRATA_PREFILL_FIRST|STRATA_STAGER_THREADS)
            # Numeric controls: explicit 0 differs from an omitted value.
            envs+=(-e "$k=${setting[$k]}") ;;
        STRATA_*)
            # Keep the established handling of Strata switches tested by presence.
            case "${setting[$k]}" in 0|"") ;; *) envs+=(-e "$k=${setting[$k]}") ;; esac ;;
        *)
            # SYCL, UR and driver settings interpret their values, including 0.
            envs+=(-e "$k=${setting[$k]}") ;;
    esac
done
exec docker run --rm -i --name "$name" --device /dev/dri --oom-score-adj 1000 --stop-timeout 30 --no-healthcheck \
    -v "$root:/work" \
    "${envs[@]}" \
    "${STRATA_SYCL_IMAGE:-strata-sycl-dev}" \
    "cd $repo_in && exec ${STRATA_SYCL_BIN:-build-sycl-aot/strata}$args"
