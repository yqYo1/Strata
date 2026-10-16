set -euo pipefail
for variant in j both acc2; do
 /tmp/strata-sycl-goal-iq-looped-${variant}-bench "$HOME/.local/share/strata-sycl/models/qwen3.8-flash-next-iq3_s/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" > "/tmp/strata-sycl-goal-iq-looped-${variant}.csv"
done
