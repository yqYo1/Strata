# Full-context controller smoke check

CPU protocol stub only. No model arithmetic, GPU allocation, real prefill,
decoding or throughput is tested by these files.

The controller passes valid request/reply sequences at context limits 64 and
262,144. Each sequence includes a four-output full context, a forced two-token
verify tail, full-input refusal, one-token overflow refusal and a valid request
after refusal. The test checks the actual request lengths sent to the stub.

All nine injected faults are rejected: a verify overrun, a missing last KV
cell, an incorrect tail that still reaches the last cell, early EOS, wrong
prompt counts, a wrong finish reason, NaN logprobs, a missing newly written
head and execution of a refused request. The stale-head case follows a valid
request, so it also checks that the old head cannot satisfy the next request.

`summary.json` records expected/observed completion, exit codes, terminal
errors and controller/stub SHA-256. Each case has a controller report and a
process log. Temporary stub executables and zero-filled head files are not
versioned. These successes establish the controller's checks, not the engine's
full-context correctness. Actual 256K engine tests remain pending GPU recovery.

Reproduce from the parent directory:

```sh
python3 check_full_context_controller.py \
  --out ~/.local/state/strata-sycl/full-context-controller-smoke
```

The embedding service must remain inactive, as required by the controller.
