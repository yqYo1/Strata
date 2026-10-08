#!/bin/bash
export DEBUGINFOD_URLS=""
exec /usr/bin/gdb -nx -q -batch --return-child-result -iex "set debuginfod enabled off" -ex "set pagination off" -ex "set logging file /home/yayoi/.local/state/strata-sycl/orderly-serve-shutdown-20261006/gdb-shutdown.txt" -ex "set logging overwrite on" -ex "set logging redirect on" -ex "set logging enabled on" -ex run -ex "thread apply all bt 12" -ex "info sharedlibrary" --args /home/yayoi/.local/state/strata-sycl/orderly-serve-shutdown-20261006/strata-orderly-serve "$@"
