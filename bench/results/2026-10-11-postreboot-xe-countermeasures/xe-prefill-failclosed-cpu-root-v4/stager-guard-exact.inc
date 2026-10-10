struct StagerDone {
            Stager* st;
            strata::gpu::Stream cs, copy;
            void complete() {
                if (!st) return;
                st->cancel();
                publication::cleanup_require(strata::gpu::stream_sync(cs), "stager compute drain", strata::gpu::last_error);
                publication::cleanup_require(strata::gpu::stream_sync(copy), "stager copy drain", strata::gpu::last_error);
                st->finish();
                st->failure.rethrow();
                st = nullptr;
            }
            ~StagerDone() {
                if (st) {
                    st->cancel();
                    publication::cleanup_require(strata::gpu::stream_sync(cs), "unwind cs drain", strata::gpu::last_error);
                    publication::cleanup_require(strata::gpu::stream_sync(copy), "unwind copy drain", strata::gpu::last_error);
                    st->finish();
                }
            }
        };
