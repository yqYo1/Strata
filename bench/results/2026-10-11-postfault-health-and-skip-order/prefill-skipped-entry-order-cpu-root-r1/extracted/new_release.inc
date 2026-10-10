                            auto release_to = [&](int32_t e_stop) {
                                try {
                            while (k < kend && seq[k].e < e_stop) {
                                    if (gg_nslots > 0) flush();   // the open group's slots get their event first
                                    wait_issued(k);   // finish issuer slot reads before publishing skipped-entry reuse
                                    const int sl = (int) (k % (size_t) m.ring);
                                    dpct::sync_barrier(m.used[sl], m.cs);
                                    m.used_of[sl] = sl;
                                    consumed = ++k;
                                    give_back(consumed);
                                }
                            }
                            catch (sycl::exception const &exc) {
                              std::cerr
                                  << exc.what()
                                  << "Exception caught at file:" << __FILE__
                                  << ", line:" << __LINE__ << std::endl;
                              if (route_census.on) detail::RouteCensus::fatal_receipt("prefill_fatal_no_unwind");
                              std::exit(1);
                            }
                            };
