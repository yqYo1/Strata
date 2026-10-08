    strata::core::session_graphs_free(gr);
    strata::core::doorbell_free(db);
    sycl::free(d_next, dpct::get_in_order_queue());
    sycl::free(d_logits, dpct::get_in_order_queue());
    sycl::free(d_emb, dpct::get_in_order_queue());
    sycl::free(d_parts, dpct::get_in_order_queue());
    sycl::free(sbuf, dpct::get_in_order_queue());
    sycl::free(arena, dpct::get_in_order_queue());
    return 0;
