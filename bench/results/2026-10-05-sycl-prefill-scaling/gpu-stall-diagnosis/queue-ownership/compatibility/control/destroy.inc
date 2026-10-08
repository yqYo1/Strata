  void destroy_queue(sycl::queue *&queue) {
    std::lock_guard<mutex_type> lock(m_mutex);
    _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                  [=](const std::shared_ptr<sycl::queue> &q) -> bool {
                                    return q.get() == queue;
                                  }),
                   _queues.end());
    queue = nullptr;
  }
