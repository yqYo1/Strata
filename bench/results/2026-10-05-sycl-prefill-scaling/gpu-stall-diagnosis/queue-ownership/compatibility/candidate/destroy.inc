  void destroy_queue(sycl::queue *&queue) {
    std::lock_guard<mutex_type> lock(m_mutex);
    if (!queue) return;
    // The default queues remain borrowed by device_ext and its callers.
    // Removing either would leave those raw references dangling.
    if (queue == _q_in_order || queue == _q_out_of_order)
      throw std::invalid_argument("destroy_queue: cannot destroy a borrowed default queue");
    auto found = std::find_if(_queues.begin(), _queues.end(),
                             [=](const std::shared_ptr<sycl::queue> &q) {
                               return q.get() == queue;
                             });
    if (found == _queues.end())
      throw std::invalid_argument("destroy_queue: queue is not owned by this device");
    _queues.erase(found);
    queue = nullptr;
  }
