class descriptor {
public:
  void set_queue(::dpct::cs::queue_ptr q_ptr) noexcept { _queue_ptr = q_ptr; }
  sycl::queue &get_queue() noexcept { return *_queue_ptr; }
  void set_math_mode(math_mode mm) noexcept { _mm = mm; }
  math_mode get_math_mode() const noexcept { return _mm; }
  static inline void set_saved_queue(::dpct::cs::queue_ptr q_ptr) noexcept {
    _saved_queue_ptr = q_ptr;
  }
  static inline sycl::queue &get_saved_queue() noexcept {
    return *_saved_queue_ptr;
  }

private:
  ::dpct::cs::queue_ptr _queue_ptr = &::dpct::cs::get_default_queue();
  math_mode _mm = math_mode::mm_default;
  static inline ::dpct::cs::queue_ptr _saved_queue_ptr =
      &::dpct::cs::get_default_queue();
};
