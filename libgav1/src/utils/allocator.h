#ifndef LIBGAV1_SRC_UTILS_ALLOCATOR_H_
#define LIBGAV1_SRC_UTILS_ALLOCATOR_H_

#include <cstddef>
#include <utility>

namespace libgav1 {

class AllocatorBase {
 public:
  bool ok() const { return ok_; }

 protected:
  explicit AllocatorBase(size_t element_size);

  void* Allocate(size_t s);
  bool Deallocate(void* p, size_t nb_elements);
  void Init(size_t element_size);

  size_t element_size_;
  bool ok_;
};

// This allocator will NOT call the constructor, since the construct()
// method has been voided.
template <typename T>
class AllocatorNoCtor : protected AllocatorBase {
 public:
  using value_type = T;
  AllocatorNoCtor() : AllocatorBase(sizeof(T)) {}
  template <typename U>
  explicit AllocatorNoCtor(const AllocatorNoCtor<U>& /*other*/)
      : AllocatorBase(sizeof(T)) {}

  T* allocate(size_t nb_elements) {
    return static_cast<T*>(AllocatorBase::Allocate(nb_elements * sizeof(T)));
  }
  void deallocate(T* p, size_t nb_elements) {
    ok_ = ok_ && AllocatorBase::Deallocate(p, nb_elements);
  }
  bool ok() const { return AllocatorBase::ok(); }

  // The allocator disables any construction...
  template <typename U, typename... Args>
  void construct(U*, Args&&...) noexcept {}

  // ...but copy and move constructions, which are called by the vector
  // implementation itself.
  void construct(T* p, const T& v) noexcept {
    static_assert(noexcept(new ((void*)p) T(v)),
                  "needs a noexcept copy constructor");
    if (ok_) new ((void*)p) T(v);
  }
  void construct(T* p, T&& v) noexcept {
    static_assert(noexcept(new ((void*)p) T(std::move(v))),
                  "needs a noexcept move constructor");
    if (ok_) new ((void*)p) T(std::move(v));
  }

  template <typename U>
  void destroy(U* p) noexcept {
    if (ok_) p->~U();
  }
};

// This allocator calls the constructors
template <typename T>
class Allocator : public AllocatorNoCtor<T> {
 public:
  using value_type = T;
  Allocator() : AllocatorNoCtor<T>() {}
  template <typename U>
  explicit Allocator(const Allocator<U>& other) : AllocatorNoCtor<T>(other) {}
  // Enable the constructor.
  template <typename U, typename... Args>
  void construct(U* p, Args&&... args) noexcept {
    static_assert(noexcept(new ((void*)p) U(std::forward<Args>(args)...)),
                  "needs a noexcept constructor");
    if (AllocatorBase::ok_) new ((void*)p) U(std::forward<Args>(args)...);
  }
};

template <typename U, typename V>
bool operator==(const AllocatorNoCtor<U>&, const AllocatorNoCtor<V>&) {
  return true;
}
template <typename U, typename V>
bool operator!=(const AllocatorNoCtor<U>&, const AllocatorNoCtor<V>&) {
  return false;
}

}  // namespace libgav1

#endif  // LIBGAV1_SRC_UTILS_ALLOCATOR_H_
