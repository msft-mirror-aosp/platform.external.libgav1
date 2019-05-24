#ifndef LIBGAV1_SRC_UTILS_VECTOR_H_
#define LIBGAV1_SRC_UTILS_VECTOR_H_

#include <algorithm>
#include <initializer_list>
#include <utility>
#include <vector>

#include "src/utils/allocator.h"
#include "src/utils/compiler_attributes.h"

namespace libgav1 {

//------------------------------------------------------------------------------
// Vector class that does *NOT* initialize the content by default, unless an
// explicit value is passed to the constructor.
// Should be reserved to POD preferably.

// resize(), reserve(), and push_back() are overridden to return bool.
//
// New methods: ok() and CopyFrom().
//
// DO NOT USE emplace_back(), insert(), and emplace().
template <typename T, typename super = std::vector<T, AllocatorNoCtor<T>>>
class VectorNoCtor : public super {
 public:
  using super::super;
  bool ok() const { return this->get_allocator().ok(); }
  T* operator*() = delete;
  LIBGAV1_MUST_USE_RESULT inline bool resize(size_t n) {
    return ok() && (super::resize(n), ok());
  }
  LIBGAV1_MUST_USE_RESULT inline bool reserve(size_t n) {
    return ok() && (super::reserve(n), ok());
  }
  // Release the memory.
  inline void reset() {
    VectorNoCtor<T, super> tmp;
    super::swap(tmp);
  }

  // disable resizing ctors
  VectorNoCtor(size_t size) noexcept = delete;
  VectorNoCtor(size_t size, const T&) noexcept = delete;
  VectorNoCtor& operator=(const VectorNoCtor& A) noexcept = delete;
  VectorNoCtor(const VectorNoCtor& other) noexcept = delete;
  template <typename InputIt>
  VectorNoCtor(InputIt first, InputIt last) = delete;
  VectorNoCtor(std::initializer_list<T> init) = delete;

  // benign ctors
  VectorNoCtor() noexcept : super() {}
  VectorNoCtor& operator=(VectorNoCtor&& A) = default;
  VectorNoCtor(VectorNoCtor&& A) noexcept : super(std::move(A)) {}

  void assign(size_t count, const T& value) = delete;
  template <typename InputIt>
  void assign(InputIt first, InputIt last) = delete;
  void assign(std::initializer_list<T> ilist) = delete;

  // To be used instead of copy-ctor:
  bool CopyFrom(const VectorNoCtor& A) {
    if (!resize(A.size())) return false;
    std::copy(A.begin(), A.end(), super::begin());
    return true;
  }
  // Performs a push back *if* the vector was properly allocated.
  // *NO* re-allocation happens.
  LIBGAV1_MUST_USE_RESULT inline bool push_back(const T& v) {
    if (super::size() < super::capacity()) {
      super::push_back(v);
      return true;
    }
    return false;
  }
  LIBGAV1_MUST_USE_RESULT inline bool push_back(T&& v) {
    if (super::size() < super::capacity()) {
      super::push_back(v);
      return true;
    }
    return false;
  }
};

// This generic vector class will call the constructors
template <typename T>
using Vector = VectorNoCtor<T, std::vector<T, Allocator<T>>>;

}  // namespace libgav1

#endif  // LIBGAV1_SRC_UTILS_VECTOR_H_
