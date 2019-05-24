#include "src/utils/allocator.h"

#include <cstdlib>

namespace libgav1 {

AllocatorBase::AllocatorBase(size_t element_size)
    : element_size_(element_size), ok_(true) {}

void* AllocatorBase::Allocate(size_t s) {
  if (!ok_) return nullptr;
  void* const ptr = malloc(s);
  ok_ = !(s > 0 && ptr == nullptr);
  return ptr;
}

bool AllocatorBase::Deallocate(void* p, size_t /*nb_elements*/) {
  if (!ok_) return false;
  free(p);
  return true;
}

void AllocatorBase::Init(size_t /*element_size*/) {}

}  // namespace libgav1
