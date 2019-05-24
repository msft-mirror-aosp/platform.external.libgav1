#ifndef LIBGAV1_SRC_UTILS_MEMORY_H_
#define LIBGAV1_SRC_UTILS_MEMORY_H_

#if defined(__ANDROID__) || defined(_MSC_VER)
#include <malloc.h>
#endif

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>

namespace libgav1 {

// AlignedAlloc, AlignedFree
//
// void* AlignedAlloc(size_t alignment, size_t size);
//   Allocate aligned memory.
//   |alignment| must be a power of 2.
//   Unlike posix_memalign(), |alignment| may be smaller than sizeof(void*).
//   Unlike aligned_alloc(), |size| does not need to be a multiple of
//   |alignment|.
//   The returned pointer should be freed by AlignedFree().
//
// void AlignedFree(void* aligned_memory);
//   Free aligned memory.

#if defined(_MSC_VER)  // MSVC

inline void* AlignedAlloc(size_t alignment, size_t size) {
  return _aligned_malloc(size, alignment);
}

inline void AlignedFree(void* aligned_memory) { _aligned_free(aligned_memory); }

#else  // !defined(_MSC_VER)

inline void* AlignedAlloc(size_t alignment, size_t size) {
#if defined(__ANDROID__)
  // Although posix_memalign() was introduced in Android API level 17, it is
  // more convenient to use memalign(). Unlike glibc, Android does not consider
  // memalign() an obsolete function.
  return memalign(alignment, size);
#else   // !defined(__ANDROID__)
  void* ptr = nullptr;
  // posix_memalign requires that the requested alignment be at least
  // sizeof(void*). In this case, fall back on malloc which should return
  // memory aligned to at least the size of a pointer.
  const size_t required_alignment = sizeof(void*);
  if (alignment < required_alignment) return malloc(size);
  const int error = posix_memalign(&ptr, alignment, size);
  if (error != 0) {
    errno = error;
    return nullptr;
  }
  return ptr;
#endif  // defined(__ANDROID__)
}

inline void AlignedFree(void* aligned_memory) { free(aligned_memory); }

#endif  // defined(_MSC_VER)

inline void Memset(uint8_t* const dst, int value, size_t count) {
  memset(dst, value, count);
}

inline void Memset(uint16_t* const dst, int value, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dst[i] = static_cast<uint16_t>(value);
  }
}

struct MallocDeleter {
  void operator()(void* ptr) const { free(ptr); }
};

struct AlignedDeleter {
  void operator()(void* ptr) const { AlignedFree(ptr); }
};

template <typename T>
using AlignedUniquePtr = std::unique_ptr<T, AlignedDeleter>;

// Allocates aligned memory for an array of |count| elements of type T.
template <typename T>
inline AlignedUniquePtr<T> MakeAlignedUniquePtr(size_t alignment,
                                                size_t count) {
  return AlignedUniquePtr<T>(
      static_cast<T*>(AlignedAlloc(alignment, count * sizeof(T))));
}

// A base class with custom new and delete operators. The exception-throwing
// new operators are deleted. The "new (std::nothrow)" form must be used.
//
// The new operators return nullptr if the requested size is greater than
// 0x40000000 bytes (1 GB). TODO(wtc): Make the maximum allocable memory size
// a compile-time configuration macro.
//
// See https://en.cppreference.com/w/cpp/memory/new/operator_new and
// https://en.cppreference.com/w/cpp/memory/new/operator_delete.
//
// NOTE: The allocation and deallocation functions are static member functions
// whether the keyword 'static' is used or not.
struct Allocable {
  // Class-specific allocation functions.
  static void* operator new(size_t size) = delete;
  static void* operator new[](size_t size) = delete;

  // Class-specific non-throwing allocation functions
  static void* operator new(size_t size, const std::nothrow_t& tag) noexcept {
    if (size > 0x40000000) return nullptr;
    return ::operator new(size, tag);
  }
  static void* operator new[](size_t size, const std::nothrow_t& tag) noexcept {
    if (size > 0x40000000) return nullptr;
    return ::operator new[](size, tag);
  }

  // Class-specific deallocation functions.
  static void operator delete(void* ptr) noexcept { ::operator delete(ptr); }
  static void operator delete[](void* ptr) noexcept {
    ::operator delete[](ptr);
  }

  // Only called if new (std::nothrow) is used and the constructor throws an
  // exception.
  static void operator delete(void* ptr, const std::nothrow_t& tag) noexcept {
    ::operator delete(ptr, tag);
  }
  // Only called if new[] (std::nothrow) is used and the constructor throws an
  // exception.
  static void operator delete[](void* ptr, const std::nothrow_t& tag) noexcept {
    ::operator delete[](ptr, tag);
  }
};

}  // namespace libgav1

#endif  // LIBGAV1_SRC_UTILS_MEMORY_H_
