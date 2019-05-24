#include "src/threading_strategy.h"

#include <algorithm>

#include "src/utils/logging.h"

namespace libgav1 {
namespace {

// Maximum number of threads that the library will ever create.
constexpr int kMaxThreads = 32;

}  // namespace

bool ThreadingStrategy::Reset(const ObuFrameHeader& frame_header,
                              int thread_count) {
  if (thread_count <= 1) {
    thread_pool_.reset(nullptr);
    use_tile_threads_ = false;
    max_tile_index_for_row_threads_ = 0;
    return true;
  }

  // We do work in the main thread, so it is sufficient to create
  // |thread_count|-1 threads in the threadpool.
  thread_count = std::min(thread_count - 1, kMaxThreads);

  if (thread_pool_ == nullptr || thread_pool_->num_threads() != thread_count) {
    thread_pool_ = ThreadPool::Create("libgav1", thread_count);
    if (thread_pool_ == nullptr) {
      LIBGAV1_DLOG(ERROR, "Failed to create a thread pool with %d threads.",
                   thread_count);
      use_tile_threads_ = false;
      max_tile_index_for_row_threads_ = 0;
      return false;
    }
  }

  // Prefer tile threads first (but only if there is more than one tile).
  const int tile_count = frame_header.tile_info.tile_count;
  if (tile_count > 1) {
    use_tile_threads_ = true;
    thread_count -= tile_count;
    if (thread_count <= 0) {
      max_tile_index_for_row_threads_ = 0;
      return true;
    }
  } else {
    use_tile_threads_ = false;
  }

  // Assign the remaining threads to each Tile.
  for (int i = 0; i < tile_count; ++i) {
    const int count = thread_count / tile_count +
                      static_cast<int>(i < thread_count % tile_count);
    if (count == 0) {
      // Once we see a 0 value, all subsequent values will be 0 since it is
      // supposed to be assigned in a round-robin fashion.
      break;
    }
    max_tile_index_for_row_threads_ = i + 1;
  }
  return true;
}

}  // namespace libgav1
