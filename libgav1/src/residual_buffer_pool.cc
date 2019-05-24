#include "src/residual_buffer_pool.h"

#include <algorithm>
#include <new>
#include <utility>

namespace libgav1 {
namespace {

// The maximum queue size is derived using the following formula:
//   ((sb_size * sb_size) / 16) + (2 * (((sb_size / x) * (sb_size / y)) / 16)).
// Where:
//   sb_size is the superblock size (64 or 128).
//   16 is 4*4 which is kMinTransformWidth * kMinTransformHeight.
//   x is subsampling_x + 1.
//   y is subsampling_y + 1.
// The first component is for the Y plane and the second component is for the U
// and V planes.
// For example, for 128x128 superblocks with 422 subsampling the size is:
//   ((128 * 128) / 16) + (2 * (((128 / 2) * (128 / 1)) / 16)) = 2048.
//
// First dimension: use_128x128_superblock.
// Second dimension: subsampling_x.
// Third dimension: subsampling_y.
constexpr int kMaxQueueSize[2][2][2] = {
    // 64x64 superblocks.
    {
        {768, 512},
        {512, 384},
    },
    // 128x128 superblocks.
    {
        {3072, 2048},
        {2048, 1536},
    },
};

}  // namespace

ResidualBufferPool::ResidualBufferPool(bool use_128x128_superblock,
                                       int subsampling_x, int subsampling_y,
                                       size_t residual_size)
    : buffer_size_(GetResidualBufferSize(
          use_128x128_superblock ? 128 : 64, use_128x128_superblock ? 128 : 64,
          subsampling_x, subsampling_y, residual_size)),
      queue_size_(kMaxQueueSize[static_cast<int>(use_128x128_superblock)]
                               [subsampling_x][subsampling_y]) {}

void ResidualBufferPool::Reset(bool use_128x128_superblock, int subsampling_x,
                               int subsampling_y, size_t residual_size) {
  const size_t buffer_size = GetResidualBufferSize(
      use_128x128_superblock ? 128 : 64, use_128x128_superblock ? 128 : 64,
      subsampling_x, subsampling_y, residual_size);
  const int queue_size = kMaxQueueSize[static_cast<int>(use_128x128_superblock)]
                                      [subsampling_x][subsampling_y];
  if (buffer_size == buffer_size_ && queue_size == queue_size_) {
    // The existing buffers (if any) are still valid, so don't do anything.
    return;
  }
  buffer_size_ = buffer_size;
  queue_size_ = queue_size;
  // The existing buffers (if any) are no longer valid since the buffer size or
  // the queue size has changed. Clear the stack.
  std::lock_guard<std::mutex> lock(mutex_);
  while (!buffers_.empty()) {
    buffers_.pop();
  }
}

std::unique_ptr<ResidualBuffer> ResidualBufferPool::Get() {
  std::unique_ptr<ResidualBuffer> buffer = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!buffers_.empty()) {
      buffer = std::move(buffers_.top());
      buffers_.pop();
    }
  }
  if (buffer == nullptr) {
    buffer.reset(new (std::nothrow) ResidualBuffer(buffer_size_, queue_size_));
  }
  return buffer;
}

void ResidualBufferPool::Release(std::unique_ptr<ResidualBuffer> buffer) {
  buffer->transform_parameters.Reset();
  std::lock_guard<std::mutex> lock(mutex_);
  buffers_.push(std::move(buffer));
}

size_t ResidualBufferPool::Size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return buffers_.size();
}

}  // namespace libgav1
