#include "src/decoder.h"

#include <new>

#include "src/decoder_impl.h"
#include "src/utils/logging.h"

namespace libgav1 {

Decoder::Decoder() = default;

Decoder::~Decoder() = default;

StatusCode Decoder::Init(const DecoderSettings* const settings) {
  if (initialized_) return kLibgav1StatusAlready;
  if (settings != nullptr) {
    if (settings->threads < 0) return kLibgav1StatusInvalidArgument;
    settings_ = *settings;
  }
  impl_.reset(new (std::nothrow) DecoderImpl(&settings_));
  if (impl_ == nullptr) {
    LIBGAV1_DLOG(ERROR, "Failed to allocate DecoderImpl.");
    return kLibgav1StatusOutOfMemory;
  }
  initialized_ = true;
  return kLibgav1StatusOk;
}

StatusCode Decoder::EnqueueFrame(const uint8_t* data, const size_t size,
                                 int64_t user_private_data) {
  if (!initialized_) return kLibgav1StatusNotInitialized;
  return impl_->EnqueueFrame(data, size, user_private_data);
}

StatusCode Decoder::DequeueFrame(const DecoderBuffer** out_ptr) {
  if (!initialized_) return kLibgav1StatusNotInitialized;
  return impl_->DequeueFrame(out_ptr);
}

int Decoder::GetMaxAllowedFrames() const {
  return settings_.frame_parallel ? settings_.threads : 1;
}

}  // namespace libgav1
