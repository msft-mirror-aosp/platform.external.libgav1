#include "src/utils/entropy_decoder.h"

#include <cassert>

#include "src/utils/common.h"
#include "src/utils/constants.h"

namespace {

constexpr uint32_t kWindowSize = static_cast<uint32_t>(sizeof(uint32_t)) * 8;
constexpr int kCdfPrecision = 6;
constexpr int kMinimumProbabilityPerSymbol = 4;
constexpr uint32_t kReadBitMask = ~255;
// This constant is used to set the value of |bits_| so that bits can be read
// after end of stream without trying to refill the buffer for a reasonably long
// time.
constexpr int kLargeBitCount = 0x4000;

void UpdateCdf(uint16_t* cdf, int value, int symbol_count) {
  const uint16_t count = cdf[symbol_count];
  // rate is computed in the spec as:
  //  3 + ( cdf[N] > 15 ) + ( cdf[N] > 31 ) + Min(FloorLog2(N), 2)
  // In this case cdf[N] is |count|.
  // Min(FloorLog2(N), 2) is 1 for symbol_count == {2, 3} and 2 for all
  // symbol_count > 3. So the equation becomes:
  //  4 + (count > 15) + (count > 31) + (symbol_count > 3).
  // Note that the largest value for count is 32 (it is not incremented beyond
  // 32). So using that information:
  //  count >> 4 is 0 for count from 0 to 15.
  //  count >> 4 is 1 for count from 16 to 31.
  //  count >> 4 is 2 for count == 31.
  // Now, the equation becomes:
  //  4 + (count >> 4) + (symbol_count > 3).
  // Since (count >> 4) can only be 0 or 1 or 2, the addition can be replaced
  // with bitwise or. So the final equation is:
  // (4 | (count >> 4)) + (symbol_count > 3).
  const int rate = (4 | (count >> 4)) + static_cast<int>(symbol_count > 3);
  for (int i = 0; i < symbol_count - 1; ++i) {
    if (i < value) {
      cdf[i] += (libgav1::kCdfMaxProbability - cdf[i]) >> rate;
    } else {
      cdf[i] -= cdf[i] >> rate;
    }
  }
  cdf[symbol_count] += static_cast<uint16_t>(count < 32);
}

}  // namespace

namespace libgav1 {

#if defined(LIBGAV1_USE_LIBAOM_BIT_READER)
const int kProbabilityHalf = 128;

DaalaBitReaderAom::DaalaBitReaderAom(const uint8_t* data, size_t size,
                                     bool allow_update_cdf)
    : allow_update_cdf_(allow_update_cdf) {
  aom_daala_reader_init(&reader_, data, static_cast<int>(size));
}

int DaalaBitReaderAom::ReadBit() {
  return aom_daala_read(&reader_, kProbabilityHalf);
}

int64_t DaalaBitReaderAom::ReadLiteral(int num_bits) {
  if (num_bits > 32) return -1;
  uint32_t literal = 0;
  for (int bit = num_bits - 1; bit >= 0; bit--) {
    literal |= static_cast<uint32_t>(ReadBit()) << bit;
  }
  return literal;
}

int DaalaBitReaderAom::ReadSymbol(uint16_t* cdf, int symbol_count) {
  const int symbol = daala_read_symbol(&reader_, cdf, symbol_count);
  if (allow_update_cdf_) {
    UpdateCdf(cdf, symbol, symbol_count);
  }
  return symbol;
}

bool DaalaBitReaderAom::ReadSymbol(uint16_t* cdf) {
  const int symbol = daala_read_symbol(&reader_, cdf, kBooleanSymbolCount);
  if (allow_update_cdf_) {
    UpdateCdf(cdf, symbol, kBooleanSymbolCount);
  }
  return symbol != 0;
}

bool DaalaBitReaderAom::ReadSymbolWithoutCdfUpdate(uint16_t* cdf) {
  return daala_read_symbol(&reader_, cdf, kBooleanSymbolCount) != 0;
}
#endif  // defined(LIBGAV1_USE_LIBAOM_BIT_READER)

DaalaBitReaderNative::DaalaBitReaderNative(const uint8_t* data, size_t size,
                                           bool allow_update_cdf)
    : data_(data),
      size_(size),
      data_index_(0),
      allow_update_cdf_(allow_update_cdf) {
  window_diff_ = (uint32_t{1} << (kWindowSize - 1)) - 1;
  values_in_range_ = kCdfMaxProbability;
  bits_ = -15;
  PopulateBits();
}

// This is similar to the ReadSymbol() implementation but it is optimized based
// on the following facts:
//   * The probability is fixed at half. So some multiplications can be replaced
//     with bit operations.
//   * Symbol count is fixed at 2.
int DaalaBitReaderNative::ReadBit() {
  const uint32_t curr =
      ((values_in_range_ & kReadBitMask) >> 1) + kMinimumProbabilityPerSymbol;
  const uint32_t zero_threshold = curr << (kWindowSize - 16);
  int bit = 1;
  if (window_diff_ >= zero_threshold) {
    values_in_range_ -= curr;
    window_diff_ -= zero_threshold;
    bit = 0;
  } else {
    values_in_range_ = curr;
  }
  NormalizeRange();
  return bit;
}

int64_t DaalaBitReaderNative::ReadLiteral(int num_bits) {
  if (num_bits > 32) return -1;
  uint32_t literal = 0;
  for (int bit = num_bits - 1; bit >= 0; --bit) {
    literal |= static_cast<uint32_t>(ReadBit()) << bit;
  }
  return literal;
}

int DaalaBitReaderNative::ReadSymbol(uint16_t* const cdf, int symbol_count) {
  const int symbol = ReadSymbolImpl(cdf, symbol_count);
  if (allow_update_cdf_) {
    // TODO(vigneshv): This call can be replaced with the function contents
    // inline once the DaalaBitReaderAom is removed.
    UpdateCdf(cdf, symbol, symbol_count);
  }
  return symbol;
}

bool DaalaBitReaderNative::ReadSymbol(uint16_t* cdf) {
  const bool symbol = ReadSymbolImpl(cdf, kBooleanSymbolCount) != 0;
  if (allow_update_cdf_) {
    const uint16_t count = cdf[2];
    // rate is computed in the spec as:
    //  3 + ( cdf[N] > 15 ) + ( cdf[N] > 31 ) + Min(FloorLog2(N), 2)
    // In this case N is 2 and cdf[N] is |count|. So the equation becomes:
    //  4 + (count > 15) + (count > 31)
    // Note that the largest value for count is 32 (it is not incremented beyond
    // 32). So using that information:
    //  count >> 4 is 0 for count from 0 to 15.
    //  count >> 4 is 1 for count from 16 to 31.
    //  count >> 4 is 2 for count == 31.
    // Now, the equation becomes:
    //  4 + (count >> 4).
    // Since (count >> 4) can only be 0 or 1 or 2, the addition can be replaced
    // with bitwise or. So the final equation is:
    //  4 | (count >> 4).
    const uint8_t rate = 4 | (count >> 4);
    if (symbol) {
      cdf[0] += (kCdfMaxProbability - cdf[0]) >> rate;
    } else {
      cdf[0] -= cdf[0] >> rate;
    }
    cdf[2] += static_cast<uint16_t>(count < 32);
  }
  return symbol;
}

bool DaalaBitReaderNative::ReadSymbolWithoutCdfUpdate(uint16_t* cdf) {
  return ReadSymbolImpl(cdf, kBooleanSymbolCount) != 0;
}

int DaalaBitReaderNative::ReadSymbolImpl(const uint16_t* const cdf,
                                         int symbol_count) {
  assert(cdf[symbol_count - 1] == 0);
  --symbol_count;
  uint32_t curr = values_in_range_;
  int symbol = -1;
  uint32_t prev;
  uint32_t symbol_value = window_diff_ >> (kWindowSize - 16);
  do {
    prev = curr;
    curr = values_in_range_ >> 8;
    curr *= cdf[++symbol] >> kCdfPrecision;
    curr >>= 1;
    curr += kMinimumProbabilityPerSymbol * (symbol_count - symbol);
  } while (symbol_value < curr);
  values_in_range_ = prev - curr;
  window_diff_ -= curr << (kWindowSize - 16);
  NormalizeRange();
  return symbol;
}

void DaalaBitReaderNative::PopulateBits() {
  int shift = kWindowSize - 9 - (bits_ + 15);
  for (; shift >= 0 && data_index_ < size_; shift -= 8) {
    window_diff_ ^= static_cast<uint32_t>(data_[data_index_++]) << shift;
    bits_ += 8;
  }
  if (data_index_ >= size_) {
    bits_ = kLargeBitCount;
  }
}

void DaalaBitReaderNative::NormalizeRange() {
  const int bits_used = 15 - FloorLog2(values_in_range_);
  bits_ -= bits_used;
  window_diff_ = ((window_diff_ + 1) << bits_used) - 1;
  values_in_range_ <<= bits_used;
  if (bits_ < 0) PopulateBits();
}

}  // namespace libgav1
