#ifndef LIBGAV1_SRC_UTILS_ENTROPY_DECODER_H_
#define LIBGAV1_SRC_UTILS_ENTROPY_DECODER_H_

#include <cstddef>
#include <cstdint>

#if defined(LIBGAV1_USE_LIBAOM_BIT_READER)
#include "third_party/libaom/git_root/aom_dsp/daalaboolreader.h"
#endif

#include "src/utils/bit_reader.h"

namespace libgav1 {

#if defined(LIBGAV1_USE_LIBAOM_BIT_READER)
class DaalaBitReaderAom : public BitReader {
 public:
  DaalaBitReaderAom(const uint8_t* data, size_t size, bool allow_update_cdf);
  ~DaalaBitReaderAom() override = default;

  int ReadBit() override;
  int64_t ReadLiteral(int num_bits) override;
  int ReadSymbol(uint16_t* cdf, int symbol_count);
  bool ReadSymbol(uint16_t* cdf);
  bool ReadSymbolWithoutCdfUpdate(uint16_t* cdf);

 private:
  bool allow_update_cdf_;
  daala_reader reader_;
};
#endif  // defined(LIBGAV1_USE_LIBAOM_BIT_READER)

class DaalaBitReaderNative : public BitReader {
 public:
  DaalaBitReaderNative(const uint8_t* data, size_t size, bool allow_update_cdf);
  ~DaalaBitReaderNative() override = default;

  // Move only.
  DaalaBitReaderNative(DaalaBitReaderNative&& rhs) noexcept;
  DaalaBitReaderNative& operator=(DaalaBitReaderNative&& rhs) noexcept;

  int ReadBit() override;
  int64_t ReadLiteral(int num_bits) override;
  int ReadSymbol(uint16_t* cdf, int symbol_count);
  bool ReadSymbol(uint16_t* cdf);
  bool ReadSymbolWithoutCdfUpdate(uint16_t* cdf);

 private:
  int ReadSymbolImpl(const uint16_t* cdf, int symbol_count);
  void PopulateBits();
  // Normalizes the range so that 32768 <= |values_in_range_| < 65536. Also
  // calls PopulateBits() if necessary.
  void NormalizeRange();

  const uint8_t* data_;
  const size_t size_;
  size_t data_index_;
  const bool allow_update_cdf_;
  // Number of bits of data in the current value.
  int bits_;
  // Number of values in the current range.
  uint16_t values_in_range_;
  // The difference between the high end of the current range and the coded
  // value minus 1. The 16 least significant bits of this variable is used to
  // decode the next symbol. It is filled in whenever |bits_| is less than 0.
  uint32_t window_diff_;
};

#if defined(LIBGAV1_USE_LIBAOM_BIT_READER)
using DaalaBitReader = DaalaBitReaderAom;
#else
using DaalaBitReader = DaalaBitReaderNative;
#endif

}  // namespace libgav1

#endif  // LIBGAV1_SRC_UTILS_ENTROPY_DECODER_H_
