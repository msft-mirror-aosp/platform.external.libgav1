#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>

#include "src/obu_parser.h"
#include "src/symbol_decoder_context.h"
#include "src/tile.h"
#include "src/utils/common.h"
#include "src/utils/constants.h"
#include "src/utils/entropy_decoder.h"
#include "src/utils/types.h"

namespace libgav1 {
namespace {

const int kNumPaletteNeighbors = 3;
const uint8_t kPaletteColorHashMultiplier[kNumPaletteNeighbors] = {1, 2, 2};
const int kPaletteColorIndexContext[kPaletteColorIndexSymbolCount + 1] = {
    -1, -1, 0, -1, -1, 4, 3, 2, 1};

// Add |value| to the |cache| if it doesn't already exist.
inline void MaybeAddToPaletteCache(uint16_t value, uint16_t* const cache,
                                   int* const n) {
  assert(cache != nullptr);
  assert(n != nullptr);
  if (*n > 0 && value == cache[*n - 1]) return;
  cache[(*n)++] = value;
}

// Palette colors are generated using two ascending arrays. So sorting them is
// simply a matter of finding the pivot point and merging two sorted arrays.
void SortPaletteColors(uint16_t* const color, const int size, const int pivot) {
  if (pivot == 0 || pivot == size || color[pivot - 1] < color[pivot]) {
    // The array is already sorted.
    return;
  }
  uint16_t temp_color[kMaxPaletteSize];
  memcpy(temp_color, color, size * sizeof(color[0]));
  int i = 0;
  int j = pivot;
  int k = 0;
  while (i < pivot && j < size) {
    if (temp_color[i] < temp_color[j]) {
      color[k++] = temp_color[i++];
    } else {
      color[k++] = temp_color[j++];
    }
  }
  while (i < pivot) {
    color[k++] = temp_color[i++];
  }
  while (j < size) {
    color[k++] = temp_color[j++];
  }
}

}  // namespace

int Tile::GetPaletteCache(const Block& block, PlaneType plane_type,
                          uint16_t* const cache) {
  const int top_n =
      (block.top_available && Mod64(MultiplyBy4(block.row4x4)) != 0)
          ? block.bp_top->palette_mode_info.size[plane_type]
          : 0;
  const int left_n = block.left_available
                         ? block.bp_left->palette_mode_info.size[plane_type]
                         : 0;
  int top_index = 0;
  int left_index = 0;
  int n = 0;
  while (top_index < top_n && left_index < left_n) {
    const int top_color =
        block.bp_top->palette_mode_info.color[plane_type][top_index];
    const int left_color =
        block.bp_left->palette_mode_info.color[plane_type][left_index];
    if (left_color < top_color) {
      MaybeAddToPaletteCache(left_color, cache, &n);
      ++left_index;
    } else {
      MaybeAddToPaletteCache(top_color, cache, &n);
      ++top_index;
      if (top_color == left_color) ++left_index;
    }
  }
  while (top_index < top_n) {
    MaybeAddToPaletteCache(
        block.bp_top->palette_mode_info.color[plane_type][top_index], cache,
        &n);
    ++top_index;
  }
  while (left_index < left_n) {
    MaybeAddToPaletteCache(
        block.bp_left->palette_mode_info.color[plane_type][left_index], cache,
        &n);
    ++left_index;
  }
  return n;
}

void Tile::ReadPaletteColors(const Block& block, Plane plane) {
  const PlaneType plane_type = GetPlaneType(plane);
  uint16_t cache[2 * kMaxPaletteSize];
  const int n = GetPaletteCache(block, plane_type, cache);
  BlockParameters& bp = *block.bp;
  const uint8_t palette_size = bp.palette_mode_info.size[plane_type];
  uint16_t* const palette_color = bp.palette_mode_info.color[plane];
  const int8_t bitdepth = sequence_header_.color_config.bitdepth;
  int index = 0;
  for (int i = 0; i < n && index < palette_size; ++i) {
    if (reader_.ReadBit() != 0) {  // use_palette_color_cache.
      palette_color[index++] = cache[i];
    }
  }
  const int merge_pivot = index;
  if (index < palette_size) {
    palette_color[index++] =
        static_cast<uint16_t>(reader_.ReadLiteral(bitdepth));
  }
  if (index < palette_size) {
    int bits = bitdepth - 3 + static_cast<int>(reader_.ReadLiteral(2));
    for (; index < palette_size; ++index) {
      const int delta = static_cast<int>(reader_.ReadLiteral(bits)) +
                        (plane_type == kPlaneTypeY ? 1 : 0);
      palette_color[index] =
          Clip3(palette_color[index - 1] + delta, 0, (1 << bitdepth) - 1);
      const int range = (1 << bitdepth) - palette_color[index] -
                        (plane_type == kPlaneTypeY ? 1 : 0);
      bits = std::min(bits, CeilLog2(range));
    }
  }
  SortPaletteColors(palette_color, palette_size, merge_pivot);
  if (plane_type == kPlaneTypeUV) {
    uint16_t* const palette_color_v = bp.palette_mode_info.color[kPlaneV];
    if (reader_.ReadBit() != 0) {  // delta_encode_palette_colors_v.
      const int max_value = 1 << bitdepth;
      const int bits = bitdepth - 4 + static_cast<int>(reader_.ReadLiteral(2));
      palette_color_v[0] = reader_.ReadLiteral(bitdepth);
      for (int i = 1; i < palette_size; ++i) {
        int delta = static_cast<int>(reader_.ReadLiteral(bits));
        if (delta != 0 && reader_.ReadBit() != 0) delta = -delta;
        int value = palette_color_v[i - 1] + delta;
        if (value < 0) value += max_value;
        if (value >= max_value) value -= max_value;
        palette_color_v[i] = Clip3(value, 0, (1 << bitdepth) - 1);
      }
    } else {
      for (int i = 0; i < palette_size; ++i) {
        palette_color_v[i] =
            static_cast<uint16_t>(reader_.ReadLiteral(bitdepth));
      }
    }
  }
}

int Tile::GetHasPaletteYContext(const Block& block) const {
  int context = 0;
  if (block.top_available &&
      block.bp_top->palette_mode_info.size[kPlaneTypeY] > 0) {
    ++context;
  }
  if (block.left_available &&
      block.bp_left->palette_mode_info.size[kPlaneTypeY] > 0) {
    ++context;
  }
  return context;
}

void Tile::ReadPaletteModeInfo(const Block& block) {
  BlockParameters& bp = *block.bp;
  if (IsBlockSmallerThan8x8(block.size) || kBlockWidthPixels[block.size] > 64 ||
      kBlockHeightPixels[block.size] > 64 ||
      !frame_header_.allow_screen_content_tools) {
    bp.palette_mode_info.size[kPlaneTypeY] = 0;
    bp.palette_mode_info.size[kPlaneTypeUV] = 0;
    return;
  }
  const int block_size_context =
      k4x4WidthLog2[block.size] + k4x4HeightLog2[block.size] - 2;
  if (bp.y_mode == kPredictionModeDc) {
    const int context = GetHasPaletteYContext(block);
    const bool has_palette_y = reader_.ReadSymbol(
        symbol_decoder_context_.has_palette_y_cdf[block_size_context][context]);
    if (has_palette_y) {
      bp.palette_mode_info.size[kPlaneTypeY] =
          kMinPaletteSize +
          reader_.ReadSymbol(
              symbol_decoder_context_.palette_y_size_cdf[block_size_context],
              kPaletteSizeSymbolCount);
      ReadPaletteColors(block, kPlaneY);
    }
  }
  if (PlaneCount() > 1 && bp.uv_mode == kPredictionModeDc &&
      block.HasChroma()) {
    const int context =
        static_cast<int>(bp.palette_mode_info.size[kPlaneTypeY] > 0);
    const bool has_palette_uv =
        reader_.ReadSymbol(symbol_decoder_context_.has_palette_uv_cdf[context]);
    if (has_palette_uv) {
      bp.palette_mode_info.size[kPlaneTypeUV] =
          kMinPaletteSize +
          reader_.ReadSymbol(
              symbol_decoder_context_.palette_uv_size_cdf[block_size_context],
              kPaletteSizeSymbolCount);
      ReadPaletteColors(block, kPlaneU);
    }
  }
}

int Tile::GetPaletteColorContext(const Block& block, PlaneType plane_type,
                                 int row, int column, int palette_size,
                                 uint8_t color_order[kMaxPaletteSize]) {
  for (int i = 0; i < kMaxPaletteSize; ++i) {
    color_order[i] = i;
  }
  int scores[kMaxPaletteSize] = {};
  const PredictionParameters& prediction_parameters =
      *block.bp->prediction_parameters;
  if (row > 0 && column > 0) {
    ++scores[prediction_parameters
                 .color_index_map[plane_type][row - 1][column - 1]];
  }
  if (row > 0) {
    scores[prediction_parameters
               .color_index_map[plane_type][row - 1][column]] += 2;
  }
  if (column > 0) {
    scores[prediction_parameters
               .color_index_map[plane_type][row][column - 1]] += 2;
  }
  // Move the top 3 scores (largest first) and the corresponding color_order
  // entry to the front of the array.
  for (int i = 0; i < kNumPaletteNeighbors; ++i) {
    const auto max_element =
        std::max_element(scores + i, scores + palette_size);
    const auto max_score = *max_element;
    const auto max_index = static_cast<int>(std::distance(scores, max_element));
    if (max_index != i) {
      const uint8_t max_color_order = color_order[max_index];
      for (int j = max_index; j > i; --j) {
        scores[j] = scores[j - 1];
        color_order[j] = color_order[j - 1];
      }
      scores[i] = max_score;
      color_order[i] = max_color_order;
    }
  }
  int context = 0;
  for (int i = 0; i < kNumPaletteNeighbors; ++i) {
    context += scores[i] * kPaletteColorHashMultiplier[i];
  }
  return kPaletteColorIndexContext[context];
}

void Tile::ReadPaletteTokens(const Block& block) {
  const PaletteModeInfo& palette_mode_info = block.bp->palette_mode_info;
  PredictionParameters& prediction_parameters =
      *block.bp->prediction_parameters;
  for (int plane_type = kPlaneTypeY;
       plane_type < (block.HasChroma() ? kNumPlaneTypes : kPlaneTypeUV);
       ++plane_type) {
    const int palette_size = palette_mode_info.size[plane_type];
    if (palette_size == 0) continue;
    int block_height = kBlockHeightPixels[block.size];
    int block_width = kBlockWidthPixels[block.size];
    int screen_height = std::min(
        block_height, MultiplyBy4(frame_header_.rows4x4 - block.row4x4));
    int screen_width = std::min(
        block_width, MultiplyBy4(frame_header_.columns4x4 - block.column4x4));
    if (plane_type == kPlaneTypeUV) {
      block_height >>= sequence_header_.color_config.subsampling_y;
      block_width >>= sequence_header_.color_config.subsampling_x;
      screen_height >>= sequence_header_.color_config.subsampling_y;
      screen_width >>= sequence_header_.color_config.subsampling_x;
      if (block_height < 4) {
        block_height += 2;
        screen_height += 2;
      }
      if (block_width < 4) {
        block_width += 2;
        screen_width += 2;
      }
    }
    uint8_t color_order[kMaxPaletteSize];
    int first_value = 0;
    reader_.DecodeUniform(palette_size, &first_value);
    prediction_parameters.color_index_map[plane_type][0][0] = first_value;
    for (int i = 1; i < screen_height + screen_width - 1; ++i) {
      for (int j = std::min(i, screen_width - 1);
           j >= std::max(0, i - screen_height + 1); --j) {
        const int context =
            GetPaletteColorContext(block, static_cast<PlaneType>(plane_type),
                                   i - j, j, palette_size, color_order);
        assert(context >= 0);
        uint16_t* const cdf =
            symbol_decoder_context_
                .palette_color_index_cdf[plane_type][palette_size -
                                                     kMinPaletteSize][context];
        const int color_order_index = reader_.ReadSymbol(cdf, palette_size);
        prediction_parameters.color_index_map[plane_type][i - j][j] =
            color_order[color_order_index];
      }
    }
    if (screen_width < block_width) {
      for (int i = 0; i < screen_height; ++i) {
        memset(
            &prediction_parameters.color_index_map[plane_type][i][screen_width],
            prediction_parameters
                .color_index_map[plane_type][i][screen_width - 1],
            block_width - screen_width);
      }
    }
    for (int i = screen_height; i < block_height; ++i) {
      memcpy(
          prediction_parameters.color_index_map[plane_type][i],
          prediction_parameters.color_index_map[plane_type][screen_height - 1],
          block_width);
    }
  }
}

}  // namespace libgav1
