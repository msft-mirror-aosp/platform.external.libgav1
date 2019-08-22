#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>

#include "src/obu_parser.h"
#include "src/symbol_decoder_context.h"
#include "src/tile.h"
#include "src/utils/bit_mask_set.h"
#include "src/utils/common.h"
#include "src/utils/constants.h"
#include "src/utils/entropy_decoder.h"
#include "src/utils/types.h"

namespace libgav1 {
namespace {

// Add |value| to the |cache| if it doesn't already exist.
void MaybeAddToPaletteCache(uint16_t value, uint16_t* const cache,
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
  if (IsBlockSmallerThan8x8(block.size) || block.size > kBlock64x64 ||
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
          reader_.ReadSymbol<kPaletteSizeSymbolCount>(
              symbol_decoder_context_.palette_y_size_cdf[block_size_context]);
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
          reader_.ReadSymbol<kPaletteSizeSymbolCount>(
              symbol_decoder_context_.palette_uv_size_cdf[block_size_context]);
      ReadPaletteColors(block, kPlaneU);
    }
  }
}

void Tile::PopulatePaletteColorContexts(
    const Block& block, PlaneType plane_type, int i, int start, int end,
    uint8_t color_order[kMaxPaletteSquare][kMaxPaletteSize],
    uint8_t color_context[kMaxPaletteSquare]) {
  const PredictionParameters& prediction_parameters =
      *block.bp->prediction_parameters;
  for (int column = start, counter = 0; column >= end; --column, ++counter) {
    const int row = i - column;
    assert(row > 0 || column > 0);
    const uint8_t top =
        (row > 0)
            ? prediction_parameters.color_index_map[plane_type][row - 1][column]
            : 0;
    const uint8_t left =
        (column > 0)
            ? prediction_parameters.color_index_map[plane_type][row][column - 1]
            : 0;
    uint8_t index_mask;
    static_assert(kMaxPaletteSize <= 8, "");
    int index;
    if (column <= 0) {
      color_context[counter] = 0;
      color_order[counter][0] = top;
      index_mask = 1 << top;
      index = 1;
    } else if (row <= 0) {
      color_context[counter] = 0;
      color_order[counter][0] = left;
      index_mask = 1 << left;
      index = 1;
    } else {
      const uint8_t top_left =
          prediction_parameters
              .color_index_map[plane_type][row - 1][column - 1];
      index_mask = (1 << top) | (1 << left) | (1 << top_left);
      if (top == left && top == top_left) {
        color_context[counter] = 4;
        color_order[counter][0] = top;
        index = 1;
      } else if (top == left) {
        color_context[counter] = 3;
        color_order[counter][0] = top;
        color_order[counter][1] = top_left;
        index = 2;
      } else if (top == top_left) {
        color_context[counter] = 2;
        color_order[counter][0] = top_left;
        color_order[counter][1] = left;
        index = 2;
      } else if (left == top_left) {
        color_context[counter] = 2;
        color_order[counter][0] = top_left;
        color_order[counter][1] = top;
        index = 2;
      } else {
        color_context[counter] = 1;
        color_order[counter][0] = std::min(top, left);
        color_order[counter][1] = std::max(top, left);
        color_order[counter][2] = top_left;
        index = 3;
      }
    }
    // Even though only the first |palette_size| entries of this array are ever
    // used, it is faster to populate all 8 because of the vectorization of the
    // constant sized loop.
    for (uint8_t j = 0; j < kMaxPaletteSize; ++j) {
      if (BitMaskSet::MaskContainsValue(index_mask, j)) continue;
      color_order[counter][index++] = j;
    }
  }
}

bool Tile::ReadPaletteTokens(const Block& block) {
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
    if (!prediction_parameters.color_index_map[plane_type].Reset(
            block_height, block_width, /*zero_initialize=*/false)) {
      return false;
    }
    int first_value = 0;
    reader_.DecodeUniform(palette_size, &first_value);
    prediction_parameters.color_index_map[plane_type][0][0] = first_value;
    for (int i = 1; i < screen_height + screen_width - 1; ++i) {
      const int start = std::min(i, screen_width - 1);
      const int end = std::max(0, i - screen_height + 1);
      uint8_t color_order[kMaxPaletteSquare][kMaxPaletteSize];
      uint8_t color_context[kMaxPaletteSquare];
      PopulatePaletteColorContexts(block, static_cast<PlaneType>(plane_type), i,
                                   start, end, color_order, color_context);
      for (int j = start, counter = 0; j >= end; --j, ++counter) {
        uint16_t* const cdf =
            symbol_decoder_context_
                .palette_color_index_cdf[plane_type]
                                        [palette_size - kMinPaletteSize]
                                        [color_context[counter]];
        const int color_order_index = reader_.ReadSymbol(cdf, palette_size);
        prediction_parameters.color_index_map[plane_type][i - j][j] =
            color_order[counter][color_order_index];
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
  return true;
}

}  // namespace libgav1
