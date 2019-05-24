#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "src/dsp/constants.h"
#include "src/obu_parser.h"
#include "src/symbol_decoder_context.h"
#include "src/tile.h"
#include "src/utils/array_2d.h"
#include "src/utils/block_parameters_holder.h"
#include "src/utils/common.h"
#include "src/utils/constants.h"
#include "src/utils/entropy_decoder.h"
#include "src/utils/segmentation.h"
#include "src/utils/types.h"

namespace libgav1 {
namespace {

const uint8_t kMaxVariableTransformTreeDepth = 2;

inline TransformSize GetSquareTransformSize(uint8_t pixels) {
  switch (pixels) {
    case 128:
    case 64:
      return kTransformSize64x64;
    case 32:
      return kTransformSize32x32;
    case 16:
      return kTransformSize16x16;
    case 8:
      return kTransformSize8x8;
    default:
      return kTransformSize4x4;
  }
}

}  // namespace

int Tile::GetTopTransformWidth(const Block& block, int row4x4, int column4x4,
                               bool ignore_skip) {
  if (row4x4 == block.row4x4) {
    if (!block.top_available) return 64;
    const BlockParameters& bp_top =
        *block_parameters_holder_.Find(row4x4 - 1, column4x4);
    if ((ignore_skip || bp_top.skip) && bp_top.is_inter) {
      return kBlockWidthPixels[bp_top.size];
    }
  }
  return kTransformWidth[inter_transform_sizes_[row4x4 - 1][column4x4]];
}

int Tile::GetLeftTransformHeight(const Block& block, int row4x4, int column4x4,
                                 bool ignore_skip) {
  if (column4x4 == block.column4x4) {
    if (!block.left_available) return 64;
    const BlockParameters& bp_left =
        *block_parameters_holder_.Find(row4x4, column4x4 - 1);
    if ((ignore_skip || bp_left.skip) && bp_left.is_inter) {
      return kBlockHeightPixels[bp_left.size];
    }
  }
  return kTransformHeight[inter_transform_sizes_[row4x4][column4x4 - 1]];
}

TransformSize Tile::ReadFixedTransformSize(const Block& block) {
  BlockParameters& bp = *block.bp;
  if (frame_header_.segmentation.lossless[bp.segment_id]) {
    return kTransformSize4x4;
  }
  const TransformSize max_rect_tx_size = kMaxTransformSizeRectangle[block.size];
  const bool allow_select = !bp.skip || !bp.is_inter;
  if (block.size <= kBlock4x4 || !allow_select ||
      frame_header_.tx_mode != kTxModeSelect) {
    return max_rect_tx_size;
  }
  const int max_tx_width = kTransformWidth[max_rect_tx_size];
  const int max_tx_height = kTransformHeight[max_rect_tx_size];
  const int top_width =
      block.top_available
          ? GetTopTransformWidth(block, block.row4x4, block.column4x4, true)
          : 0;
  const int left_height =
      block.left_available
          ? GetLeftTransformHeight(block, block.row4x4, block.column4x4, true)
          : 0;
  const auto context = static_cast<int>(top_width >= max_tx_width) +
                       static_cast<int>(left_height >= max_tx_height);
  const int max_tx_depth = kMaxTransformDepth[block.size];
  const int cdf_index = (max_tx_depth > 0) ? max_tx_depth - 1 : 0;
  const int symbol_count = (cdf_index == 0) ? 2 : 3;
  const int tx_depth = reader_.ReadSymbol(
      symbol_decoder_context_.tx_depth_cdf[cdf_index][context], symbol_count);
  TransformSize tx_size = max_rect_tx_size;
  for (int i = 0; i < tx_depth; ++i) {
    tx_size = kSplitTransformSize[tx_size];
  }
  return tx_size;
}

void Tile::ReadVariableTransformTree(const Block& block, int row4x4,
                                     int column4x4, TransformSize tx_size) {
  const uint8_t pixels =
      std::max(kBlockWidthPixels[block.size], kBlockHeightPixels[block.size]);
  const TransformSize max_tx_size = GetSquareTransformSize(pixels);
  const int context_delta = (kNumSquareTransformSizes - 1 -
                             TransformSizeToSquareTransformIndex(max_tx_size)) *
                            6;

  std::vector<TransformTreeNode> stack;
  // Branching factor is 4 and maximum depth is 2. So the maximum stack size
  // necessary is 8.
  stack.reserve(8);
  stack.emplace_back(row4x4, column4x4, tx_size, 0);

  while (!stack.empty()) {
    TransformTreeNode node = stack.back();
    stack.pop_back();
    const int tx_width4x4 = DivideBy4(kTransformWidth[node.tx_size]);
    const int tx_height4x4 = DivideBy4(kTransformHeight[node.tx_size]);
    if (node.tx_size != kTransformSize4x4 &&
        node.depth != kMaxVariableTransformTreeDepth) {
      const auto top = static_cast<int>(
          GetTopTransformWidth(block, node.row4x4, node.column4x4, false) <
          kTransformWidth[node.tx_size]);
      const auto left = static_cast<int>(
          GetLeftTransformHeight(block, node.row4x4, node.column4x4, false) <
          kTransformHeight[node.tx_size]);
      const int context =
          static_cast<int>(max_tx_size > kTransformSize8x8 &&
                           kTransformSizeSquareMax[node.tx_size] !=
                               max_tx_size) *
              3 +
          context_delta + top + left;
      // tx_split.
      if (reader_.ReadSymbol(symbol_decoder_context_.tx_split_cdf[context])) {
        const TransformSize sub_tx_size = kSplitTransformSize[node.tx_size];
        const int step_width4x4 = DivideBy4(kTransformWidth[sub_tx_size]);
        const int step_height4x4 = DivideBy4(kTransformHeight[sub_tx_size]);
        // The loops have to run in reverse order because we use a stack for
        // DFS.
        for (int i = tx_height4x4 - step_height4x4; i >= 0;
             i -= step_height4x4) {
          for (int j = tx_width4x4 - step_width4x4; j >= 0;
               j -= step_width4x4) {
            if (node.row4x4 + i >= frame_header_.rows4x4 ||
                node.column4x4 + j >= frame_header_.columns4x4) {
              continue;
            }
            stack.emplace_back(node.row4x4 + i, node.column4x4 + j, sub_tx_size,
                               node.depth + 1);
          }
        }
        continue;
      }
    }
    // tx_split is false.
    for (int i = 0; i < tx_height4x4; ++i) {
      static_assert(sizeof(TransformSize) == 1, "");
      memset(&inter_transform_sizes_[node.row4x4 + i][node.column4x4],
             node.tx_size, tx_width4x4);
    }
    block_parameters_holder_.Find(node.row4x4, node.column4x4)->transform_size =
        node.tx_size;
  }
}

void Tile::DecodeTransformSize(const Block& block) {
  const int block_width4x4 = kNum4x4BlocksWide[block.size];
  const int block_height4x4 = kNum4x4BlocksHigh[block.size];
  BlockParameters& bp = *block.bp;
  if (frame_header_.tx_mode == kTxModeSelect && block.size > kBlock4x4 &&
      bp.is_inter && !bp.skip &&
      !frame_header_.segmentation.lossless[bp.segment_id]) {
    const TransformSize max_tx_size = kMaxTransformSizeRectangle[block.size];
    const int tx_width4x4 = kTransformWidth[max_tx_size] / 4;
    const int tx_height4x4 = kTransformHeight[max_tx_size] / 4;
    for (int row = block.row4x4; row < block.row4x4 + block_height4x4;
         row += tx_height4x4) {
      for (int column = block.column4x4;
           column < block.column4x4 + block_width4x4; column += tx_width4x4) {
        ReadVariableTransformTree(block, row, column, max_tx_size);
      }
    }
  } else {
    bp.transform_size = ReadFixedTransformSize(block);
    for (int row = block.row4x4; row < block.row4x4 + block_height4x4; ++row) {
      static_assert(sizeof(TransformSize) == 1, "");
      memset(&inter_transform_sizes_[row][block.column4x4], bp.transform_size,
             block_width4x4);
    }
  }
}

}  // namespace libgav1
