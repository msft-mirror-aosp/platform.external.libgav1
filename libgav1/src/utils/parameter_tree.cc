#include "src/utils/parameter_tree.h"

#include <cassert>
#include <new>

#include "src/utils/common.h"
#include "src/utils/constants.h"
#include "src/utils/logging.h"
#include "src/utils/types.h"

namespace libgav1 {

void ParameterTree::SetPartitionType(Partition partition) {
  assert(!partition_type_set_);
  partition_ = partition;
  partition_type_set_ = true;
  const int block_width4x4 = kNum4x4BlocksWide[block_size_];
  const int half_block4x4 = block_width4x4 >> 1;
  const int quarter_block4x4 = half_block4x4 >> 1;
  const BlockSize sub_size = kSubSize[partition][block_size_];
  const BlockSize split_size = kSubSize[kPartitionSplit][block_size_];
  assert(partition == kPartitionNone || sub_size != kBlockInvalid);
  switch (partition) {
    case kPartitionNone:
      parameters_.reset(new (std::nothrow) BlockParameters());
      return;
    case kPartitionHorizontal:
      children_[0].reset(new (std::nothrow) ParameterTree(row4x4_, column4x4_,
                                                          sub_size, true));
      children_[1].reset(new (std::nothrow) ParameterTree(
          row4x4_ + half_block4x4, column4x4_, sub_size, true));
      return;
    case kPartitionVertical:
      children_[0].reset(new (std::nothrow) ParameterTree(row4x4_, column4x4_,
                                                          sub_size, true));
      children_[1].reset(new (std::nothrow) ParameterTree(
          row4x4_, column4x4_ + half_block4x4, sub_size, true));
      return;
    case kPartitionSplit:
      children_[0].reset(new (std::nothrow) ParameterTree(row4x4_, column4x4_,
                                                          sub_size, false));
      children_[1].reset(new (std::nothrow) ParameterTree(
          row4x4_, column4x4_ + half_block4x4, sub_size, false));
      children_[2].reset(new (std::nothrow) ParameterTree(
          row4x4_ + half_block4x4, column4x4_, sub_size, false));
      children_[3].reset(new (std::nothrow) ParameterTree(
          row4x4_ + half_block4x4, column4x4_ + half_block4x4, sub_size,
          false));
      return;
    case kPartitionHorizontalWithTopSplit:
      assert(split_size != kBlockInvalid);
      children_[0].reset(new (std::nothrow) ParameterTree(row4x4_, column4x4_,
                                                          split_size, true));
      children_[1].reset(new (std::nothrow) ParameterTree(
          row4x4_, column4x4_ + half_block4x4, split_size, true));
      children_[2].reset(new (std::nothrow) ParameterTree(
          row4x4_ + half_block4x4, column4x4_, sub_size, true));
      return;
    case kPartitionHorizontalWithBottomSplit:
      assert(split_size != kBlockInvalid);
      children_[0].reset(new (std::nothrow) ParameterTree(row4x4_, column4x4_,
                                                          sub_size, true));
      children_[1].reset(new (std::nothrow) ParameterTree(
          row4x4_ + half_block4x4, column4x4_, split_size, true));
      children_[2].reset(new (std::nothrow) ParameterTree(
          row4x4_ + half_block4x4, column4x4_ + half_block4x4, split_size,
          true));
      return;
    case kPartitionVerticalWithLeftSplit:
      assert(split_size != kBlockInvalid);
      children_[0].reset(new (std::nothrow) ParameterTree(row4x4_, column4x4_,
                                                          split_size, true));
      children_[1].reset(new (std::nothrow) ParameterTree(
          row4x4_ + half_block4x4, column4x4_, split_size, true));
      children_[2].reset(new (std::nothrow) ParameterTree(
          row4x4_, column4x4_ + half_block4x4, sub_size, true));
      return;
    case kPartitionVerticalWithRightSplit:
      assert(split_size != kBlockInvalid);
      children_[0].reset(new (std::nothrow) ParameterTree(row4x4_, column4x4_,
                                                          sub_size, true));
      children_[1].reset(new (std::nothrow) ParameterTree(
          row4x4_, column4x4_ + half_block4x4, split_size, true));
      children_[2].reset(new (std::nothrow) ParameterTree(
          row4x4_ + half_block4x4, column4x4_ + half_block4x4, split_size,
          true));
      return;
    case kPartitionHorizontal4:
      for (int i = 0; i < 4; ++i) {
        children_[i].reset(new (std::nothrow) ParameterTree(
            row4x4_ + i * quarter_block4x4, column4x4_, sub_size, true));
      }
      return;
    case kPartitionVertical4:
      for (int i = 0; i < 4; ++i) {
        children_[i].reset(new (std::nothrow) ParameterTree(
            row4x4_, column4x4_ + i * quarter_block4x4, sub_size, true));
      }
      return;
  }
}

BlockParameters* ParameterTree::Find(int row4x4, int column4x4) const {
  if (!partition_type_set_ || row4x4 < row4x4_ || column4x4 < column4x4_ ||
      row4x4 >= row4x4_ + kNum4x4BlocksHigh[block_size_] ||
      column4x4 >= column4x4_ + kNum4x4BlocksWide[block_size_]) {
    // Either partition type is not set or the search range is out of bound.
    return nullptr;
  }
  const ParameterTree* node = this;
  while (node->partition_ != kPartitionNone) {
    if (!node->partition_type_set_) {
      LIBGAV1_DLOG(ERROR,
                   "Partition type was not set for one of the nodes in the "
                   "path to row4x4: %d column4x4: %d.",
                   row4x4, column4x4);
      return nullptr;
    }
    const int block_width4x4 = kNum4x4BlocksWide[node->block_size_];
    const int half_block4x4 = block_width4x4 >> 1;
    const int quarter_block4x4 = half_block4x4 >> 1;
    switch (node->partition_) {
      case kPartitionNone:
        assert(false);
        break;
      case kPartitionHorizontal:
        if (row4x4 < node->row4x4_ + half_block4x4) {
          node = node->children_[0].get();
        } else {
          node = node->children_[1].get();
        }
        break;
      case kPartitionVertical:
        if (column4x4 < node->column4x4_ + half_block4x4) {
          node = node->children_[0].get();
        } else {
          node = node->children_[1].get();
        }
        break;
      case kPartitionSplit:
        if (row4x4 < node->row4x4_ + half_block4x4 &&
            column4x4 < node->column4x4_ + half_block4x4) {
          node = node->children_[0].get();
        } else if (row4x4 < node->row4x4_ + half_block4x4) {
          node = node->children_[1].get();
        } else if (column4x4 < node->column4x4_ + half_block4x4) {
          node = node->children_[2].get();
        } else {
          node = node->children_[3].get();
        }
        break;
      case kPartitionHorizontalWithTopSplit:
        if (row4x4 < node->row4x4_ + half_block4x4 &&
            column4x4 < node->column4x4_ + half_block4x4) {
          node = node->children_[0].get();
        } else if (row4x4 < node->row4x4_ + half_block4x4) {
          node = node->children_[1].get();
        } else {
          node = node->children_[2].get();
        }
        break;
      case kPartitionHorizontalWithBottomSplit:
        if (row4x4 < node->row4x4_ + half_block4x4) {
          node = node->children_[0].get();
        } else if (column4x4 < node->column4x4_ + half_block4x4) {
          node = node->children_[1].get();
        } else {
          node = node->children_[2].get();
        }
        break;
      case kPartitionVerticalWithLeftSplit:
        if (row4x4 < node->row4x4_ + half_block4x4 &&
            column4x4 < node->column4x4_ + half_block4x4) {
          node = node->children_[0].get();
        } else if (column4x4 < node->column4x4_ + half_block4x4) {
          node = node->children_[1].get();
        } else {
          node = node->children_[2].get();
        }
        break;
      case kPartitionVerticalWithRightSplit:
        if (column4x4 < node->column4x4_ + half_block4x4) {
          node = node->children_[0].get();
        } else if (row4x4 < node->row4x4_ + half_block4x4) {
          node = node->children_[1].get();
        } else {
          node = node->children_[2].get();
        }
        break;
      case kPartitionHorizontal4:
        for (int i = 0; i < 4; ++i) {
          if (row4x4 < node->row4x4_ + quarter_block4x4 * (i + 1)) {
            node = node->children_[i].get();
            break;
          }
        }
        break;
      case kPartitionVertical4:
        for (int i = 0; i < 4; ++i) {
          if (column4x4 < node->column4x4_ + quarter_block4x4 * (i + 1)) {
            node = node->children_[i].get();
            break;
          }
        }
        break;
    }
  }
  return node->parameters_.get();
}

}  // namespace libgav1
