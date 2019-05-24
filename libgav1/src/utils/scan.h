#ifndef LIBGAV1_SRC_UTILS_SCAN_H_
#define LIBGAV1_SRC_UTILS_SCAN_H_

#include <cstdint>

#include "src/utils/constants.h"

namespace libgav1 {

const uint16_t* GetScan(TransformSize tx_size,
                        TransformType tx_type);  // 5.11.41.

}  // namespace libgav1

#endif  // LIBGAV1_SRC_UTILS_SCAN_H_
