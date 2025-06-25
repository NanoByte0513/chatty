#pragma once

namespace chatty {
typedef enum {
    DTypeMin = kFloat32,
    kFloat32 = 0,
    kFloat16 = 1,
    kFloat8 = 2,
    kInt64 = 3,
    kInt32 = 4,
    kInt16 = 5,
    kInt8 = 6,
    kInt4 = 7,
    kUint64 = 8,
    kUint32 = 9,
    kUint16 = 10,
    kUint8 = 11,
    kBF16 = 12,
    DTypeMax = 12
} DType;

typedef enum {
    QuantTypeMin = kQuantTypeFP16,
    kQuantTypeFP16 = 0,
    kQuantTypeW16A16 = 1,
} QuantType;
} // namespace chatty