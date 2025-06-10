#pragma once

namespace chatty {
typedef enum {
    DTypeMin = 0,
    Float32 = 0,
    Float16 = 1,
    Float8 = 2,
    Int64 = 3,
    Int32 = 4,
    Int16 = 5,
    Int8 = 6,
    Int4 = 7,
    Uint64 = 8,
    Uint32 = 9,
    Uint16 = 10,
    Uint8 = 11,
    BF16 = 12,
    DTypeMax = 12
} DType;
} // namespace chatty