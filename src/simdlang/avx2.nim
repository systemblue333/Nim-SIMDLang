import avx

export avx

{.push header: "immintrin.h".}

# operating : blend packed 32-bit integers from 'a' and 'b' using an immediate control mask
# lane unit : int32(epi32) / lane numbers : 4 (128-bit)
func mm_blend_epi32*(a, b: M128i, imm8: int32): M128i {.importc: "_mm_blend_epi32".}

# operating : compute the absolute value of packed 16-bit integers
# lane unit : int16(epi16) / lane numbers : 16 (256-bit)
func mm256_abs_epi16*(a: M256i): M256i {.importc: "_mm256_abs_epi16".}

# operating : compute the absolute value of packed 32-bit integers
# lane unit : int32(epi32) / lane numbers : 8 (256-bit)
func mm256_abs_epi32*(a: M256i): M256i {.importc: "_mm256_abs_epi32".}

# operating : compute the absolute value of packed 8-bit integers
# lane unit : int8(epi8) / lane numbers : 32 (256-bit)
func mm256_abs_epi8*(a: M256i): M256i {.importc: "_mm256_abs_epi8".}

# operating : add packed 16-bit integers
# lane unit : int16(epi16) / lane numbers : 16 (256-bit)
func mm256_add_epi16*(a, b: M256i): M256i {.importc: "_mm256_add_epi16".}

# operating : add packed 32-bit integers
# lane unit : int32(epi32) / lane numbers : 8 (256-bit)
func mm256_add_epi32*(a, b: M256i): M256i {.importc: "_mm256_add_epi32".}

# operating : add packed 64-bit integers
# lane unit : int64(epi64) / lane numbers : 4 (256-bit)
func mm256_add_epi64*(a, b: M256i): M256i {.importc: "_mm256_add_epi64".}

# operating : add packed 8-bit integers
# lane unit : int8(epi8) / lane numbers : 32 (256-bit)
func mm256_add_epi8*(a, b: M256i): M256i {.importc: "_mm256_add_epi8".}

# operating : add packed 16-bit integers with signed saturation
# lane unit : int16(epi16) / lane numbers : 16 (256-bit)
func mm256_adds_epi16*(a, b: M256i): M256i {.importc: "_mm256_adds_epi16".}

# operating : add packed 8-bit integers with signed saturation
# lane unit : int8(epi8) / lane numbers : 32 (256-bit)
func mm256_adds_epi8*(a, b: M256i): M256i {.importc: "_mm256_adds_epi8".}

# operating : add packed 16-bit integers with unsigned saturation
# lane unit : uint16(epu16) / lane numbers : 16 (256-bit)
func mm256_adds_epu16*(a, b: M256i): M256i {.importc: "_mm256_adds_epu16".}

# operating : add packed 8-bit integers with unsigned saturation
# lane unit : uint8(epu8) / lane numbers : 32 (256-bit)
func mm256_adds_epu8*(a, b: M256i): M256i {.importc: "_mm256_adds_epu8".}

# operating : concatenate 128-bit blocks and shift right by imm8 bytes within 128-bit lanes
# lane unit : int8(epi8) / lane numbers : 32 (256-bit)
func mm256_alignr_epi8*(a, b: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_alignr_epi8".}

# operating : compute the bitwise AND of 256-bit vectors
# lane unit : any (integer) / lane numbers : 256-bit total
func mm256_and_si256*(a, b: M256i): M256i {.importc: "_mm256_and_si256".}

# operating : compute the bitwise AND of NOT 'a' and 'b'
# lane unit : any (integer) / lane numbers : 256-bit total
func mm256_andnot_si256*(a, b: M256i): M256i {.importc: "_mm256_andnot_si256".}

# operating : average packed unsigned 16-bit integers (rounded up)
# lane unit : uint16(epu16) / lane numbers : 16 (256-bit)
func mm256_avg_epu16*(a, b: M256i): M256i {.importc: "_mm256_avg_epu16".}

# operating : average packed unsigned 8-bit integers (rounded up)
# lane unit : uint8(epu8) / lane numbers : 32 (256-bit)
func mm256_avg_epu8*(a, b: M256i): M256i {.importc: "_mm256_avg_epu8".}

# operating : blend packed 16-bit integers from 'a' and 'b' using an immediate control mask
# lane unit : int16(epi16) / lane numbers : 16 (256-bit)
func mm256_blend_epi16*(a, b: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_blend_epi16".}

# operating : blend packed 32-bit integers from 'a' and 'b' using an immediate control mask
# lane unit : int32(epi32) / lane numbers : 8 (256-bit)
func mm256_blend_epi32*(a, b: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_blend_epi32".}

# operating : blend packed 8-bit integers from 'a' and 'b' using a variable mask
# lane unit : int8(epi8) / lane numbers : 32 (256-bit)
func mm256_blendv_epi8*(a, b, mask: M256i): M256i {.importc: "_mm256_blendv_epi8".}

# operating : broadcast the lowest 8-bit integer from 'a' to all elements in a 256-bit vector
# lane unit : int8(epi8) / lane numbers : 32 (256-bit)
func mm256_broadcastb_epi8*(a: M128i): M256i {.importc: "_mm256_broadcastb_epi8".}

# operating : broadcast the lowest 32-bit integer from 'a' to all elements in a 256-bit vector
# lane unit : int32(epi32) / lane numbers : 8 (256-bit)
func mm256_broadcastd_epi32*(a: M128i): M256i {.importc: "_mm256_broadcastd_epi32".}

# operating : broadcast the lowest 64-bit integer from 'a' to all elements in a 256-bit vector
# lane unit : int64(epi64) / lane numbers : 4 (256-bit)
func mm256_broadcastq_epi64*(a: M128i): M256i {.importc: "_mm256_broadcastq_epi64".}

# operating : broadcast the lowest 64-bit integer from 'a' to all elements in a 128-bit vector
# lane unit : int64(epi64) / lane numbers : 2 (128-bit)
func mm_broadcastq_epi64*(a: M128i): M128i {.importc: "_mm_broadcastq_epi64".}

# operating : broadcast the lowest 32-bit integer from 'a' to all elements in a 128-bit vector
# lane unit : int32(epi32) / lane numbers : 4 (128-bit)
func mm_broadcastd_epi32*(a: M128i): M128i {.importc: "_mm_broadcastd_epi32".}

# operating : broadcast the lowest 16-bit integer from 'a' to all elements in a 128-bit vector
# lane unit : int16(epi16) / lane numbers : 8 (128-bit)
func mm_broadcastw_epi16*(a: M128i): M128i {.importc: "_mm_broadcastw_epi16".}

# operating : broadcast the lowest 8-bit integer from 'a' to all elements in a 128-bit vector
# lane unit : int8(epi8) / lane numbers : 16 (128-bit)
func mm_broadcastb_epi8*(a: M128i): M128i {.importc: "_mm_broadcastb_epi8".}

# operating : broadcast the lowest double-precision element from 'a' to all elements in a 128-bit vector
# lane unit : float64(pd) / lane numbers : 2 (128-bit)
func mm_broadcastsd_pd*(a: M128d): M128d {.importc: "_mm_broadcastsd_pd".}

# operating : broadcast the lowest single-precision element from 'a' to all elements in a 128-bit vector
# lane unit : float32(ps) / lane numbers : 4 (128-bit)
func mm_broadcastss_ps*(a: M128): M128 {.importc: "_mm_broadcastss_ps".}

# operating : broadcast 128-bits of integer data from 'a' to both 128-bit halves of a 256-bit vector
# lane unit : any (integer) / lane numbers : 128-bit x 2 (256-bit)
func mm_broadcastsi128_si256*(a: M128i): M256i {.importc: "_mm_broadcastsi128_si256".}
func mm256_broadcastsi128_si256*(a: M128i): M256i {.importc: "_mm256_broadcastsi128_si256".}

# operating : broadcast the lowest 16-bit integer from 'a' to all elements in a 256-bit vector
# lane unit : int16(epi16) / lane numbers : 16 (256-bit)
func mm256_broadcastw_epi16*(a: M128i): M256i {.importc: "_mm256_broadcastw_epi16".}

# operating : shift 128-bit blocks in 'a' left by imm8 bytes while filling with zeros
# lane unit : 128-bit block / lane numbers : 2 (256-bit)
func mm256_bslli_epi128*(a: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_bslli_epi128".}

# operating : shift 128-bit blocks in 'a' right by imm8 bytes while filling with zeros
# lane unit : 128-bit block / lane numbers : 2 (256-bit)
func mm256_bsrli_epi128*(a: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_bsrli_epi128".}

# operating : compare packed integers for equality
# lane unit : int16 / int32 / int64 / int8
# lane numbers : 16 / 8 / 4 / 32 (256-bit)
func mm256_cmpeq_epi16*(a, b: M256i): M256i {.importc: "_mm256_cmpeq_epi16".}
func mm256_cmpeq_epi32*(a, b: M256i): M256i {.importc: "_mm256_cmpeq_epi32".}
func mm256_cmpeq_epi64*(a, b: M256i): M256i {.importc: "_mm256_cmpeq_epi64".}
func mm256_cmpeq_epi8*(a, b: M256i): M256i {.importc: "_mm256_cmpeq_epi8".}

# operating : compare packed integers for greater-than
# lane unit : int16 / int32 / int64 / int8
# lane numbers : 16 / 8 / 4 / 32 (256-bit)
func mm256_cmpgt_epi16*(a, b: M256i): M256i {.importc: "_mm256_cmpgt_epi16".}
func mm256_cmpgt_epi32*(a, b: M256i): M256i {.importc: "_mm256_cmpgt_epi32".}
func mm256_cmpgt_epi64*(a, b: M256i): M256i {.importc: "_mm256_cmpgt_epi64".}
func mm256_cmpgt_epi8*(a, b: M256i): M256i {.importc: "_mm256_cmpgt_epi8".}

# operating : sign-extend packed integers from 128-bit source to 256-bit destination
# lane unit : epi16->32 / epi16->64 / epi32->64 / epi8->16 / epi8->32 / epi8->64
func mm256_cvtepi16_epi32*(a: M128i): M256i {.importc: "_mm256_cvtepi16_epi32".}
func mm256_cvtepi16_epi64*(a: M128i): M256i {.importc: "_mm256_cvtepi16_epi64".}
func mm256_cvtepi32_epi64*(a: M128i): M256i {.importc: "_mm256_cvtepi32_epi64".}
func mm256_cvtepi8_epi16*(a: M128i): M256i {.importc: "_mm256_cvtepi8_epi16".}
func mm256_cvtepi8_epi32*(a: M128i): M256i {.importc: "_mm256_cvtepi8_epi32".}
func mm256_cvtepi8_epi64*(a: M128i): M256i {.importc: "_mm256_cvtepi8_epi64".}

# operating : zero-extend packed unsigned integers from 128-bit source to 256-bit destination
# lane unit : epu16->32 / epu16->64 / epu32->64 / epu8->16 / epu8->32 / epu8->64
func mm256_cvtepu16_epi32*(a: M128i): M256i {.importc: "_mm256_cvtepu16_epi32".}
func mm256_cvtepu16_epi64*(a: M128i): M256i {.importc: "_mm256_cvtepu16_epi64".}
func mm256_cvtepu32_epi64*(a: M128i): M256i {.importc: "_mm256_cvtepu32_epi64".}
func mm256_cvtepu8_epi16*(a: M128i): M256i {.importc: "_mm256_cvtepu8_epi16".}
func mm256_cvtepu8_epi32*(a: M128i): M256i {.importc: "_mm256_cvtepu8_epi32".}
func mm256_cvtepu8_epi64*(a: M128i): M256i {.importc: "_mm256_cvtepu8_epi64".}

# operating : extract a 16-bit integer from 'a' at the specified index
# lane unit : int16(epi16) / lane numbers : 1 (from 16 elements)
func mm256_extract_epi16*(a: M256i, index: int32 | uint32): int32 {.importc: "_mm256_extract_epi16".}

# operating : extract an 8-bit integer from 'a' at the specified index
# lane unit : int8(epi8) / lane numbers : 1 (from 32 elements)
func mm256_extract_epi8*(a: M256i, index: int32 | uint32): int32 {.importc: "_mm256_extract_epi8".}

# operating : extract a 128-bit integer block from 'a' at the specified index (0 or 1)
# lane unit : 128-bit block / lane numbers : 1 (from 2 blocks)
func mm256_extracti128_si256*(a: M256i, imm8: int32 | uint32): M128i {.importc: "_mm256_extracti128_si256".}

# operating : horizontally add adjacent pairs of packed integers from 'a' and 'b'
# lane unit : int16(epi16) / int32(epi32) / lane numbers : 16 / 8 (256-bit)
func mm256_hadd_epi16*(a, b: M256i): M256i {.importc: "_mm256_hadd_epi16".}
func mm256_hadd_epi32*(a, b: M256i): M256i {.importc: "_mm256_hadd_epi32".}

# operating : horizontally add adjacent pairs of packed 16-bit integers with signed saturation
# lane unit : int16(epi16) / lane numbers : 16 (256-bit)
func mm256_hadds_epi16*(a, b: M256i): M256i {.importc: "_mm256_hadds_epi16".}

# operating : horizontally subtract adjacent pairs of packed integers from 'a' and 'b'
# lane unit : int16(epi16) / int32(epi32) / lane numbers : 16 / 8 (256-bit)
func mm256_hsub_epi16*(a, b: M256i): M256i {.importc: "_mm256_hsub_epi16".}
func mm256_hsub_epi32*(a, b: M256i): M256i {.importc: "_mm256_hsub_epi32".}

# operating : horizontally subtract adjacent pairs of packed 16-bit integers with signed saturation
# lane unit : int16(epi16) / lane numbers : 16 (256-bit)
func mm256_hsubs_epi16*(a, b: M256i): M256i {.importc: "_mm256_hsubs_epi16".}

# operating : gather packed 32-bit integers from memory using 32-bit indices
# lane unit : int32(epi32) / lane numbers : 8 (256-bit)
func mm256_i32gather_epi32*(p: pointer, vindex: M256i, scale: int32 | uint32): M256i {.importc: "_mm256_i32gather_epi32".}

# operating : conditionally gather packed 32-bit integers using a mask
# lane unit : int32(epi32) / lane numbers : 8 (256-bit)
func mm256_mask_i32gather_epi32*(src: M256i, p: pointer, vindex, mask: M256i, scale: int32 | uint32): M256i {.importc: "_mm256_mask_i32gather_epi32".}

# operating : gather packed 64-bit integers from memory using 32-bit indices
# lane unit : int64(epi64) / lane numbers : 4 (256-bit)
func mm256_i32gather_epi64*(p: pointer, vindex: M128i, scale: int32 | uint32): M256i {.importc: "_mm256_i32gather_epi64".}

# operating : conditionally gather packed 64-bit integers using a mask
# lane unit : int64(epi64) / lane numbers : 4 (256-bit)
func mm256_mask_i32gather_epi64*(src: M256i, p: pointer, vindex: M128i, mask: M256i, scale: int32 | uint32): M256i {.importc: "_mm256_mask_i32gather_epi64".}

# operating : gather packed single-precision floating-point values using 32-bit indices
# lane unit : float32(ps) / lane numbers : 8 (256-bit)
func mm256_i32gather_ps*(p: pointer, vindex: M256i, scale: int32 | uint32): M256 {.importc: "_mm256_i32gather_ps".}

# operating : conditionally gather packed single-precision values using a mask
# lane unit : float32(ps) / lane numbers : 8 (256-bit)
func mm256_mask_i32gather_ps*(src: M256, p: pointer, vindex: M256i, mask: M256, scale: int32 | uint32): M256 {.importc: "_mm256_mask_i32gather_ps".}

# operating : gather packed 32-bit integers from memory using 64-bit indices
# lane unit : int32(epi32) / lane numbers : 4 (result in 128-bit)
func mm256_i64gather_epi32*(p: pointer, vindex: M256i, scale: int32 | uint32): M128i {.importc: "_mm256_i64gather_epi32".}

# operating : conditionally gather packed 32-bit integers using 64-bit indices and a mask
# lane unit : int32(epi32) / lane numbers : 4 (result in 128-bit)
func mm256_mask_i64gather_epi32*(src: M128i, p: pointer, vindex: M256i, mask: M128i, scale: int32 | uint32): M128i {.importc: "_mm256_mask_i64gather_epi32".}

# operating : gather packed 64-bit integers from memory using 64-bit indices
# lane unit : int64(epi64) / lane numbers : 4 (256-bit)
func mm256_i64gather_epi64*(p: pointer, vindex: M256i, scale: int32 | uint32): M256i {.importc: "_mm256_i64gather_epi64".}

# operating : conditionally gather packed 64-bit integers using 64-bit indices and a mask
# lane unit : int64(epi64) / lane numbers : 4 (256-bit)
func mm256_mask_i64gather_epi64*(src: M256i, p: pointer, vindex, mask: M256i, scale: int32 | uint32): M256i {.importc: "_mm256_mask_i64gather_epi64".}

# operating : gather packed double-precision floating-point values using 64-bit indices
# lane unit : float64(pd) / lane numbers : 4 (256-bit)
func mm256_i64gather_pd*(p: pointer, vindex: M256i, scale: int32 | uint32): M256d {.importc: "_mm256_i64gather_pd".}

# operating : conditionally gather packed double-precision values using 64-bit indices and a mask
# lane unit : float64(pd) / lane numbers : 2 (128-bit output variant)
func mm256_mask_i64gather_pd*(src: M256d, p: pointer, vindex: M256i, mask: M256d, scale: int32 | uint32): M128d {.importc: "_mm256_mask_i64gather_pd".}

# operating : gather packed single-precision values using 64-bit indices
# lane unit : float32(ps) / lane numbers : 4 (result in 128-bit)
func mm256_i64gather_ps*(p: pointer, vindex: M256i, scale: int32 | uint32): M128 {.importc: "_mm256_i64gather_ps".}

# operating : conditionally gather packed single-precision values using 64-bit indices and a mask
# lane unit : float32(ps) / lane numbers : 4 (result in 128-bit)
func mm256_mask_i64gather_ps*(src: M128, p: pointer, vindex: M256i, mask: M128, scale: int32 | uint32): M128 {.importc: "_mm256_mask_i64gather_ps".}

# operating : insert a 128-bit integer block into a 256-bit vector
# lane unit : 128-bit block / lane numbers : 1 (target block)
func mm256_inserti128_si256*(a: M256i, b: M128i, imm8: int32 | uint32): M256i {.importc: "_mm256_inserti128_si256".}

# operating : multiply packed 16-bit integers, then add adjacent pairs for 32-bit results
# lane unit : int16->int32 / lane numbers : 8 (256-bit)
func mm256_madd_epi16*(a, b: M256i): M256i {.importc: "_mm256_madd_epi16".}

# operating : multiply unsigned 8-bit by signed 8-bit integers, add/saturate adjacent pairs
# lane unit : uint8/int8->int16 / lane numbers : 16 (256-bit)
func mm256_maddubs_epi16*(a, b: M256i): M256i {.importc: "_mm256_maddubs_epi16".}

# operating : load packed 32-bit integers from memory using a mask
# lane unit : int32(epi32) / lane numbers : 8 (256-bit)
func mm256_maskload_epi32(p: ptr int32, mask: M256i): M256i {.importc: "_mm256_maskload_epi32".}

# operating : template wrapper for maskload_epi32 with pointer casting
template mm256_maskload_epi32*(p: pointer, mask: M256i): M256i =
  mm256_maskload_epi32(cast[ptr int32](p), mask)

# operating : load packed 64-bit integers from memory using a mask
# lane unit : int64(epi64) / lane numbers : 4 (256-bit)
func mm256_maskload_epi64*(p: pointer, mask: M256i): M256i {.importc: "_mm256_maskload_epi64".}

# operating : store packed 32-bit integers to memory using a mask
# lane unit : int32(epi32) / lane numbers : 8 (256-bit)
func mm256_maskstore_epi32(p: ptr int32, mask, a: M256i) {.importc: "_mm256_maskstore_epi32".}

# operating : template wrapper for maskstore_epi32 with pointer casting
template mm256_maskstore_epi32*(p: pointer, mask, a: M256i) =
  mm256_maskstore_epi32(cast[ptr int32](p), mask, a)

# operating : store packed 64-bit integers to memory using a mask
# lane unit : int64(epi64) / lane numbers : 4 (256-bit)
func mm256_maskstore_epi64*(p: pointer, mask, a: M256i) {.importc: "_mm256_maskstore_epi64".}

# operating : find the maximum value of packed integers (signed/unsigned)
# lane unit : int16 / int32 / int8 / uint16 / uint32 / uint8
func mm256_max_epi16*(a, b: M256i): M256i {.importc: "_mm256_max_epi16".}
func mm256_max_epi32*(a, b: M256i): M256i {.importc: "_mm256_max_epi32".}
func mm256_max_epi8*(a, b: M256i): M256i {.importc: "_mm256_max_epi8".}
func mm256_max_epu16*(a, b: M256i): M256i {.importc: "_mm256_max_epu16".}
func mm256_max_epu32*(a, b: M256i): M256i {.importc: "_mm256_max_epu32".}
func mm256_max_epu8*(a, b: M256i): M256i {.importc: "_mm256_max_epu8".}

# operating : find the minimum value of packed integers (signed/unsigned)
# lane unit : int16 / int32 / int8 / uint16 / uint32 / uint8
func mm256_min_epi16*(a, b: M256i): M256i {.importc: "_mm256_min_epi16".}
func mm256_min_epi32*(a, b: M256i): M256i {.importc: "_mm256_min_epi32".}
func mm256_min_epi8*(a, b: M256i): M256i {.importc: "_mm256_min_epi8".}
func mm256_min_epu16*(a, b: M256i): M256i {.importc: "_mm256_min_epu16".}
func mm256_min_epu32*(a, b: M256i): M256i {.importc: "_mm256_min_epu32".}
func mm256_min_epu8*(a, b: M256i): M256i {.importc: "_mm256_min_epu8".}

# operating : create a 32-bit mask from the MSB of each 8-bit element
# lane unit : int8(epi8) / lane numbers : 32
func mm256_movemask_epi8*(a: M256i): int32 {.importc: "_mm256_movemask_epi8".}

# operating : compute sum-absolute-difference of uint8 with multiple offsets
# lane unit : uint8(epu8) / lane numbers : 32 (256-bit)
func mm256_mpsadbw_epu8*(a, b: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_mpsadbw_epu8".}

# operating : multiply packed 32-bit integers (signed/unsigned) and return 64-bit results
# lane unit : int32->int64 / uint32->uint64 / lane numbers : 4 (256-bit)
func mm256_mul_epi32*(a, b: M256i): M256i {.importc: "_mm256_mul_epi32".}
func mm256_mul_epu32*(a, b: M256i): M256i {.importc: "_mm256_mul_epu32".}

# operating : multiply packed 16-bit integers and return high/low/rounded bits
# lane unit : int16(epi16) / lane numbers : 16 (256-bit)
func mm256_mulhi_epi16*(a, b: M256i): M256i {.importc: "_mm256_mulhi_epi16".}
func mm256_mulhi_epu16*(a, b: M256i): M256i {.importc: "_mm256_mulhi_epu16".}
func mm256_mulhrs_epi16*(a, b: M256i): M256i {.importc: "_mm256_mulhrs_epi16".}
func mm256_mullo_epi16*(a, b: M256i): M256i {.importc: "_mm256_mullo_epi16".}
func mm256_mullo_epi32*(a, b: M256i): M256i {.importc: "_mm256_mullo_epi32".}

# operating : bitwise OR of two 256-bit vectors
# lane unit : any (integer) / lane numbers : 256-bit total
func mm256_or_si256*(a, b: M256i): M256i {.importc: "_mm256_or_si256".}

# operating : pack integers from larger to smaller types with saturation
# lane unit : int16->int8 / int32->int16 / unsigned variants
func mm256_packs_epi16*(a, b: M256i): M256i {.importc: "_mm256_packs_epi16".}
func mm256_packs_epi32*(a, b: M256i): M256i {.importc: "_mm256_packs_epi32".}
func mm256_packus_epi16*(a, b: M256i): M256i {.importc: "_mm256_packus_epi16".}
func mm256_packus_epi32*(a, b: M256i): M256i {.importc: "_mm256_packus_epi32".}

# operating : shuffle 128-bit blocks within 256-bit vector using imm8
# lane unit : 128-bit block / lane numbers : 2 (from 4 sources)
func mm256_permute2x128_si256*(a, b: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_permute2x128_si256".}

# operating : permute 64-bit elements within 256-bit vector using imm8
# lane unit : int64(epi64) / float64(pd) / lane numbers : 4 (256-bit)
func mm256_permute4x64_epi64*(a: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_permute4x64_epi64".}
func mm256_permute4x64_pd*(a: M256d, imm8: int32 | uint32): M256d {.importc: "_mm256_permute4x64_pd".}

# operating : permute 32-bit elements using a variable index vector
# lane unit : int32(epi32) / float32(ps) / lane numbers : 8 (256-bit)
func mm256_permutevar8x32_epi32*(a, idx: M256i): M256i {.importc: "_mm256_permutevar8x32_epi32".}
func mm256_permutevar8x32_ps*(a: M256, idx: M256i): M256 {.importc: "_mm256_permutevar8x32_ps".}

# operating : compute sum of absolute differences of unsigned 8-bit integers
# lane unit : uint8(epu8) / lane numbers : 32 (256-bit)
func mm256_sad_epu8*(a, b: M256i): M256i {.importc: "_mm256_sad_epu8".}

# operating : shuffle packed elements (32-bit/8-bit) using immediate or mask
# lane unit : int32 / int8 / hi-lo 16-bit
func mm256_shuffle_epi32*(a: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_shuffle_epi32".}
func mm256_shuffle_epi8*(a, b: M256i): M256i {.importc: "_mm256_shuffle_epi8".}
func mm256_shufflehi_epi16*(a: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_shufflehi_epi16".}
func mm256_shufflelo_epi16*(a: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_shufflelo_epi16".}

# operating : multiply packed integers by sign of 'b' (-1, 0, 1)
# lane unit : int16 / int32 / int8 / lane numbers : 16 / 8 / 32 (256-bit)
func mm256_sign_epi16*(a, b: M256i): M256i {.importc: "_mm256_sign_epi16".}
func mm256_sign_epi32*(a, b: M256i): M256i {.importc: "_mm256_sign_epi32".}
func mm256_sign_epi8*(a, b: M256i): M256i {.importc: "_mm256_sign_epi8".}

# operating : shift packed integers left (logical)
# lane unit : int16 / int32 / int64 / lane numbers : 16 / 8 / 4 (256-bit)
func mm256_sll_epi16*(a: M256i, count: M128i): M256i {.importc: "_mm256_sll_epi16".}
func mm256_sll_epi32*(a: M256i, count: M128i): M256i {.importc: "_mm256_sll_epi32".}
func mm256_sll_epi64*(a: M256i, count: M128i): M256i {.importc: "_mm256_sll_epi64".}
func mm256_slli_epi16*(a: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_slli_epi16".}
func mm256_slli_epi32*(a: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_slli_epi32".}
func mm256_slli_epi64*(a: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_slli_epi64".}
func mm256_slli_si256*(a: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_slli_si256".}
func mm256_sllv_epi32*(a: M256i, count: M256i): M256i {.importc: "_mm256_sllv_epi32".}
func mm256_sllv_epi64*(a: M256i, count: M256i): M256i {.importc: "_mm256_sllv_epi64".}

# operating : shift packed integers right (arithmetic/sign-fill)
# lane unit : int16 / int32 / lane numbers : 16 / 8 (256-bit)
func mm256_sra_epi16*(a: M256i, count: M128i): M256i {.importc: "_mm256_sra_epi16".}
func mm256_sra_epi32*(a: M256i, count: M128i): M256i {.importc: "_mm256_sra_epi32".}
func mm256_srai_epi16*(a: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_srai_epi16".}
func mm256_srai_epi32*(a: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_srai_epi32".}
func mm256_srav_epi32*(a: M256i, count: M256i): M256i {.importc: "_mm256_srav_epi32".}

# operating : shift packed integers right (logical/zero-fill)
# lane unit : int16 / int32 / int64 / lane numbers : 16 / 8 / 4 (256-bit)
func mm256_srl_epi16*(a: M256i, count: M128i): M256i {.importc: "_mm256_srl_epi16".}
func mm256_srl_epi32*(a: M256i, count: M128i): M256i {.importc: "_mm256_srl_epi32".}
func mm256_srl_epi64*(a: M256i, count: M128i): M256i {.importc: "_mm256_srl_epi64".}
func mm256_srli_epi16*(a: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_srli_epi16".}
func mm256_srli_epi32*(a: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_srli_epi32".}
func mm256_srli_epi64*(a: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_srli_epi64".}
func mm256_srli_si256*(a: M256i, imm8: int32 | uint32): M256i {.importc: "_mm256_srli_si256".}
func mm256_srlv_epi32*(a: M256i, count: M256i): M256i {.importc: "_mm256_srlv_epi32".}
func mm256_srlv_epi64*(a: M256i, count: M256i): M256i {.importc: "_mm256_srlv_epi64".}

# operating : load a 256-bit vector from memory using a non-temporal hint (streaming)
# lane unit : any (integer) / lane numbers : 256-bit total
func mm256_stream_load_si256*(p: pointer): M256i {.importc: "_mm256_stream_load_si256".}

# operating : subtract packed integers (standard/saturated)
# lane unit : int16 / int32 / int64 / int8 / unsigned variants
func mm256_sub_epi16*(a, b: M256i): M256i {.importc: "_mm256_sub_epi16".}
func mm256_sub_epi32*(a, b: M256i): M256i {.importc: "_mm256_sub_epi32".}
func mm256_sub_epi64*(a, b: M256i): M256i {.importc: "_mm256_sub_epi64".}
func mm256_sub_epi8*(a, b: M256i): M256i {.importc: "_mm256_sub_epi8".}
func mm256_subs_epi16*(a, b: M256i): M256i {.importc: "_mm256_subs_epi16".}
func mm256_subs_epi8*(a, b: M256i): M256i {.importc: "_mm256_subs_epi8".}
func mm256_subs_epu16*(a, b: M256i): M256i {.importc: "_mm256_subs_epu16".}
func mm256_subs_epu8*(a, b: M256i): M256i {.importc: "_mm256_subs_epu8".}

# operating : unpack and interleave high/low elements from two vectors
# lane unit : int8 / int16 / int32 / int64 / lane numbers : 16x2 / 8x2 / 4x2 / 2x2 per 128-bit lane
func mm256_unpackhi_epi16*(a, b: M256i): M256i {.importc: "_mm256_unpackhi_epi16".}
func mm256_unpackhi_epi32*(a, b: M256i): M256i {.importc: "_mm256_unpackhi_epi32".}
func mm256_unpackhi_epi64*(a, b: M256i): M256i {.importc: "_mm256_unpackhi_epi64".}
func mm256_unpackhi_epi8*(a, b: M256i): M256i {.importc: "_mm256_unpackhi_epi8".}
func mm256_unpacklo_epi16*(a, b: M256i): M256i {.importc: "_mm256_unpacklo_epi16".}
func mm256_unpacklo_epi32*(a, b: M256i): M256i {.importc: "_mm256_unpacklo_epi32".}
func mm256_unpacklo_epi64*(a, b: M256i): M256i {.importc: "_mm256_unpacklo_epi64".}
func mm256_unpacklo_epi8*(a, b: M256i): M256i {.importc: "_mm256_unpacklo_epi8".}

# operating : bitwise XOR of two 256-bit vectors
# lane unit : any (integer) / lane numbers : 256-bit total
func mm256_xor_si256*(a, b: M256i): M256i {.importc: "_mm256_xor_si256".}

{.pop.}
