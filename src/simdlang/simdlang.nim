import ../../../Utility/src/Utility/codeutils/indexutils
import ../../../Utility/src/Utility/codeutils/bits
import ../../../Utility/src/Utility/codeutils/errorutils
import ../../../Utility/src/Utility/dataformat/dataformat
import ../../../Utility/src/Utility/codeutils/arraytype
import ../../../Utility/src/Utility/codeutils/autoopt
import ../../../Utility/src/Utility/codeutils/envconst
import std/[monotimes, times]
import std/bitops
import strutils
import simdExists
import sse
import sse2
import sse3
import sse41
import sse42
import ssse3
import avx
import avx2
import avx10
import avx512f
import neon
import lsx
import avx512vl
import lasx

func compileTimeParseInt(s: string): int =
  if s.len == 0: return 0
  for c in s:
    if c in {'0'..'9'}:
      result = result * 10 + (ord(c) - ord('0'))
    else:
      return 0

const
  Limit*: int = (when defined(amd64): 512 
                 elif defined(arm) or defined(aarch64) or 
                      defined(riscv32) or defined(riscv64) or
                      defined(mips) or defined(mipsel) or
                      defined(mips64) or defined(mips64el) or
                      defined(ppc) or defined(ppc64) or
                      defined(ppc64el): 128
                 elif defined(loongarch): 256) div 8
  SIMD_LEN_STR {.strdefine: "simdlen".} = "0"

  Expect*: int = block:
    let parsed = compileTimeParseInt(SIMD_LEN_STR)
    if parsed <= 0:
      Limit
    else:
      parsed

type
  vector[B: static int, T] = array[B div (sizeof(T) * 8), T]

  v128[T] = vector[128, T]
  v256[T] = vector[256, T]
  v512[T] = vector[512, T]

  # --- 128-bit Vectors ---
  f32v128* = v128[float32]
  f64v128* = v128[float64]
  i8v128* = v128[int8]
  i16v128* = v128[int16]
  i32v128* = v128[int32]
  i64v128* = v128[int64]
  u8v128* = v128[uint8]
  u16v128* = v128[uint16]
  u32v128* = v128[uint32]
  u64v128* = v128[uint64]

  # --- 256-bit Vectors ---
  f32v256* = v256[float32]
  f64v256* = v256[float64]
  i8v256* = v256[int8]
  i16v256* = v256[int16]
  i32v256* = v256[int32]
  i64v256* = v256[int64]
  u8v256* = v256[uint8]
  u16v256* = v256[uint16]
  u32v256* = v256[uint32]
  u64v256* = v256[uint64]

  # --- 512-bit Vectors ---
  f32v512* = v512[float32]
  f64v512* = v512[float64]
  i8v512* = v512[int8]
  i16v512* = v512[int16]
  i32v512* = v512[int32]
  i64v512* = v512[int64]
  u8v512* = v512[uint8]
  u16v512* = v512[uint16]
  u32v512* = v512[uint32]
  u64v512* = v512[uint64]

when defined(amd64):
  type
    f32r128* = sse.M128
    f64r128* = sse2.M128d
    i8r128* = sse2.M128i
    i16r128* = sse2.M128i
    i32r128* = sse2.M128i
    i64r128* = sse2.M128i
    u8r128* = sse2.M128i
    u16r128* = sse2.M128i
    u32r128* = sse2.M128i
    u64r128* = sse2.M128i

    f32r256* = avx.M256
    f64r256* = avx.M256d
    i8r256* = avx.M256i
    i16r256* = avx.M256i
    i32r256* = avx.M256i
    i64r256* = avx.M256i
    u8r256* = avx.M256i
    u16r256* = avx.M256i
    u32r256* = avx.M256i
    u64r256* = avx.M256i

    f32r512* = avx512f.M512
    f64r512* = avx512f.M512d
    i8r512* = avx512f.M512i
    i16r512* = avx512f.M512i
    i32r512* = avx512f.M512i
    i64r512* = avx512f.M512i
    u8r512* = avx512f.M512i
    u16r512* = avx512f.M512i
    u32r512* = avx512f.M512i
    u64r512* = avx512f.M512i
elif defined(aarch64):
  type
    f32r128* = neon.Float32x4
    f64r128* = neon.Float64x2
    i8r128* = neon.Int8x16
    i16r128* = neon.Int16x8
    i32r128* = neon.Int32x4
    i64r128* = neon.Int64x2
    u8r128* = neon.Uint8x16
    u16r128* = neon.Uint16x8
    u32r128* = neon.Uint32x4
    u64r128* = neon.Uint64x2

    f32r256* = array[2, f32r128]
    f64r256* = array[2, f64r128]
    i8r256* = array[2, i8r128]
    i16r256* = array[2, i16r128]
    i32r256* = array[2, i32r128]
    i64r256* = array[2, i64r128]
    u8r256* = array[2, u8r128]
    u16r256* = array[2, u16r128]
    u32r256* = array[2, u32r128]
    u64r256* = array[2, u64r128]

    f32r512* = array[4, f32r128]
    f64r512* = array[4, f64r128]
    i8r512* = array[4, i8r128]
    i16r512* = array[4, i16r128]
    i32r512* = array[4, i32r128]
    i64r512* = array[4, i64r128]
    u8r512* = array[4, u8r128]
    u16r512* = array[4, u16r128]
    u32r512* = array[4, u32r128]
    u64r512* = array[4, u64r128]
elif defined(loongarch):
  type
    f32r128* = lsx.M128
    f64r128* = lsx.M128d
    i8r128* = lsx.M128i
    i16r128* = lsx.M128i
    i32r128* = lsx.M128i
    i64r128* = lsx.M128i
    u8r128* = lsx.M128i
    u16r128* = lsx.M128i
    u32r128* = lsx.M128i
    u64r128* = lsx.M128i

    f32r256* = lasx.M256
    f64r256* = lasx.M256d
    i8r256* = lasx.M256i
    i16r256* = lasx.M256i
    i32r256* = lasx.M256i
    i64r256* = lasx.M256i
    u8r256* = lasx.M256i
    u16r256* = lasx.M256i
    u32r256* = lasx.M256i
    u64r256* = lasx.M256i

    f32r512* = array[2, f32r256]
    f64r512* = array[2, f64r256]
    i8r512* = array[2, i8r256]
    i16r512* = array[2, i16r256]
    i32r512* = array[2, i32r256]
    i64r512* = array[2, i64r256]
    u8r512* = array[2, u8r256]
    u16r512* = array[2, u16r256]
    u32r512* = array[2, u32r256]
    u64r512* = array[2, u64r256]

# +(128), -, *, div, mod, sqrt, abs, min, max, addSat, subSat
# and, or, not, xor, nand, nor
# shl, shr, rotl, rotr
# ==, !=, >=, <=, >, <
# loadU(Unaligned), loadA(aligned), store, broadcast
# set1, setzero
# hadd, hsub, dotproduct(horizontal)
# shuffle(not now), permute(not now), unpackHi/Lo, blend
# convert, bitcast
# gather, scatter
# maskLoad, maskStore
# sar(not now), fmadd(not now)
# floor, ceil, round
# testZ, testC
# movemask
# clmul, pmull, popcnt

# ========================
# load / value
# ========================

template load*(a: f32v128): f32r128 = 
  cast[f32r128](a)

template load*(a: f64v128): f64r128 = 
  cast[f64r128](a)

template load*(a: i8v128): i8r128 = 
  cast[i8r128](a)

template load*(a: i16v128): i16r128 = 
  cast[i16r128](a)

template load*(a: i32v128): i32r128 = 
  cast[i32r128](a)

template load*(a: i64v128): i64r128 = 
  cast[i64r128](a)

template load*(a: u8v128): u8r128 = 
  cast[u8r128](a)

template load*(a: u16v128): u16r128 = 
  cast[u16r128](a)

template load*(a: u32v128): u32r128 = 
  cast[u32r128](a)

template load*(a: u64v128): u64r128 = 
  cast[u64r128](a)



template load*(a: f32v256): f32r256 =
  cast[f32r256](a)

template load*(a: f64v256): f64r256 =
  cast[f64r256](a)

template load*(a: i8v256): i8r256 =
  cast[i8r256](a)

template load*(a: i16v256): i16r256 =
  cast[i16r256](a)

template load*(a: i32v256): i32r256 =
  cast[i32r256](a)

template load*(a: i64v256): i64r256 =
  cast[i64r256](a)

template load*(a: u8v256): u8r256 =
  cast[u8r256](a)

template load*(a: u16v256): u16r256 =
  cast[u16r256](a)

template load*(a: u32v256): u32r256 =
  cast[u32r256](a)

template load*(a: u64v256): u64r256 =
  cast[u64r256](a)



template load*(a: f32v512): f32r512 =
  cast[f32r512](a)

template load*(a: f64v512): f64r512 =
  cast[f64r512](a)

template load*(a: i8v512): i8r512 =
  cast[i8r512](a)

template load*(a: i16v512): i16r512 =
  cast[i16r512](a)

template load*(a: i32v512): i32r512 =
  cast[i32r512](a)

template load*(a: i64v512): i64r512 =
  cast[i64r512](a)

template load*(a: u8v512): u8r512 =
  cast[u8r512](a)

template load*(a: u16v512): u16r512 =
  cast[u16r512](a)

template load*(a: u32v512): u32r512 =
  cast[u32r512](a)

template load*(a: u64v512): u64r512 =
  cast[u64r512](a)

# ========================
# store / value
# ========================
  

# ========================
# add / f32v128 / value
# ========================

func add_f32v128_fallback(a, b: f32v128): f32v128 =
  var output: f32v128
  unroll(i, 0, 3):
    output[i] = a[i] + b[i]
  return output

func add_f32v128_sse(a, b: f32v128): f32v128 =
  return cast[f32v128](mm_add_ps(cast[sse.M128](a), cast[sse.M128](b)))

func add_f32v128_avx512(a, b: f32v128): f32v128 = 
  let mask: MMask8 = cast[MMask8](0x0F'u8)
  return cast[f32v128](mm_maskz_add_ps(mask, cast[sse.M128](a), cast[sse.M128](b)))

let x86_add_f32v128: proc (a, b: f32v128): f32v128 = 
  if avx512Exists(): add_f32v128_avx512
  elif sseExists(): add_f32v128_sse 
  else: add_f32v128_fallback 

func add_f32v128_neon(a, b: f32v128): f32v128 =
  return cast[f32v128](vaddq_f32(cast[Float32x4](a), cast[Float32x4](b)))

let arm_add_f32v128: proc (a, b: f32v128): f32v128 = if neonExists(): add_f32v128_neon else: add_f32v128_fallback

func add_f32v128_lsx(a, b: f32v128): f32v128 =
  return cast[f32v128](lsx_vfadd_s(cast[lsx.M128](a), cast[lsx.M128](b)))

let la_add_f32v128: proc (a, b: f32v128): f32v128 = if lsxExists(): add_f32v128_lsx else: add_f32v128_fallback

template `+`(a, b: f32v128): f32v128 = 
  when defined(amd64):
    x86_add_f32v128(a, b)
  elif defined(aarch64):
    arm_add_f32v128(a, b)
  elif defined(loongarch):
    la_add_f32v128(a, b)

# ========================
# add / f64v128 / value
# ========================

func add_f64v128_fallback(a, b: f64v128): f64v128 =
  var output: f64v128
  unroll(i, 0, 1):
    output[i] = a[i] + b[i]
  return output

func add_f64v128_avx512(a, b: f64v128): f64v128 =
  let mask = cast[MMask8](0x03'u8)
  return cast[f64v128](mm_maskz_add_pd(mask, cast[sse2.M128d](a), cast[sse2.M128d](b)))

func add_f64v128_sse(a, b: f64v128): f64v128 =
  return cast[f64v128](mm_add_pd(cast[sse2.M128d](a), cast[sse2.M128d](b)))

let x86_add_f64v128: proc (a, b: f64v128): f64v128 = 
  if avx512Exists(): add_f64v128_avx512
  elif sseExists(): add_f64v128_sse 
  else: add_f64v128_fallback 

func add_f64v128_neon(a, b: f64v128): f64v128 =
  return cast[f64v128](vaddq_f64(cast[Float64x2](a), cast[Float64x2](b)))

let arm_add_f64v128: proc (a, b: f64v128): f64v128 = if neonExists(): add_f64v128_neon else: add_f64v128_fallback

func add_f64v128_lsx(a, b: f64v128): f64v128 =
  return cast[f64v128](lsx_vfadd_d(cast[lsx.M128d](a), cast[lsx.M128d](b)))

let la_add_f64v128: proc (a, b: f64v128): f64v128 = if lsxExists(): add_f64v128_lsx else: add_f64v128_fallback

template `+`(a, b: f64v128): f64v128 = 
  when defined(amd64):
    x86_add_f64v128(a, b)
  elif defined(aarch64):
    arm_add_f64v128(a, b)
  elif defined(loongarch):
    la_add_f64v128(a, b)

# ========================
# add / i8v128 / value
# ========================

func add_i8v128_fallback(a, b: i8v128): i8v128 =
  var output: i8v128
  unroll(i, 0, 15):
    output[i] = a[i] + b[i]
  return output

func add_i8v128_sse(a, b: i8v128): i8v128 =
  return cast[i8v128](mm_add_epi8(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_add_i8v128: proc (a, b: i8v128): i8v128 = 
  if sseExists(): add_i8v128_sse 
  else: add_i8v128_fallback 

func add_i8v128_neon(a, b: i8v128): i8v128 =
  return cast[i8v128](vaddq_s8(cast[Int8x16](a), cast[Int8x16](b)))

let arm_add_i8v128: proc (a, b: i8v128): i8v128 = if neonExists(): add_i8v128_neon else: add_i8v128_fallback

func add_i8v128_lsx(a, b: i8v128): i8v128 =
  return cast[i8v128](lsx_vadd_b(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_add_i8v128: proc (a, b: i8v128): i8v128 = if lsxExists(): add_i8v128_lsx else: add_i8v128_fallback

template `+`(a, b: i8v128): i8v128 = 
  when defined(amd64):
    x86_add_i8v128(a, b)
  elif defined(aarch64):
    arm_add_i8v128(a, b)
  elif defined(loongarch):
    la_add_i8v128(a, b)

# ========================
# add / i16v128 / value
# ========================

func add_i16v128_fallback(a, b: i16v128): i16v128 =
  var output: i16v128
  unroll(i, 0, 7):
    output[i] = a[i] + b[i]
  return output

func add_i16v128_sse(a, b: i16v128): i16v128 =
  return cast[i16v128](mm_add_epi16(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_add_i16v128: proc (a, b: i16v128): i16v128 = 
  if sseExists(): add_i16v128_sse 
  else: add_i16v128_fallback 

func add_i16v128_neon(a, b: i16v128): i16v128 =
  return cast[i16v128](vaddq_s16(cast[Int16x8](a), cast[Int16x8](b)))

let arm_add_i16v128: proc (a, b: i16v128): i16v128 = if neonExists(): add_i16v128_neon else: add_i16v128_fallback

func add_i16v128_lsx(a, b: i16v128): i16v128 =
  return cast[i16v128](lsx_vadd_h(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_add_i16v128: proc (a, b: i16v128): i16v128 = if lsxExists(): add_i16v128_lsx else: add_i16v128_fallback

template `+`(a, b: i16v128): i16v128 = 
  when defined(amd64):
    x86_add_i16v128(a, b)
  elif defined(aarch64):
    arm_add_i16v128(a, b)
  elif defined(loongarch):
    la_add_i16v128(a, b)

# ========================
# add / i32v128 / value
# ========================

func add_i32v128_fallback(a, b: i32v128): i32v128 =
  var output: i32v128
  unroll(i, 0, 3):
    output[i] = a[i] + b[i]
  return output

func add_i32v128_sse(a, b: i32v128): i32v128 =
  return cast[i32v128](mm_add_epi32(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_add_i32v128: proc (a, b: i32v128): i32v128 = 
  if sseExists(): add_i32v128_sse 
  else: add_i32v128_fallback 

func add_i32v128_neon(a, b: i32v128): i32v128 =
  return cast[i32v128](vaddq_s32(cast[Int32x4](a), cast[Int32x4](b)))

let arm_add_i32v128: proc (a, b: i32v128): i32v128 = if neonExists(): add_i32v128_neon else: add_i32v128_fallback

func add_i32v128_lsx(a, b: i32v128): i32v128 =
  return cast[i32v128](lsx_vadd_w(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_add_i32v128: proc (a, b: i32v128): i32v128 = if lsxExists(): add_i32v128_lsx else: add_i32v128_fallback

template `+`(a, b: i32v128): i32v128 = 
  when defined(amd64):
    x86_add_i32v128(a, b)
  elif defined(aarch64):
    arm_add_i32v128(a, b)
  elif defined(loongarch):
    la_add_i32v128(a, b)

# ========================
# add / i64v128 / value
# ========================

func add_i64v128_fallback(a, b: i64v128): i64v128 =
  var output: i64v128
  unroll(i, 0, 1):
    output[i] = a[i] + b[i]
  return output

func add_i64v128_sse(a, b: i64v128): i64v128 =
  return cast[i64v128](mm_add_epi64(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_add_i64v128: proc (a, b: i64v128): i64v128 = 
  if sseExists(): add_i64v128_sse 
  else: add_i64v128_fallback 

func add_i64v128_neon(a, b: i64v128): i64v128 =
  return cast[i64v128](vaddq_s64(cast[Int64x2](a), cast[Int64x2](b)))

let arm_add_i64v128: proc (a, b: i64v128): i64v128 = if neonExists(): add_i64v128_neon else: add_i64v128_fallback

func add_i64v128_lsx(a, b: i64v128): i64v128 =
  return cast[i64v128](lsx_vadd_d(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_add_i64v128: proc (a, b: i64v128): i64v128 = if lsxExists(): add_i64v128_lsx else: add_i64v128_fallback

template `+`(a, b: i64v128): i64v128 = 
  when defined(amd64):
    x86_add_i64v128(a, b)
  elif defined(aarch64):
    arm_add_i64v128(a, b)
  elif defined(loongarch):
    la_add_i64v128(a, b)

# ========================
# add / u8v128 / value
# ========================

func add_u8v128_fallback(a, b: u8v128): u8v128 =
  var output: u8v128
  unroll(i, 0, 15):
    output[i] = a[i] + b[i]
  return output

func add_u8v128_sse(a, b: u8v128): u8v128 =
  return cast[u8v128](mm_add_epi8(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_add_u8v128: proc (a, b: u8v128): u8v128 = 
  if sseExists(): add_u8v128_sse 
  else: add_u8v128_fallback 

func add_u8v128_neon(a, b: u8v128): u8v128 =
  return cast[u8v128](vaddq_u8(cast[Uint8x16](a), cast[Uint8x16](b)))

let arm_add_u8v128: proc (a, b: u8v128): u8v128 = if neonExists(): add_u8v128_neon else: add_u8v128_fallback

func add_u8v128_lsx(a, b: u8v128): u8v128 =
  return cast[u8v128](lsx_vadd_b(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_add_u8v128: proc (a, b: u8v128): u8v128 = if lsxExists(): add_u8v128_lsx else: add_u8v128_fallback

template `+`(a, b: u8v128): u8v128 = 
  when defined(amd64):
    x86_add_u8v128(a, b)
  elif defined(aarch64):
    arm_add_u8v128(a, b)
  elif defined(loongarch):
    la_add_u8v128(a, b)

# ========================
# add / u16v128 / value
# ========================

func add_u16v128_fallback(a, b: u16v128): u16v128 =
  var output: u16v128
  unroll(i, 0, 7):
    output[i] = a[i] + b[i]
  return output

func add_u16v128_sse(a, b: u16v128): u16v128 =
  return cast[u16v128](mm_add_epi16(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_add_u16v128: proc (a, b: u16v128): u16v128 = 
  if sseExists(): add_u16v128_sse 
  else: add_u16v128_fallback 

func add_u16v128_neon(a, b: u16v128): u16v128 =
  return cast[u16v128](vaddq_u16(cast[Uint16x8](a), cast[Uint16x8](b)))

let arm_add_u16v128: proc (a, b: u16v128): u16v128 = if neonExists(): add_u16v128_neon else: add_u16v128_fallback

func add_u16v128_lsx(a, b: u16v128): u16v128 =
  return cast[u16v128](lsx_vadd_h(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_add_u16v128: proc (a, b: u16v128): u16v128 = if lsxExists(): add_u16v128_lsx else: add_u16v128_fallback

template `+`(a, b: u16v128): u16v128 = 
  when defined(amd64):
    x86_add_u16v128(a, b)
  elif defined(aarch64):
    arm_add_u16v128(a, b)
  elif defined(loongarch):
    la_add_u16v128(a, b)

# ========================
# add / u32v128 / value
# ========================

func add_u32v128_fallback(a, b: u32v128): u32v128 =
  var output: u32v128
  unroll(i, 0, 3):
    output[i] = a[i] + b[i]
  return output

func add_u32v128_sse(a, b: u32v128): u32v128 =
  return cast[u32v128](mm_add_epi32(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_add_u32v128: proc (a, b: u32v128): u32v128 = 
  if sseExists(): add_u32v128_sse 
  else: add_u32v128_fallback 

func add_u32v128_neon(a, b: u32v128): u32v128 =
  return cast[u32v128](vaddq_u32(cast[Uint32x4](a), cast[Uint32x4](b)))

let arm_add_u32v128: proc (a, b: u32v128): u32v128 = if neonExists(): add_u32v128_neon else: add_u32v128_fallback

func add_u32v128_lsx(a, b: u32v128): u32v128 =
  return cast[u32v128](lsx_vadd_w(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_add_u32v128: proc (a, b: u32v128): u32v128 = if lsxExists(): add_u32v128_lsx else: add_u32v128_fallback

template `+`(a, b: u32v128): u32v128 = 
  when defined(amd64):
    x86_add_u32v128(a, b)
  elif defined(aarch64):
    arm_add_u32v128(a, b)
  elif defined(loongarch):
    la_add_u32v128(a, b)

# ========================
# add / u64v128 / value
# ========================

func add_u64v128_fallback(a, b: u64v128): u64v128 =
  var output: u64v128
  unroll(i, 0, 1):
    output[i] = a[i] + b[i]
  return output

func add_u64v128_sse(a, b: u64v128): u64v128 =
  return cast[u64v128](mm_add_epi64(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_add_u64v128: proc (a, b: u64v128): u64v128 = 
  if sseExists(): add_u64v128_sse 
  else: add_u64v128_fallback 

func add_u64v128_neon(a, b: u64v128): u64v128 =
  return cast[u64v128](vaddq_u64(cast[Uint64x2](a), cast[Uint64x2](b)))

let arm_add_u64v128: proc (a, b: u64v128): u64v128 = if neonExists(): add_u64v128_neon else: add_u64v128_fallback

func add_u64v128_lsx(a, b: u64v128): u64v128 =
  return cast[u64v128](lsx_vadd_d(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_add_u64v128: proc (a, b: u64v128): u64v128 = if lsxExists(): add_u64v128_lsx else: add_u64v128_fallback

template `+`(a, b: u64v128): u64v128 = 
  when defined(amd64):
    x86_add_u64v128(a, b)
  elif defined(aarch64):
    arm_add_u64v128(a, b)
  elif defined(loongarch):
    la_add_u64v128(a, b)

# ========================
# sub / f32v128 / value
# ========================

func sub_f32v128_fallback(a, b: f32v128): f32v128 =
  var output: f32v128
  unroll(i, 0, 3):
    output[i] = a[i] - b[i]
  return output

func sub_f32v128_sse(a, b: f32v128): f32v128 =
  return cast[f32v128](mm_sub_ps(cast[sse.M128](a), cast[sse.M128](b)))

func sub_f32v128_avx512(a, b: f32v128): f32v128 = 
  let mask: MMask8 = cast[MMask8](0x0F'u8)
  return cast[f32v128](mm_maskz_sub_ps(mask, cast[sse.M128](a), cast[sse.M128](b)))

let x86_sub_f32v128: proc (a, b: f32v128): f32v128 = 
  if avx512Exists(): sub_f32v128_avx512
  elif sseExists(): sub_f32v128_sse 
  else: sub_f32v128_fallback 

func sub_f32v128_neon(a, b: f32v128): f32v128 =
  return cast[f32v128](vsubq_f32(cast[Float32x4](a), cast[Float32x4](b)))

let arm_sub_f32v128: proc (a, b: f32v128): f32v128 = if neonExists(): sub_f32v128_neon else: sub_f32v128_fallback

func sub_f32v128_lsx(a, b: f32v128): f32v128 =
  return cast[f32v128](lsx_vfsub_s(cast[lsx.M128](a), cast[lsx.M128](b)))

let la_sub_f32v128: proc (a, b: f32v128): f32v128 = if lsxExists(): sub_f32v128_lsx else: sub_f32v128_fallback

template `-`(a, b: f32v128): f32v128 = 
  when defined(amd64):
    x86_sub_f32v128(a, b)
  elif defined(aarch64):
    arm_sub_f32v128(a, b)
  elif defined(loongarch):
    la_sub_f32v128(a, b)

# ========================
# sub / f64v128 / value
# ========================

func sub_f64v128_fallback(a, b: f64v128): f64v128 =
  var output: f64v128
  unroll(i, 0, 1):
    output[i] = a[i] - b[i]
  return output

func sub_f64v128_avx512(a, b: f64v128): f64v128 =
  let mask = cast[MMask8](0x03'u8)
  return cast[f64v128](mm_maskz_sub_pd(mask, cast[sse2.M128d](a), cast[sse2.M128d](b)))

func sub_f64v128_sse(a, b: f64v128): f64v128 =
  return cast[f64v128](mm_sub_pd(cast[sse2.M128d](a), cast[sse2.M128d](b)))

let x86_sub_f64v128: proc (a, b: f64v128): f64v128 = 
  if avx512Exists(): sub_f64v128_avx512
  elif sseExists(): sub_f64v128_sse 
  else: sub_f64v128_fallback 

func sub_f64v128_neon(a, b: f64v128): f64v128 =
  return cast[f64v128](vsubq_f64(cast[Float64x2](a), cast[Float64x2](b)))

let arm_sub_f64v128: proc (a, b: f64v128): f64v128 = if neonExists(): sub_f64v128_neon else: sub_f64v128_fallback

func sub_f64v128_lsx(a, b: f64v128): f64v128 =
  return cast[f64v128](lsx_vfsub_d(cast[lsx.M128d](a), cast[lsx.M128d](b)))

let la_sub_f64v128: proc (a, b: f64v128): f64v128 = if lsxExists(): sub_f64v128_lsx else: sub_f64v128_fallback

template `-`(a, b: f64v128): f64v128 = 
  when defined(amd64):
    x86_sub_f64v128(a, b)
  elif defined(aarch64):
    arm_sub_f64v128(a, b)
  elif defined(loongarch):
    la_sub_f64v128(a, b)

# ========================
# sub / i8v128 / value
# ========================

func sub_i8v128_fallback(a, b: i8v128): i8v128 =
  var output: i8v128
  unroll(i, 0, 15):
    output[i] = a[i] - b[i]
  return output

func sub_i8v128_sse(a, b: i8v128): i8v128 =
  return cast[i8v128](mm_sub_epi8(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_sub_i8v128: proc (a, b: i8v128): i8v128 = 
  if sseExists(): sub_i8v128_sse 
  else: sub_i8v128_fallback 

func sub_i8v128_neon(a, b: i8v128): i8v128 =
  return cast[i8v128](vsubq_s8(cast[Int8x16](a), cast[Int8x16](b)))

let arm_sub_i8v128: proc (a, b: i8v128): i8v128 = if neonExists(): sub_i8v128_neon else: sub_i8v128_fallback

func sub_i8v128_lsx(a, b: i8v128): i8v128 =
  return cast[i8v128](lsx_vsub_b(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_sub_i8v128: proc (a, b: i8v128): i8v128 = if lsxExists(): sub_i8v128_lsx else: sub_i8v128_fallback

template `-`(a, b: i8v128): i8v128 = 
  when defined(amd64):
    x86_sub_i8v128(a, b)
  elif defined(aarch64):
    arm_sub_i8v128(a, b)
  elif defined(loongarch):
    la_sub_i8v128(a, b)

# ========================
# sub / i16v128 / value
# ========================

func sub_i16v128_fallback(a, b: i16v128): i16v128 =
  var output: i16v128
  unroll(i, 0, 7):
    output[i] = a[i] - b[i]
  return output

func sub_i16v128_sse(a, b: i16v128): i16v128 =
  return cast[i16v128](mm_sub_epi16(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_sub_i16v128: proc (a, b: i16v128): i16v128 = 
  if sseExists(): sub_i16v128_sse 
  else: sub_i16v128_fallback 

func sub_i16v128_neon(a, b: i16v128): i16v128 =
  return cast[i16v128](vsubq_s16(cast[Int16x8](a), cast[Int16x8](b)))

let arm_sub_i16v128: proc (a, b: i16v128): i16v128 = if neonExists(): sub_i16v128_neon else: sub_i16v128_fallback

func sub_i16v128_lsx(a, b: i16v128): i16v128 =
  return cast[i16v128](lsx_vsub_h(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_sub_i16v128: proc (a, b: i16v128): i16v128 = if lsxExists(): sub_i16v128_lsx else: sub_i16v128_fallback

template `-`(a, b: i16v128): i16v128 = 
  when defined(amd64):
    x86_sub_i16v128(a, b)
  elif defined(aarch64):
    arm_sub_i16v128(a, b)
  elif defined(loongarch):
    la_sub_i16v128(a, b)

# ========================
# sub / i32v128 / value
# ========================

func sub_i32v128_fallback(a, b: i32v128): i32v128 =
  var output: i32v128
  unroll(i, 0, 3):
    output[i] = a[i] - b[i]
  return output

func sub_i32v128_sse(a, b: i32v128): i32v128 =
  return cast[i32v128](mm_sub_epi32(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_sub_i32v128: proc (a, b: i32v128): i32v128 = 
  if sseExists(): sub_i32v128_sse 
  else: sub_i32v128_fallback 

func sub_i32v128_neon(a, b: i32v128): i32v128 =
  return cast[i32v128](vsubq_s32(cast[Int32x4](a), cast[Int32x4](b)))

let arm_sub_i32v128: proc (a, b: i32v128): i32v128 = if neonExists(): sub_i32v128_neon else: sub_i32v128_fallback

func sub_i32v128_lsx(a, b: i32v128): i32v128 =
  return cast[i32v128](lsx_vsub_w(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_sub_i32v128: proc (a, b: i32v128): i32v128 = if lsxExists(): sub_i32v128_lsx else: sub_i32v128_fallback

template `-`(a, b: i32v128): i32v128 = 
  when defined(amd64):
    x86_sub_i32v128(a, b)
  elif defined(aarch64):
    arm_sub_i32v128(a, b)
  elif defined(loongarch):
    la_sub_i32v128(a, b)

# ========================
# sub / i64v128 / value
# ========================

func sub_i64v128_fallback(a, b: i64v128): i64v128 =
  var output: i64v128
  unroll(i, 0, 1):
    output[i] = a[i] - b[i]
  return output

func sub_i64v128_sse(a, b: i64v128): i64v128 =
  return cast[i64v128](mm_sub_epi64(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_sub_i64v128: proc (a, b: i64v128): i64v128 = 
  if sseExists(): sub_i64v128_sse 
  else: sub_i64v128_fallback 

func sub_i64v128_neon(a, b: i64v128): i64v128 =
  return cast[i64v128](vsubq_s64(cast[Int64x2](a), cast[Int64x2](b)))

let arm_sub_i64v128: proc (a, b: i64v128): i64v128 = if neonExists(): sub_i64v128_neon else: sub_i64v128_fallback

func sub_i64v128_lsx(a, b: i64v128): i64v128 =
  return cast[i64v128](lsx_vsub_d(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_sub_i64v128: proc (a, b: i64v128): i64v128 = if lsxExists(): sub_i64v128_lsx else: sub_i64v128_fallback

template `-`(a, b: i64v128): i64v128 = 
  when defined(amd64):
    x86_sub_i64v128(a, b)
  elif defined(aarch64):
    arm_sub_i64v128(a, b)
  elif defined(loongarch):
    la_sub_i64v128(a, b)

# ========================
# sub / u8v128 / value
# ========================

func sub_u8v128_fallback(a, b: u8v128): u8v128 =
  var output: u8v128
  unroll(i, 0, 15):
    output[i] = a[i] - b[i]
  return output

func sub_u8v128_sse(a, b: u8v128): u8v128 =
  return cast[u8v128](mm_sub_epi8(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_sub_u8v128: proc (a, b: u8v128): u8v128 = 
  if sseExists(): sub_u8v128_sse 
  else: sub_u8v128_fallback 

func sub_u8v128_neon(a, b: u8v128): u8v128 =
  return cast[u8v128](vsubq_u8(cast[Uint8x16](a), cast[Uint8x16](b)))

let arm_sub_u8v128: proc (a, b: u8v128): u8v128 = if neonExists(): sub_u8v128_neon else: sub_u8v128_fallback

func sub_u8v128_lsx(a, b: u8v128): u8v128 =
  return cast[u8v128](lsx_vsub_b(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_sub_u8v128: proc (a, b: u8v128): u8v128 = if lsxExists(): sub_u8v128_lsx else: sub_u8v128_fallback

template `-`(a, b: u8v128): u8v128 = 
  when defined(amd64):
    x86_sub_u8v128(a, b)
  elif defined(aarch64):
    arm_sub_u8v128(a, b)
  elif defined(loongarch):
    la_sub_u8v128(a, b)

# ========================
# sub / u16v128 / value
# ========================

func sub_u16v128_fallback(a, b: u16v128): u16v128 =
  var output: u16v128
  unroll(i, 0, 7):
    output[i] = a[i] - b[i]
  return output

func sub_u16v128_sse(a, b: u16v128): u16v128 =
  return cast[u16v128](mm_sub_epi16(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_sub_u16v128: proc (a, b: u16v128): u16v128 = 
  if sseExists(): sub_u16v128_sse 
  else: sub_u16v128_fallback 

func sub_u16v128_neon(a, b: u16v128): u16v128 =
  return cast[u16v128](vsubq_u16(cast[Uint16x8](a), cast[Uint16x8](b)))

let arm_sub_u16v128: proc (a, b: u16v128): u16v128 = if neonExists(): sub_u16v128_neon else: sub_u16v128_fallback

func sub_u16v128_lsx(a, b: u16v128): u16v128 =
  return cast[u16v128](lsx_vsub_h(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_sub_u16v128: proc (a, b: u16v128): u16v128 = if lsxExists(): sub_u16v128_lsx else: sub_u16v128_fallback

template `-`(a, b: u16v128): u16v128 = 
  when defined(amd64):
    x86_sub_u16v128(a, b)
  elif defined(aarch64):
    arm_sub_u16v128(a, b)
  elif defined(loongarch):
    la_sub_u16v128(a, b)

# ========================
# sub / u32v128 / value
# ========================

func sub_u32v128_fallback(a, b: u32v128): u32v128 =
  var output: u32v128
  unroll(i, 0, 3):
    output[i] = a[i] - b[i]
  return output

func sub_u32v128_sse(a, b: u32v128): u32v128 =
  return cast[u32v128](mm_sub_epi32(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_sub_u32v128: proc (a, b: u32v128): u32v128 = 
  if sseExists(): sub_u32v128_sse 
  else: sub_u32v128_fallback 

func sub_u32v128_neon(a, b: u32v128): u32v128 =
  return cast[u32v128](vsubq_u32(cast[Uint32x4](a), cast[Uint32x4](b)))

let arm_sub_u32v128: proc (a, b: u32v128): u32v128 = if neonExists(): sub_u32v128_neon else: sub_u32v128_fallback

func sub_u32v128_lsx(a, b: u32v128): u32v128 =
  return cast[u32v128](lsx_vsub_w(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_sub_u32v128: proc (a, b: u32v128): u32v128 = if lsxExists(): sub_u32v128_lsx else: sub_u32v128_fallback

template `-`(a, b: u32v128): u32v128 = 
  when defined(amd64):
    x86_sub_u32v128(a, b)
  elif defined(aarch64):
    arm_sub_u32v128(a, b)
  elif defined(loongarch):
    la_sub_u32v128(a, b)

# ========================
# sub / u64v128 / value
# ========================

func sub_u64v128_fallback(a, b: u64v128): u64v128 =
  var output: u64v128
  unroll(i, 0, 1):
    output[i] = a[i] - b[i]
  return output

func sub_u64v128_sse(a, b: u64v128): u64v128 =
  return cast[u64v128](mm_sub_epi64(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_sub_u64v128: proc (a, b: u64v128): u64v128 = 
  if sseExists(): sub_u64v128_sse 
  else: sub_u64v128_fallback 

func sub_u64v128_neon(a, b: u64v128): u64v128 =
  return cast[u64v128](vsubq_u64(cast[Uint64x2](a), cast[Uint64x2](b)))

let arm_sub_u64v128: proc (a, b: u64v128): u64v128 = if neonExists(): sub_u64v128_neon else: sub_u64v128_fallback

func sub_u64v128_lsx(a, b: u64v128): u64v128 =
  return cast[u64v128](lsx_vsub_d(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_sub_u64v128: proc (a, b: u64v128): u64v128 = if lsxExists(): sub_u64v128_lsx else: sub_u64v128_fallback

template `-`(a, b: u64v128): u64v128 = 
  when defined(amd64):
    x86_sub_u64v128(a, b)
  elif defined(aarch64):
    arm_sub_u64v128(a, b)
  elif defined(loongarch):
    la_sub_u64v128(a, b)

# ========================
# mul / f32v128 / value
# ========================

func mul_f32v128_fallback(a, b: f32v128): f32v128 =
  var output: f32v128
  unroll(i, 0, 3):
    output[i] = a[i] * b[i]
  return output

func mul_f32v128_sse(a, b: f32v128): f32v128 =
  return cast[f32v128](mm_mul_ps(cast[sse.M128](a), cast[sse.M128](b)))

func mul_f32v128_avx512(a, b: f32v128): f32v128 = 
  let mask: MMask8 = cast[MMask8](0x0F'u8)
  return cast[f32v128](mm_maskz_mul_ps(mask, cast[sse.M128](a), cast[sse.M128](b)))

let x86_mul_f32v128: proc (a, b: f32v128): f32v128 = 
  if avx512Exists(): mul_f32v128_avx512
  elif sseExists(): mul_f32v128_sse 
  else: mul_f32v128_fallback 

func mul_f32v128_neon(a, b: f32v128): f32v128 =
  return cast[f32v128](vmulq_f32(cast[Float32x4](a), cast[Float32x4](b)))

let arm_mul_f32v128: proc (a, b: f32v128): f32v128 = if neonExists(): mul_f32v128_neon else: mul_f32v128_fallback

func mul_f32v128_lsx(a, b: f32v128): f32v128 =
  return cast[f32v128](lsx_vfmul_s(cast[lsx.M128](a), cast[lsx.M128](b)))

let la_mul_f32v128: proc (a, b: f32v128): f32v128 = if lsxExists(): mul_f32v128_lsx else: mul_f32v128_fallback

template `*`(a, b: f32v128): f32v128 = 
  when defined(amd64):
    x86_mul_f32v128(a, b)
  elif defined(aarch64):
    arm_mul_f32v128(a, b)
  elif defined(loongarch):
    la_mul_f32v128(a, b)

# ========================
# mul / f64v128 / value
# ========================

func mul_f64v128_fallback(a, b: f64v128): f64v128 =
  var output: f64v128
  unroll(i, 0, 1):
    output[i] = a[i] * b[i]
  return output

func mul_f64v128_sse(a, b: f64v128): f64v128 =
  return cast[f64v128](mm_mul_pd(cast[sse2.M128d](a), cast[sse2.M128d](b)))

func mul_f64v128_avx512(a, b: f64v128): f64v128 =
  let mask = cast[MMask8](0x03'u8)
  return cast[f64v128](mm_maskz_mul_pd(mask, cast[sse2.M128d](a), cast[sse2.M128d](b)))

let x86_mul_f64v128: proc (a, b: f64v128): f64v128 = 
  if avx512Exists(): mul_f64v128_avx512
  elif sseExists(): mul_f64v128_sse 
  else: mul_f64v128_fallback 

func mul_f64v128_neon(a, b: f64v128): f64v128 =
  return cast[f64v128](vmulq_f64(cast[Float64x2](a), cast[Float64x2](b)))

let arm_mul_f64v128: proc (a, b: f64v128): f64v128 = if neonExists(): mul_f64v128_neon else: mul_f64v128_fallback

func mul_f64v128_lsx(a, b: f64v128): f64v128 =
  return cast[f64v128](lsx_vfmul_d(cast[lsx.M128d](a), cast[lsx.M128d](b)))

let la_mul_f64v128: proc (a, b: f64v128): f64v128 = if lsxExists(): mul_f64v128_lsx else: mul_f64v128_fallback

template `*`(a, b: f64v128): f64v128 = 
  when defined(amd64):
    x86_mul_f64v128(a, b)
  elif defined(aarch64):
    arm_mul_f64v128(a, b)
  elif defined(loongarch):
    la_mul_f64v128(a, b)

# ========================
# mul / i8v128 / value
# ========================

func mul_i8v128_fallback(a, b: i8v128): i8v128 =
  var output: i8v128
  unroll(i, 0, 15):
    output[i] = a[i] * b[i]
  return output

let x86_mul_i8v128: proc (a, b: i8v128): i8v128 = 
  mul_i8v128_fallback 

func mul_i8v128_neon(a, b: i8v128): i8v128 =
  return cast[i8v128](vmulq_s8(cast[Int8x16](a), cast[Int8x16](b)))

let arm_mul_i8v128: proc (a, b: i8v128): i8v128 = if neonExists(): mul_i8v128_neon else: mul_i8v128_fallback

func mul_i8v128_lsx(a, b: i8v128): i8v128 =
  return cast[i8v128](lsx_vmul_b(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_mul_i8v128: proc (a, b: i8v128): i8v128 = if lsxExists(): mul_i8v128_lsx else: mul_i8v128_fallback

template `*`(a, b: i8v128): i8v128 = 
  when defined(amd64):
    x86_mul_i8v128(a, b)
  elif defined(aarch64):
    arm_mul_i8v128(a, b)
  elif defined(loongarch):
    la_mul_i8v128(a, b)

# ========================
# mul / i16v128 / value
# ========================

func mul_i16v128_fallback(a, b: i16v128): i16v128 =
  var output: i16v128
  unroll(i, 0, 7):
    output[i] = a[i] * b[i]
  return output

func mul_i16v128_sse(a, b: i16v128): i16v128 =
  return cast[i16v128](mm_mullo_epi16(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_mul_i16v128: proc (a, b: i16v128): i16v128 = 
  if sseExists(): mul_i16v128_sse 
  else: mul_i16v128_fallback 

func mul_i16v128_neon(a, b: i16v128): i16v128 =
  return cast[i16v128](vmulq_s16(cast[Int16x8](a), cast[Int16x8](b)))

let arm_mul_i16v128: proc (a, b: i16v128): i16v128 = if neonExists(): mul_i16v128_neon else: mul_i16v128_fallback

func mul_i16v128_lsx(a, b: i16v128): i16v128 =
  return cast[i16v128](lsx_vmul_h(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_mul_i16v128: proc (a, b: i16v128): i16v128 = if lsxExists(): mul_i16v128_lsx else: mul_i16v128_fallback

template `*`(a, b: i16v128): i16v128 = 
  when defined(amd64):
    x86_mul_i16v128(a, b)
  elif defined(aarch64):
    arm_mul_i16v128(a, b)
  elif defined(loongarch):
    la_mul_i16v128(a, b)

# ========================
# mul / i32v128 / value
# ========================

func mul_i32v128_fallback(a, b: i32v128): i32v128 =
  var output: i32v128
  unroll(i, 0, 3):
    output[i] = a[i] * b[i]
  return output

func mul_i32v128_sse41(a, b: i32v128): i32v128 =
  return cast[i32v128](mm_mullo_epi32(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_mul_i32v128: proc (a, b: i32v128): i32v128 = 
  if sseExists(): mul_i32v128_sse41
  else: mul_i32v128_fallback 

func mul_i32v128_neon(a, b: i32v128): i32v128 =
  return cast[i32v128](vmulq_s32(cast[Int32x4](a), cast[Int32x4](b)))

let arm_mul_i32v128: proc (a, b: i32v128): i32v128 = if neonExists(): mul_i32v128_neon else: mul_i32v128_fallback

func mul_i32v128_lsx(a, b: i32v128): i32v128 =
  return cast[i32v128](lsx_vmul_w(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_mul_i32v128: proc (a, b: i32v128): i32v128 = if lsxExists(): mul_i32v128_lsx else: mul_i32v128_fallback

template `*`(a, b: i32v128): i32v128 = 
  when defined(amd64):
    x86_mul_i32v128(a, b)
  elif defined(aarch64):
    arm_mul_i32v128(a, b)
  elif defined(loongarch):
    la_mul_i32v128(a, b)

# ========================
# mul / i64v128 / value
# ========================

func mul_i64v128_fallback(a, b: i64v128): i64v128 =
  var output: i64v128
  unroll(i, 0, 1):
    output[i] = a[i] * b[i]
  return output

let x86_mul_i64v128: proc (a, b: i64v128): i64v128 = mul_i64v128_fallback

let arm_mul_i64v128: proc (a, b: i64v128): i64v128 =  mul_i64v128_fallback

func mul_i64v128_lsx(a, b: i64v128): i64v128 =
  return cast[i64v128](lsx_vmul_d(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_mul_i64v128: proc (a, b: i64v128): i64v128 = if lsxExists(): mul_i64v128_lsx else: mul_i64v128_fallback

template `*`(a, b: i64v128): i64v128 = 
  when defined(amd64):
    x86_mul_i64v128(a, b)
  elif defined(aarch64):
    arm_mul_i64v128(a, b)
  elif defined(loongarch):
    la_mul_i64v128(a, b)

# ========================
# mul / u8v128 / value
# ========================

func mul_u8v128_fallback(a, b: u8v128): u8v128 =
  var output: u8v128
  unroll(i, 0, 15):
    output[i] = a[i] * b[i]
  return output

let x86_mul_u8v128: proc (a, b: u8v128): u8v128 = 
  mul_u8v128_fallback

func mul_u8v128_neon(a, b: u8v128): u8v128 =
  return cast[u8v128](vmulq_u8(cast[Uint8x16](a), cast[Uint8x16](b)))

let arm_mul_u8v128: proc (a, b: u8v128): u8v128 = if neonExists(): mul_u8v128_neon else: mul_u8v128_fallback

func mul_u8v128_lsx(a, b: u8v128): u8v128 =
  return cast[u8v128](lsx_vmul_b(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_mul_u8v128: proc (a, b: u8v128): u8v128 = if lsxExists(): mul_u8v128_lsx else: mul_u8v128_fallback

template `*`(a, b: u8v128): u8v128 = 
  when defined(amd64):
    x86_mul_u8v128(a, b)
  elif defined(aarch64):
    arm_mul_u8v128(a, b)
  elif defined(loongarch):
    la_mul_u8v128(a, b)

# ========================
# mul / u16v128 / value
# ========================

func mul_u16v128_fallback(a, b: u16v128): u16v128 =
  var output: u16v128
  unroll(i, 0, 7):
    output[i] = a[i] * b[i]
  return output

func mul_u16v128_sse(a, b: u16v128): u16v128 =
  return cast[u16v128](mm_mullo_epi16(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_mul_u16v128: proc (a, b: u16v128): u16v128 = 
  if sseExists(): mul_u16v128_sse 
  else: mul_u16v128_fallback 

func mul_u16v128_neon(a, b: u16v128): u16v128 =
  return cast[u16v128](vmulq_u16(cast[Uint16x8](a), cast[Uint16x8](b)))

let arm_mul_u16v128: proc (a, b: u16v128): u16v128 = if neonExists(): mul_u16v128_neon else: mul_u16v128_fallback

func mul_u16v128_lsx(a, b: u16v128): u16v128 =
  return cast[u16v128](lsx_vmul_h(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_mul_u16v128: proc (a, b: u16v128): u16v128 = if lsxExists(): mul_u16v128_lsx else: mul_u16v128_fallback

template `*`(a, b: u16v128): u16v128 = 
  when defined(amd64):
    x86_mul_u16v128(a, b)
  elif defined(aarch64):
    arm_mul_u16v128(a, b)
  elif defined(loongarch):
    la_mul_u16v128(a, b)

# ========================
# mul / u32v128 / value
# ========================

func mul_u32v128_fallback(a, b: u32v128): u32v128 =
  var output: u32v128
  unroll(i, 0, 3):
    output[i] = a[i] * b[i]
  return output

func mul_u32v128_sse41(a, b: u32v128): u32v128 =
  return cast[u32v128](mm_mullo_epi32(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_mul_u32v128: proc (a, b: u32v128): u32v128 = 
  if sseExists(): mul_u32v128_sse41 
  else: mul_u32v128_fallback 

func mul_u32v128_neon(a, b: u32v128): u32v128 =
  return cast[u32v128](vmulq_u32(cast[Uint32x4](a), cast[Uint32x4](b)))

let arm_mul_u32v128: proc (a, b: u32v128): u32v128 = if neonExists(): mul_u32v128_neon else: mul_u32v128_fallback

func mul_u32v128_lsx(a, b: u32v128): u32v128 =
  return cast[u32v128](lsx_vmul_w(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_mul_u32v128: proc (a, b: u32v128): u32v128 = if lsxExists(): mul_u32v128_lsx else: mul_u32v128_fallback

template `*`(a, b: u32v128): u32v128 = 
  when defined(amd64):
    x86_mul_u32v128(a, b)
  elif defined(aarch64):
    arm_mul_u32v128(a, b)
  elif defined(loongarch):
    la_mul_u32v128(a, b)

# ========================
# mul / u64v128 / value
# ========================

func mul_u64v128_fallback(a, b: u64v128): u64v128 =
  var output: u64v128
  unroll(i, 0, 1):
    output[i] = a[i] * b[i]
  return output

let x86_mul_u64v128: proc (a, b: u64v128): u64v128 = mul_u64v128_fallback

let arm_mul_u64v128: proc (a, b: u64v128): u64v128 = mul_u64v128_fallback

func mul_u64v128_lsx(a, b: u64v128): u64v128 =
  return cast[u64v128](lsx_vmul_d(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_mul_u64v128: proc (a, b: u64v128): u64v128 = if lsxExists(): mul_u64v128_lsx else: mul_u64v128_fallback

template `*`(a, b: u64v128): u64v128 = 
  when defined(amd64):
    x86_mul_u64v128(a, b)
  elif defined(aarch64):
    arm_mul_u64v128(a, b)
  elif defined(loongarch):
    la_mul_u64v128(a, b)

# ========================
# and / f32v128 / value
# ========================

func and_f32v128_fallback(a, b: f32v128): f32v128 =
  var output: f32v128
  when CPUBits == 64:
    let p: ptr array[2, uint64] = cast[ptr array[2, uint64]](addr output[0])
    unroll(i, 0, 1):
      p[i] = cast[array[2, uint64]](a)[i] and cast[array[2, uint64]](b)[i]
  else:
    unroll(i, 0, 3):
      output[i] = float32(cast[uint32](a) and cast[uint32](b))
  return output

func and_f32v128_sse(a, b: f32v128): f32v128 =
  return cast[f32v128](mm_and_ps(cast[sse.M128](a), cast[sse.M128](b)))

let x86_and_f32v128: proc (a, b: f32v128): f32v128 = 
  if sseExists(): and_f32v128_sse 
  else: and_f32v128_fallback 

func and_f32v128_neon(a, b: f32v128): f32v128 =
  return cast[f32v128](vandq_u32(cast[Uint32x4](a), cast[Uint32x4](b)))

let arm_and_f32v128: proc (a, b: f32v128): f32v128 = if neonExists(): and_f32v128_neon else: and_f32v128_fallback

func and_f32v128_lsx(a, b: f32v128): f32v128 =
  return cast[f32v128](lsx_vand_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_and_f32v128: proc (a, b: f32v128): f32v128 = if lsxExists(): and_f32v128_lsx else: and_f32v128_fallback

template `and`(a, b: f32v128): f32v128 = 
  when defined(amd64): x86_and_f32v128(a, b)
  elif defined(aarch64): arm_and_f32v128(a, b)
  elif defined(loongarch): la_and_f32v128(a, b)

# ========================
# and / f64v128 / value
# ========================

func and_f64v128_fallback(a, b: f64v128): f64v128 =
  var output: f64v128
  when CPUBits == 64:
    unroll(i, 0, 1):
      output[i] = float64(cast[uint32](a[i]) and cast[uint32](b[i]))
  else:
    let p: ptr array[4, uint32] = cast[ptr array[4, uint32]](addr output[0])
    unroll(i, 0, 3):
      p[i] = cast[array[4, uint32]](a)[i] and cast[array[4, uint32]](b)[i]
  return output

func and_f64v128_sse(a, b: f64v128): f64v128 =
  return cast[f64v128](mm_and_pd(cast[sse2.M128d](a), cast[sse2.M128d](b)))

let x86_and_f64v128: proc (a, b: f64v128): f64v128 = 
  if sseExists(): and_f64v128_sse 
  else: and_f64v128_fallback 

func and_f64v128_neon(a, b: f64v128): f64v128 =
  return cast[f64v128](vandq_u64(cast[Uint64x2](a), cast[Uint64x2](b)))

let arm_and_f64v128: proc (a, b: f64v128): f64v128 = if neonExists(): and_f64v128_neon else: and_f64v128_fallback

func and_f64v128_lsx(a, b: f64v128): f64v128 =
  return cast[f64v128](lsx_vand_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_and_f64v128: proc (a, b: f64v128): f64v128 = if lsxExists(): and_f64v128_lsx else: and_f64v128_fallback

template `and`(a, b: f64v128): f64v128 = 
  when defined(amd64): x86_and_f64v128(a, b)
  elif defined(aarch64): arm_and_f64v128(a, b)
  elif defined(loongarch): la_and_f64v128(a, b)

# ========================
# and / i8v128 / value
# ========================

func and_i8v128_fallback(a, b: i8v128): i8v128 = 
  var output: i8v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] and pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] and pB[i]
  return output

func and_i8v128_sse(a, b: i8v128): i8v128 = 
  return cast[i8v128](mm_and_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_and_i8v128: proc (a, b: i8v128): i8v128 = 
  if sseExists(): and_i8v128_sse 
  else: and_i8v128_fallback 

func and_i8v128_neon(a, b: i8v128): i8v128 =
  return cast[i8v128](vandq_u8(cast[neon.Uint8x16](a), cast[neon.Uint8x16](b)))

let arm_and_i8v128: proc (a, b: i8v128): i8v128 = if neonExists(): and_i8v128_neon else: and_i8v128_fallback

func and_i8v128_lsx(a, b: i8v128): i8v128 =
  return cast[i8v128](lsx_vand_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_and_i8v128: proc (a, b: i8v128): i8v128 = if lsxExists(): and_i8v128_lsx else: and_i8v128_fallback

template `and`*(a, b: i8v128): i8v128 = 
  when defined(amd64) or defined(i386): x86_and_i8v128(a, b)
  elif defined(aarch64): arm_and_i8v128(a, b)
  elif defined(loongarch): la_and_i8v128(a, b)

# ========================
# and / i16v128 / value
# ========================

func and_i16v128_fallback(a, b: i16v128): i16v128 = 
  var output: i16v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] and pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] and pB[i]
  return output

func and_i16v128_sse(a, b: i16v128): i16v128 = 
  return cast[i16v128](mm_and_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_and_i16v128: proc (a, b: i16v128): i16v128 = 
  if sseExists(): and_i16v128_sse 
  else: and_i16v128_fallback 

func and_i16v128_neon(a, b: i16v128): i16v128 =
  return cast[i16v128](vandq_u16(cast[neon.Uint16x8](a), cast[neon.Uint16x8](b)))

let arm_and_i16v128: proc (a, b: i16v128): i16v128 = if neonExists(): and_i16v128_neon else: and_i16v128_fallback

func and_i16v128_lsx(a, b: i16v128): i16v128 =
  return cast[i16v128](lsx_vand_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_and_i16v128: proc (a, b: i16v128): i16v128 = if lsxExists(): and_i16v128_lsx else: and_i16v128_fallback

template `and`*(a, b: i16v128): i16v128 = 
  when defined(amd64) or defined(i386): x86_and_i16v128(a, b)
  elif defined(aarch64): arm_and_i16v128(a, b)
  elif defined(loongarch): la_and_i16v128(a, b)

# ========================
# and / i32v128 / value
# ========================

func and_i32v128_fallback(a, b: i32v128): i32v128 = 
  var output: i32v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] and pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] and pB[i]
  return output

func and_i32v128_sse(a, b: i32v128): i32v128 = 
  return cast[i32v128](mm_and_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_and_i32v128: proc (a, b: i32v128): i32v128 = 
  if sseExists(): and_i32v128_sse 
  else: and_i32v128_fallback 

func and_i32v128_neon(a, b: i32v128): i32v128 =
  return cast[i32v128](vandq_u32(cast[neon.Uint32x4](a), cast[neon.Uint32x4](b)))

let arm_and_i32v128: proc (a, b: i32v128): i32v128 = if neonExists(): and_i32v128_neon else: and_i32v128_fallback

func and_i32v128_lsx(a, b: i32v128): i32v128 =
  return cast[i32v128](lsx_vand_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_and_i32v128: proc (a, b: i32v128): i32v128 = if lsxExists(): and_i32v128_lsx else: and_i32v128_fallback

template `and`*(a, b: i32v128): i32v128 = 
  when defined(amd64) or defined(i386): x86_and_i32v128(a, b)
  elif defined(aarch64): arm_and_i32v128(a, b)
  elif defined(loongarch): la_and_i32v128(a, b)

# ========================
# and / i64v128 / value
# ========================

func and_i64v128_fallback(a, b: i64v128): i64v128 = 
  var output: i64v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] and pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] and pB[i]
  return output

func and_i64v128_sse(a, b: i64v128): i64v128 = 
  return cast[i64v128](mm_and_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_and_i64v128: proc (a, b: i64v128): i64v128 = 
  if sseExists(): and_i64v128_sse 
  else: and_i64v128_fallback 

func and_i64v128_neon(a, b: i64v128): i64v128 =
  return cast[i64v128](vandq_u64(cast[neon.Uint64x2](a), cast[neon.Uint64x2](b)))

let arm_and_i64v128: proc (a, b: i64v128): i64v128 = if neonExists(): and_i64v128_neon else: and_i64v128_fallback

func and_i64v128_lsx(a, b: i64v128): i64v128 =
  return cast[i64v128](lsx_vand_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_and_i64v128: proc (a, b: i64v128): i64v128 = if lsxExists(): and_i64v128_lsx else: and_i64v128_fallback

template `and`*(a, b: i64v128): i64v128 = 
  when defined(amd64) or defined(i386): x86_and_i64v128(a, b)
  elif defined(aarch64): arm_and_i64v128(a, b)
  elif defined(loongarch): la_and_i64v128(a, b)

# ========================
# and / u8v128 / value
# ========================

func and_u8v128_fallback(a, b: u8v128): u8v128 = 
  var output: u8v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] and pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] and pB[i]
  return output

func and_u8v128_sse(a, b: u8v128): u8v128 = 
  return cast[u8v128](mm_and_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_and_u8v128: proc (a, b: u8v128): u8v128 = 
  if sseExists(): and_u8v128_sse 
  else: and_u8v128_fallback 

func and_u8v128_neon(a, b: u8v128): u8v128 =
  return cast[u8v128](vandq_u8(cast[neon.Uint8x16](a), cast[neon.Uint8x16](b)))

let arm_and_u8v128: proc (a, b: u8v128): u8v128 = if neonExists(): and_u8v128_neon else: and_u8v128_fallback

func and_u8v128_lsx(a, b: u8v128): u8v128 =
  return cast[u8v128](lsx_vand_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_and_u8v128: proc (a, b: u8v128): u8v128 = if lsxExists(): and_u8v128_lsx else: and_u8v128_fallback

template `and`*(a, b: u8v128): u8v128 = 
  when defined(amd64) or defined(i386): x86_and_u8v128(a, b)
  elif defined(aarch64): arm_and_u8v128(a, b)
  elif defined(loongarch): la_and_u8v128(a, b)

# ========================
# and / u16v128 / value
# ========================

func and_u16v128_fallback(a, b: u16v128): u16v128 = 
  var output: u16v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] and pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] and pB[i]
  return output

func and_u16v128_sse(a, b: u16v128): u16v128 = 
  return cast[u16v128](mm_and_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_and_u16v128: proc (a, b: u16v128): u16v128 = 
  if sseExists(): and_u16v128_sse 
  else: and_u16v128_fallback 

func and_u16v128_neon(a, b: u16v128): u16v128 =
  return cast[u16v128](vandq_u16(cast[neon.Uint16x8](a), cast[neon.Uint16x8](b)))

let arm_and_u16v128: proc (a, b: u16v128): u16v128 = if neonExists(): and_u16v128_neon else: and_u16v128_fallback

func and_u16v128_lsx(a, b: u16v128): u16v128 =
  return cast[u16v128](lsx_vand_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_and_u16v128: proc (a, b: u16v128): u16v128 = if lsxExists(): and_u16v128_lsx else: and_u16v128_fallback

template `and`*(a, b: u16v128): u16v128 = 
  when defined(amd64) or defined(i386): x86_and_u16v128(a, b)
  elif defined(aarch64): arm_and_u16v128(a, b)
  elif defined(loongarch): la_and_u16v128(a, b)

# ========================
# and / u32v128 / value
# ========================

func and_u32v128_fallback(a, b: u32v128): u32v128 = 
  var output: u32v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] and pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] and pB[i]
  return output

func and_u32v128_sse(a, b: u32v128): u32v128 = 
  return cast[u32v128](mm_and_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_and_u32v128: proc (a, b: u32v128): u32v128 = 
  if sseExists(): and_u32v128_sse 
  else: and_u32v128_fallback 

func and_u32v128_neon(a, b: u32v128): u32v128 =
  return cast[u32v128](vandq_u32(cast[neon.Uint32x4](a), cast[neon.Uint32x4](b)))

let arm_and_u32v128: proc (a, b: u32v128): u32v128 = if neonExists(): and_u32v128_neon else: and_u32v128_fallback

func and_u32v128_lsx(a, b: u32v128): u32v128 =
  return cast[u32v128](lsx_vand_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_and_u32v128: proc (a, b: u32v128): u32v128 = if lsxExists(): and_u32v128_lsx else: and_u32v128_fallback

template `and`*(a, b: u32v128): u32v128 = 
  when defined(amd64) or defined(i386): x86_and_u32v128(a, b)
  elif defined(aarch64): arm_and_u32v128(a, b)
  elif defined(loongarch): la_and_u32v128(a, b)

# ========================
# and / u64v128 / value
# ========================

func and_u64v128_fallback(a, b: u64v128): u64v128 = 
  var output: u64v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] and pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] and pB[i]
  return output

func and_u64v128_sse(a, b: u64v128): u64v128 = 
  return cast[u64v128](mm_and_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_and_u64v128: proc (a, b: u64v128): u64v128 = 
  if sseExists(): and_u64v128_sse 
  else: and_u64v128_fallback 

func and_u64v128_neon(a, b: u64v128): u64v128 =
  return cast[u64v128](vandq_u64(cast[neon.Uint64x2](a), cast[neon.Uint64x2](b)))

let arm_and_u64v128: proc (a, b: u64v128): u64v128 = if neonExists(): and_u64v128_neon else: and_u64v128_fallback

func and_u64v128_lsx(a, b: u64v128): u64v128 =
  return cast[u64v128](lsx_vand_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_and_u64v128: proc (a, b: u64v128): u64v128 = if lsxExists(): and_u64v128_lsx else: and_u64v128_fallback

template `and`*(a, b: u64v128): u64v128 = 
  when defined(amd64) or defined(i386): x86_and_u64v128(a, b)
  elif defined(aarch64): arm_and_u64v128(a, b)
  elif defined(loongarch): la_and_u64v128(a, b)

# ========================
# or / f32v128 / value
# ========================

func or_f32v128_fallback(a, b: f32v128): f32v128 =
  var output: f32v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] or pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] or pB[i]
  return output

func or_f32v128_sse(a, b: f32v128): f32v128 =
  return cast[f32v128](mm_or_ps(cast[sse.M128](a), cast[sse.M128](b)))

let x86_or_f32v128: proc (a, b: f32v128): f32v128 = 
  if sseExists(): or_f32v128_sse 
  else: or_f32v128_fallback 

func or_f32v128_neon(a, b: f32v128): f32v128 =
  return cast[f32v128](vorrq_u32(cast[neon.Uint32x4](a), cast[neon.Uint32x4](b)))

let arm_or_f32v128: proc (a, b: f32v128): f32v128 = if neonExists(): or_f32v128_neon else: or_f32v128_fallback

func or_f32v128_lsx(a, b: f32v128): f32v128 =
  return cast[f32v128](lsx_vor_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_or_f32v128: proc (a, b: f32v128): f32v128 = if lsxExists(): or_f32v128_lsx else: or_f32v128_fallback

template `or`*(a, b: f32v128): f32v128 = 
  when defined(amd64) or defined(i386): x86_or_f32v128(a, b)
  elif defined(aarch64): arm_or_f32v128(a, b)
  elif defined(loongarch): la_or_f32v128(a, b)

# ========================
# or / f64v128 / value
# ========================

func or_f64v128_fallback(a, b: f64v128): f64v128 =
  var output: f64v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] or pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] or pB[i]
  return output

func or_f64v128_sse(a, b: f64v128): f64v128 =
  return cast[f64v128](mm_or_pd(cast[sse2.M128d](a), cast[sse2.M128d](b)))

let x86_or_f64v128: proc (a, b: f64v128): f64v128 = 
  if sseExists(): or_f64v128_sse 
  else: or_f64v128_fallback 

func or_f64v128_neon(a, b: f64v128): f64v128 =
  return cast[f64v128](vorrq_u64(cast[neon.Uint64x2](a), cast[neon.Uint64x2](b)))

let arm_or_f64v128: proc (a, b: f64v128): f64v128 = if neonExists(): or_f64v128_neon else: or_f64v128_fallback

func or_f64v128_lsx(a, b: f64v128): f64v128 =
  return cast[f64v128](lsx_vor_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_or_f64v128: proc (a, b: f64v128): f64v128 = if lsxExists(): or_f64v128_lsx else: or_f64v128_fallback

template `or`*(a, b: f64v128): f64v128 = 
  when defined(amd64) or defined(i386): x86_or_f64v128(a, b)
  elif defined(aarch64): arm_or_f64v128(a, b)
  elif defined(loongarch): la_or_f64v128(a, b)

# ========================
# or / i8v128 / value
# ========================

func or_i8v128_fallback(a, b: i8v128): i8v128 = 
  var output: i8v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] or pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] or pB[i]
  return output

func or_i8v128_sse(a, b: i8v128): i8v128 = 
  return cast[i8v128](mm_or_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_or_i8v128: proc (a, b: i8v128): i8v128 = 
  if sseExists(): or_i8v128_sse 
  else: or_i8v128_fallback 

func or_i8v128_neon(a, b: i8v128): i8v128 =
  return cast[i8v128](vorrq_u8(cast[neon.Uint8x16](a), cast[neon.Uint8x16](b)))

let arm_or_i8v128: proc (a, b: i8v128): i8v128 = if neonExists(): or_i8v128_neon else: or_i8v128_fallback

func or_i8v128_lsx(a, b: i8v128): i8v128 =
  return cast[i8v128](lsx_vor_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_or_i8v128: proc (a, b: i8v128): i8v128 = if lsxExists(): or_i8v128_lsx else: or_i8v128_fallback

template `or`*(a, b: i8v128): i8v128 = 
  when defined(amd64) or defined(i386): x86_or_i8v128(a, b)
  elif defined(aarch64): arm_or_i8v128(a, b)
  elif defined(loongarch): la_or_i8v128(a, b)

# ========================
# or / i16v128 / value
# ========================

func or_i16v128_fallback(a, b: i16v128): i16v128 = 
  var output: i16v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] or pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] or pB[i]
  return output

func or_i16v128_sse(a, b: i16v128): i16v128 = 
  return cast[i16v128](mm_or_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_or_i16v128: proc (a, b: i16v128): i16v128 = 
  if sseExists(): or_i16v128_sse 
  else: or_i16v128_fallback 

func or_i16v128_neon(a, b: i16v128): i16v128 =
  return cast[i16v128](vorrq_u16(cast[neon.Uint16x8](a), cast[neon.Uint16x8](b)))

let arm_or_i16v128: proc (a, b: i16v128): i16v128 = if neonExists(): or_i16v128_neon else: or_i16v128_fallback

func or_i16v128_lsx(a, b: i16v128): i16v128 =
  return cast[i16v128](lsx_vor_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_or_i16v128: proc (a, b: i16v128): i16v128 = if lsxExists(): or_i16v128_lsx else: or_i16v128_fallback

template `or`*(a, b: i16v128): i16v128 = 
  when defined(amd64) or defined(i386): x86_or_i16v128(a, b)
  elif defined(aarch64): arm_or_i16v128(a, b)
  elif defined(loongarch): la_or_i16v128(a, b)

# ========================
# or / i32v128 / value
# ========================

func or_i32v128_fallback(a, b: i32v128): i32v128 = 
  var output: i32v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] or pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] or pB[i]
  return output

func or_i32v128_sse(a, b: i32v128): i32v128 = 
  return cast[i32v128](mm_or_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_or_i32v128: proc (a, b: i32v128): i32v128 = 
  if sseExists(): or_i32v128_sse 
  else: or_i32v128_fallback 

func or_i32v128_neon(a, b: i32v128): i32v128 =
  return cast[i32v128](vorrq_u32(cast[neon.Uint32x4](a), cast[neon.Uint32x4](b)))

let arm_or_i32v128: proc (a, b: i32v128): i32v128 = if neonExists(): or_i32v128_neon else: or_i32v128_fallback

func or_i32v128_lsx(a, b: i32v128): i32v128 =
  return cast[i32v128](lsx_vor_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_or_i32v128: proc (a, b: i32v128): i32v128 = if lsxExists(): or_i32v128_lsx else: or_i32v128_fallback

template `or`*(a, b: i32v128): i32v128 = 
  when defined(amd64) or defined(i386): x86_or_i32v128(a, b)
  elif defined(aarch64): arm_or_i32v128(a, b)
  elif defined(loongarch): la_or_i32v128(a, b)

# ========================
# or / i64v128 / value
# ========================

func or_i64v128_fallback(a, b: i64v128): i64v128 = 
  var output: i64v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] or pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] or pB[i]
  return output

func or_i64v128_sse(a, b: i64v128): i64v128 = 
  return cast[i64v128](mm_or_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_or_i64v128: proc (a, b: i64v128): i64v128 = 
  if sseExists(): or_i64v128_sse 
  else: or_i64v128_fallback 

func or_i64v128_neon(a, b: i64v128): i64v128 =
  return cast[i64v128](vorrq_u64(cast[neon.Uint64x2](a), cast[neon.Uint64x2](b)))

let arm_or_i64v128: proc (a, b: i64v128): i64v128 = if neonExists(): or_i64v128_neon else: or_i64v128_fallback

func or_i64v128_lsx(a, b: i64v128): i64v128 =
  return cast[i64v128](lsx_vor_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_or_i64v128: proc (a, b: i64v128): i64v128 = if lsxExists(): or_i64v128_lsx else: or_i64v128_fallback

template `or`*(a, b: i64v128): i64v128 = 
  when defined(amd64) or defined(i386): x86_or_i64v128(a, b)
  elif defined(aarch64): arm_or_i64v128(a, b)
  elif defined(loongarch): la_or_i64v128(a, b)

# ========================
# or / u8v128 / value
# ========================

func or_u8v128_fallback(a, b: u8v128): u8v128 = 
  var output: u8v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] or pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] or pB[i]
  return output

func or_u8v128_sse(a, b: u8v128): u8v128 = 
  return cast[u8v128](mm_or_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_or_u8v128: proc (a, b: u8v128): u8v128 = 
  if sseExists(): or_u8v128_sse 
  else: or_u8v128_fallback 

func or_u8v128_neon(a, b: u8v128): u8v128 =
  return cast[u8v128](vorrq_u8(cast[neon.Uint8x16](a), cast[neon.Uint8x16](b)))

let arm_or_u8v128: proc (a, b: u8v128): u8v128 = if neonExists(): or_u8v128_neon else: or_u8v128_fallback

func or_u8v128_lsx(a, b: u8v128): u8v128 =
  return cast[u8v128](lsx_vor_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_or_u8v128: proc (a, b: u8v128): u8v128 = if lsxExists(): or_u8v128_lsx else: or_u8v128_fallback

template `or`*(a, b: u8v128): u8v128 = 
  when defined(amd64) or defined(i386): x86_or_u8v128(a, b)
  elif defined(aarch64): arm_or_u8v128(a, b)
  elif defined(loongarch): la_or_u8v128(a, b)

# ========================
# or / u16v128 / value
# ========================

func or_u16v128_fallback(a, b: u16v128): u16v128 = 
  var output: u16v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] or pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] or pB[i]
  return output

func or_u16v128_sse(a, b: u16v128): u16v128 = 
  return cast[u16v128](mm_or_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_or_u16v128: proc (a, b: u16v128): u16v128 = 
  if sseExists(): or_u16v128_sse 
  else: or_u16v128_fallback 

func or_u16v128_neon(a, b: u16v128): u16v128 =
  return cast[u16v128](vorrq_u16(cast[neon.Uint16x8](a), cast[neon.Uint16x8](b)))

let arm_or_u16v128: proc (a, b: u16v128): u16v128 = if neonExists(): or_u16v128_neon else: or_u16v128_fallback

func or_u16v128_lsx(a, b: u16v128): u16v128 =
  return cast[u16v128](lsx_vor_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_or_u16v128: proc (a, b: u16v128): u16v128 = if lsxExists(): or_u16v128_lsx else: or_u16v128_fallback

template `or`*(a, b: u16v128): u16v128 = 
  when defined(amd64) or defined(i386): x86_or_u16v128(a, b)
  elif defined(aarch64): arm_or_u16v128(a, b)
  elif defined(loongarch): la_or_u16v128(a, b)

# ========================
# or / u32v128 / value
# ========================

func or_u32v128_fallback(a, b: u32v128): u32v128 = 
  var output: u32v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] or pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] or pB[i]
  return output

func or_u32v128_sse(a, b: u32v128): u32v128 = 
  return cast[u32v128](mm_or_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_or_u32v128: proc (a, b: u32v128): u32v128 = 
  if sseExists(): or_u32v128_sse 
  else: or_u32v128_fallback 

func or_u32v128_neon(a, b: u32v128): u32v128 =
  return cast[u32v128](vorrq_u32(cast[neon.Uint32x4](a), cast[neon.Uint32x4](b)))

let arm_or_u32v128: proc (a, b: u32v128): u32v128 = if neonExists(): or_u32v128_neon else: or_u32v128_fallback

func or_u32v128_lsx(a, b: u32v128): u32v128 =
  return cast[u32v128](lsx_vor_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_or_u32v128: proc (a, b: u32v128): u32v128 = if lsxExists(): or_u32v128_lsx else: or_u32v128_fallback

template `or`*(a, b: u32v128): u32v128 = 
  when defined(amd64) or defined(i386): x86_or_u32v128(a, b)
  elif defined(aarch64): arm_or_u32v128(a, b)
  elif defined(loongarch): la_or_u32v128(a, b)

# ========================
# or / u64v128 / value
# ========================

func or_u64v128_fallback(a, b: u64v128): u64v128 = 
  var output: u64v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] or pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] or pB[i]
  return output

func or_u64v128_sse(a, b: u64v128): u64v128 = 
  return cast[u64v128](mm_or_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_or_u64v128: proc (a, b: u64v128): u64v128 = 
  if sseExists(): or_u64v128_sse 
  else: or_u64v128_fallback 

func or_u64v128_neon(a, b: u64v128): u64v128 =
  return cast[u64v128](vorrq_u64(cast[neon.Uint64x2](a), cast[neon.Uint64x2](b)))

let arm_or_u64v128: proc (a, b: u64v128): u64v128 = if neonExists(): or_u64v128_neon else: or_u64v128_fallback

func or_u64v128_lsx(a, b: u64v128): u64v128 =
  return cast[u64v128](lsx_vor_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_or_u64v128: proc (a, b: u64v128): u64v128 = if lsxExists(): or_u64v128_lsx else: or_u64v128_fallback

template `or`*(a, b: u64v128): u64v128 = 
  when defined(amd64) or defined(i386): x86_or_u64v128(a, b)
  elif defined(aarch64): arm_or_u64v128(a, b)
  elif defined(loongarch): la_or_u64v128(a, b)

# ========================
# xor / f32v128 / value
# ========================

func xor_f32v128_fallback(a, b: f32v128): f32v128 =
  var output: f32v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] xor pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] xor pB[i]
  return output

func xor_f32v128_sse(a, b: f32v128): f32v128 =
  return cast[f32v128](mm_xor_ps(cast[sse.M128](a), cast[sse.M128](b)))

let x86_xor_f32v128: proc (a, b: f32v128): f32v128 = 
  if sseExists(): xor_f32v128_sse 
  else: xor_f32v128_fallback 

func xor_f32v128_neon(a, b: f32v128): f32v128 =
  return cast[f32v128](veorq_u32(cast[neon.Uint32x4](a), cast[neon.Uint32x4](b)))

let arm_xor_f32v128: proc (a, b: f32v128): f32v128 = if neonExists(): xor_f32v128_neon else: xor_f32v128_fallback

func xor_f32v128_lsx(a, b: f32v128): f32v128 =
  return cast[f32v128](lsx_vxor_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_xor_f32v128: proc (a, b: f32v128): f32v128 = if lsxExists(): xor_f32v128_lsx else: xor_f32v128_fallback

template `xor`*(a, b: f32v128): f32v128 = 
  when defined(amd64) or defined(i386): x86_xor_f32v128(a, b)
  elif defined(aarch64): arm_xor_f32v128(a, b)
  elif defined(loongarch): la_xor_f32v128(a, b)

# ========================
# xor / f64v128 / value
# ========================

func xor_f64v128_fallback(a, b: f64v128): f64v128 =
  var output: f64v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] xor pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] xor pB[i]
  return output

func xor_f64v128_sse(a, b: f64v128): f64v128 =
  return cast[f64v128](mm_xor_pd(cast[sse2.M128d](a), cast[sse2.M128d](b)))

let x86_xor_f64v128: proc (a, b: f64v128): f64v128 = 
  if sseExists(): xor_f64v128_sse 
  else: xor_f64v128_fallback 

func xor_f64v128_neon(a, b: f64v128): f64v128 =
  return cast[f64v128](veorq_u64(cast[neon.Uint64x2](a), cast[neon.Uint64x2](b)))

let arm_xor_f64v128: proc (a, b: f64v128): f64v128 = if neonExists(): xor_f64v128_neon else: xor_f64v128_fallback

func xor_f64v128_lsx(a, b: f64v128): f64v128 =
  return cast[f64v128](lsx_vxor_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_xor_f64v128: proc (a, b: f64v128): f64v128 = if lsxExists(): xor_f64v128_lsx else: xor_f64v128_fallback

template `xor`*(a, b: f64v128): f64v128 = 
  when defined(amd64) or defined(i386): x86_xor_f64v128(a, b)
  elif defined(aarch64): arm_xor_f64v128(a, b)
  elif defined(loongarch): la_xor_f64v128(a, b)

# ========================
# xor / i8v128 / value
# ========================

func xor_i8v128_fallback(a, b: i8v128): i8v128 = 
  var output: i8v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] xor pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] xor pB[i]
  return output

func xor_i8v128_sse(a, b: i8v128): i8v128 = 
  return cast[i8v128](mm_xor_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_xor_i8v128: proc (a, b: i8v128): i8v128 = 
  if sseExists(): xor_i8v128_sse 
  else: xor_i8v128_fallback 

func xor_i8v128_neon(a, b: i8v128): i8v128 =
  return cast[i8v128](veorq_u8(cast[neon.Uint8x16](a), cast[neon.Uint8x16](b)))

let arm_xor_i8v128: proc (a, b: i8v128): i8v128 = if neonExists(): xor_i8v128_neon else: xor_i8v128_fallback

func xor_i8v128_lsx(a, b: i8v128): i8v128 =
  return cast[i8v128](lsx_vxor_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_xor_i8v128: proc (a, b: i8v128): i8v128 = if lsxExists(): xor_i8v128_lsx else: xor_i8v128_fallback

template `xor`*(a, b: i8v128): i8v128 = 
  when defined(amd64) or defined(i386): x86_xor_i8v128(a, b)
  elif defined(aarch64): arm_xor_i8v128(a, b)
  elif defined(loongarch): la_xor_i8v128(a, b)

# ========================
# xor / i16v128 / value
# ========================

func xor_i16v128_fallback(a, b: i16v128): i16v128 = 
  var output: i16v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] xor pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] xor pB[i]
  return output

func xor_i16v128_sse(a, b: i16v128): i16v128 = 
  return cast[i16v128](mm_xor_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_xor_i16v128: proc (a, b: i16v128): i16v128 = 
  if sseExists(): xor_i16v128_sse 
  else: xor_i16v128_fallback 

func xor_i16v128_neon(a, b: i16v128): i16v128 =
  return cast[i16v128](veorq_u16(cast[neon.Uint16x8](a), cast[neon.Uint16x8](b)))

let arm_xor_i16v128: proc (a, b: i16v128): i16v128 = if neonExists(): xor_i16v128_neon else: xor_i16v128_fallback

func xor_i16v128_lsx(a, b: i16v128): i16v128 =
  return cast[i16v128](lsx_vxor_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_xor_i16v128: proc (a, b: i16v128): i16v128 = if lsxExists(): xor_i16v128_lsx else: xor_i16v128_fallback

template `xor`*(a, b: i16v128): i16v128 = 
  when defined(amd64) or defined(i386): x86_xor_i16v128(a, b)
  elif defined(aarch64): arm_xor_i16v128(a, b)
  elif defined(loongarch): la_xor_i16v128(a, b)

# ========================
# xor / i32v128 / value
# ========================

func xor_i32v128_fallback(a, b: i32v128): i32v128 = 
  var output: i32v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] xor pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] xor pB[i]
  return output

func xor_i32v128_sse(a, b: i32v128): i32v128 = 
  return cast[i32v128](mm_xor_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_xor_i32v128: proc (a, b: i32v128): i32v128 = 
  if sseExists(): xor_i32v128_sse 
  else: xor_i32v128_fallback 

func xor_i32v128_neon(a, b: i32v128): i32v128 =
  return cast[i32v128](veorq_u32(cast[neon.Uint32x4](a), cast[neon.Uint32x4](b)))

let arm_xor_i32v128: proc (a, b: i32v128): i32v128 = if neonExists(): xor_i32v128_neon else: xor_i32v128_fallback

func xor_i32v128_lsx(a, b: i32v128): i32v128 =
  return cast[i32v128](lsx_vxor_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_xor_i32v128: proc (a, b: i32v128): i32v128 = if lsxExists(): xor_i32v128_lsx else: xor_i32v128_fallback

template `xor`*(a, b: i32v128): i32v128 = 
  when defined(amd64) or defined(i386): x86_xor_i32v128(a, b)
  elif defined(aarch64): arm_xor_i32v128(a, b)
  elif defined(loongarch): la_xor_i32v128(a, b)

# ========================
# xor / i64v128 / value
# ========================

func xor_i64v128_fallback(a, b: i64v128): i64v128 = 
  var output: i64v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] xor pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] xor pB[i]
  return output

func xor_i64v128_sse(a, b: i64v128): i64v128 = 
  return cast[i64v128](mm_xor_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_xor_i64v128: proc (a, b: i64v128): i64v128 = 
  if sseExists(): xor_i64v128_sse 
  else: xor_i64v128_fallback 

func xor_i64v128_neon(a, b: i64v128): i64v128 =
  return cast[i64v128](veorq_u64(cast[neon.Uint64x2](a), cast[neon.Uint64x2](b)))

let arm_xor_i64v128: proc (a, b: i64v128): i64v128 = if neonExists(): xor_i64v128_neon else: xor_i64v128_fallback

func xor_i64v128_lsx(a, b: i64v128): i64v128 =
  return cast[i64v128](lsx_vxor_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_xor_i64v128: proc (a, b: i64v128): i64v128 = if lsxExists(): xor_i64v128_lsx else: xor_i64v128_fallback

template `xor`*(a, b: i64v128): i64v128 = 
  when defined(amd64) or defined(i386): x86_xor_i64v128(a, b)
  elif defined(aarch64): arm_xor_i64v128(a, b)
  elif defined(loongarch): la_xor_i64v128(a, b)

# ========================
# xor / u8v128 / value
# ========================

func xor_u8v128_fallback(a, b: u8v128): u8v128 = 
  var output: u8v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] xor pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] xor pB[i]
  return output

func xor_u8v128_sse(a, b: u8v128): u8v128 = 
  return cast[u8v128](mm_xor_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_xor_u8v128: proc (a, b: u8v128): u8v128 = 
  if sseExists(): xor_u8v128_sse 
  else: xor_u8v128_fallback 

func xor_u8v128_neon(a, b: u8v128): u8v128 =
  return cast[u8v128](veorq_u8(cast[neon.Uint8x16](a), cast[neon.Uint8x16](b)))

let arm_xor_u8v128: proc (a, b: u8v128): u8v128 = if neonExists(): xor_u8v128_neon else: xor_u8v128_fallback

func xor_u8v128_lsx(a, b: u8v128): u8v128 =
  return cast[u8v128](lsx_vxor_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_xor_u8v128: proc (a, b: u8v128): u8v128 = if lsxExists(): xor_u8v128_lsx else: xor_u8v128_fallback

template `xor`*(a, b: u8v128): u8v128 = 
  when defined(amd64) or defined(i386): x86_xor_u8v128(a, b)
  elif defined(aarch64): arm_xor_u8v128(a, b)
  elif defined(loongarch): la_xor_u8v128(a, b)

# ========================
# xor / u16v128 / value
# ========================

func xor_u16v128_fallback(a, b: u16v128): u16v128 = 
  var output: u16v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] xor pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] xor pB[i]
  return output

func xor_u16v128_sse(a, b: u16v128): u16v128 = 
  return cast[u16v128](mm_xor_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_xor_u16v128: proc (a, b: u16v128): u16v128 = 
  if sseExists(): xor_u16v128_sse 
  else: xor_u16v128_fallback 

func xor_u16v128_neon(a, b: u16v128): u16v128 =
  return cast[u16v128](veorq_u16(cast[neon.Uint16x8](a), cast[neon.Uint16x8](b)))

let arm_xor_u16v128: proc (a, b: u16v128): u16v128 = if neonExists(): xor_u16v128_neon else: xor_u16v128_fallback

func xor_u16v128_lsx(a, b: u16v128): u16v128 =
  return cast[u16v128](lsx_vxor_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_xor_u16v128: proc (a, b: u16v128): u16v128 = if lsxExists(): xor_u16v128_lsx else: xor_u16v128_fallback

template `xor`*(a, b: u16v128): u16v128 = 
  when defined(amd64) or defined(i386): x86_xor_u16v128(a, b)
  elif defined(aarch64): arm_xor_u16v128(a, b)
  elif defined(loongarch): la_xor_u16v128(a, b)

# ========================
# xor / u32v128 / value
# ========================

func xor_u32v128_fallback(a, b: u32v128): u32v128 = 
  var output: u32v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] xor pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] xor pB[i]
  return output

func xor_u32v128_sse(a, b: u32v128): u32v128 = 
  return cast[u32v128](mm_xor_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_xor_u32v128: proc (a, b: u32v128): u32v128 = 
  if sseExists(): xor_u32v128_sse 
  else: xor_u32v128_fallback 

func xor_u32v128_neon(a, b: u32v128): u32v128 =
  return cast[u32v128](veorq_u32(cast[neon.Uint32x4](a), cast[neon.Uint32x4](b)))

let arm_xor_u32v128: proc (a, b: u32v128): u32v128 = if neonExists(): xor_u32v128_neon else: xor_u32v128_fallback

func xor_u32v128_lsx(a, b: u32v128): u32v128 =
  return cast[u32v128](lsx_vxor_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_xor_u32v128: proc (a, b: u32v128): u32v128 = if lsxExists(): xor_u32v128_lsx else: xor_u32v128_fallback

template `xor`*(a, b: u32v128): u32v128 = 
  when defined(amd64) or defined(i386): x86_xor_u32v128(a, b)
  elif defined(aarch64): arm_xor_u32v128(a, b)
  elif defined(loongarch): la_xor_u32v128(a, b)

# ========================
# xor / u64v128 / value
# ========================

func xor_u64v128_fallback(a, b: u64v128): u64v128 = 
  var output: u64v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    let pB   = cast[ptr array[2, uint64]](addr b)
    unroll(i, 0, 1):
      pOut[i] = pA[i] xor pB[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    let pB   = cast[ptr array[4, uint32]](addr b)
    unroll(i, 0, 3):
      pOut[i] = pA[i] xor pB[i]
  return output

func xor_u64v128_sse(a, b: u64v128): u64v128 = 
  return cast[u64v128](mm_xor_si128(cast[sse2.M128i](a), cast[sse2.M128i](b)))

let x86_xor_u64v128: proc (a, b: u64v128): u64v128 = 
  if sseExists(): xor_u64v128_sse 
  else: xor_u64v128_fallback 

func xor_u64v128_neon(a, b: u64v128): u64v128 =
  return cast[u64v128](veorq_u64(cast[neon.Uint64x2](a), cast[neon.Uint64x2](b)))

let arm_xor_u64v128: proc (a, b: u64v128): u64v128 = if neonExists(): xor_u64v128_neon else: xor_u64v128_fallback

func xor_u64v128_lsx(a, b: u64v128): u64v128 =
  return cast[u64v128](lsx_vxor_v(cast[lsx.M128i](a), cast[lsx.M128i](b)))

let la_xor_u64v128: proc (a, b: u64v128): u64v128 = if lsxExists(): xor_u64v128_lsx else: xor_u64v128_fallback

template `xor`*(a, b: u64v128): u64v128 = 
  when defined(amd64) or defined(i386): x86_xor_u64v128(a, b)
  elif defined(aarch64): arm_xor_u64v128(a, b)
  elif defined(loongarch): la_xor_u64v128(a, b)

# ========================
# not / f32v128 / value
# ========================

func not_f32v128_fallback(a: f32v128): f32v128 =
  var output: f32v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    unroll(i, 0, 1):
      pOut[i] = not pA[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    unroll(i, 0, 3):
      pOut[i] = not pA[i]
  return output

func not_f32v128_sse(a: f32v128): f32v128 =
  let allOnes = mm_set1_epi32(-1)
  return cast[f32v128](mm_xor_ps(cast[sse.M128](a), cast[sse.M128](allOnes)))

let x86_not_f32v128: proc (a: f32v128): f32v128 = 
  if sseExists(): not_f32v128_sse 
  else: not_f32v128_fallback 

func not_f32v128_neon(a: f32v128): f32v128 =
  return cast[f32v128](vmvnq_u32(cast[neon.Uint32x4](a)))

let arm_not_f32v128: proc (a: f32v128): f32v128 = if neonExists(): not_f32v128_neon else: not_f32v128_fallback

func not_f32v128_lsx(a: f32v128): f32v128 =
  return cast[f32v128](lsx_vnor_v(cast[lsx.M128i](a), cast[lsx.M128i](a))) # A NOR A == NOT A

let la_not_f32v128: proc (a: f32v128): f32v128 = if lsxExists(): not_f32v128_lsx else: not_f32v128_fallback

template `not`*(a: f32v128): f32v128 = 
  when defined(amd64) or defined(i386): x86_not_f32v128(a)
  elif defined(aarch64): arm_not_f32v128(a)
  elif defined(loongarch): la_not_f32v128(a)

# ========================
# not / f64v128 / value
# ========================

func not_f64v128_fallback(a: f64v128): f64v128 =
  var output: f64v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    unroll(i, 0, 1):
      pOut[i] = not pA[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    unroll(i, 0, 3):
      pOut[i] = not pA[i]
  return output

func not_f64v128_sse(a: f64v128): f64v128 =
  let allOnes = mm_set1_epi32(-1)
  return cast[f64v128](mm_xor_pd(cast[sse2.M128d](a), cast[sse2.M128d](allOnes)))

let x86_not_f64v128: proc (a: f64v128): f64v128 = 
  if sseExists(): not_f64v128_sse 
  else: not_f64v128_fallback 

func not_f64v128_neon(a: f64v128): f64v128 =
  return cast[f64v128](vmvnq_u32(cast[neon.Uint32x4](a)))

let arm_not_f64v128: proc (a: f64v128): f64v128 = if neonExists(): not_f64v128_neon else: not_f64v128_fallback

func not_f64v128_lsx(a: f64v128): f64v128 =
  return cast[f64v128](lsx_vnor_v(cast[lsx.M128i](a), cast[lsx.M128i](a)))

let la_not_f64v128: proc (a: f64v128): f64v128 = if lsxExists(): not_f64v128_lsx else: not_f64v128_fallback

template `not`*(a: f64v128): f64v128 = 
  when defined(amd64) or defined(i386): x86_not_f64v128(a)
  elif defined(aarch64): arm_not_f64v128(a)
  elif defined(loongarch): la_not_f64v128(a)

# ========================
# not / i8v128 / value
# ========================

func not_i8v128_fallback(a: i8v128): i8v128 = 
  var output: i8v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    unroll(i, 0, 1):
      pOut[i] = not pA[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    unroll(i, 0, 3):
      pOut[i] = not pA[i]
  return output

func not_i8v128_sse(a: i8v128): i8v128 = 
  let allOnes = mm_set1_epi32(-1)
  return cast[i8v128](mm_xor_si128(cast[sse2.M128i](a), cast[sse2.M128i](allOnes)))

let x86_not_i8v128: proc (a: i8v128): i8v128 = 
  if sseExists(): not_i8v128_sse 
  else: not_i8v128_fallback 

func not_i8v128_neon(a: i8v128): i8v128 =
  return cast[i8v128](vmvnq_u8(cast[neon.Uint8x16](a)))

let arm_not_i8v128: proc (a: i8v128): i8v128 = if neonExists(): not_i8v128_neon else: not_i8v128_fallback

func not_i8v128_lsx(a: i8v128): i8v128 =
  return cast[i8v128](lsx_vnor_v(cast[lsx.M128i](a), cast[lsx.M128i](a)))

let la_not_i8v128: proc (a: i8v128): i8v128 = if lsxExists(): not_i8v128_lsx else: not_i8v128_fallback

template `not`*(a: i8v128): i8v128 = 
  when defined(amd64) or defined(i386): x86_not_i8v128(a)
  elif defined(aarch64): arm_not_i8v128(a)
  elif defined(loongarch): la_not_i8v128(a)

# ========================
# not / i16v128 / value
# ========================

func not_i16v128_fallback(a: i16v128): i16v128 = 
  var output: i16v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    unroll(i, 0, 1):
      pOut[i] = not pA[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    unroll(i, 0, 3):
      pOut[i] = not pA[i]
  return output

func not_i16v128_sse(a: i16v128): i16v128 = 
  let allOnes = mm_set1_epi32(-1)
  return cast[i16v128](mm_xor_si128(cast[sse2.M128i](a), cast[sse2.M128i](allOnes)))

let x86_not_i16v128: proc (a: i16v128): i16v128 = 
  if sseExists(): not_i16v128_sse 
  else: not_i16v128_fallback 

func not_i16v128_neon(a: i16v128): i16v128 =
  return cast[i16v128](vmvnq_u16(cast[neon.Uint16x8](a)))

let arm_not_i16v128: proc (a: i16v128): i16v128 = if neonExists(): not_i16v128_neon else: not_i16v128_fallback

func not_i16v128_lsx(a: i16v128): i16v128 =
  return cast[i16v128](lsx_vnor_v(cast[lsx.M128i](a), cast[lsx.M128i](a)))

let la_not_i16v128: proc (a: i16v128): i16v128 = if lsxExists(): not_i16v128_lsx else: not_i16v128_fallback

template `not`*(a: i16v128): i16v128 = 
  when defined(amd64) or defined(i386): x86_not_i16v128(a)
  elif defined(aarch64): arm_not_i16v128(a)
  elif defined(loongarch): la_not_i16v128(a)

# ========================
# not / i32v128 / value
# ========================

func not_i32v128_fallback(a: i32v128): i32v128 = 
  var output: i32v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    unroll(i, 0, 1):
      pOut[i] = not pA[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    unroll(i, 0, 3):
      pOut[i] = not pA[i]
  return output

func not_i32v128_sse(a: i32v128): i32v128 = 
  let allOnes = mm_set1_epi32(-1)
  return cast[i32v128](mm_xor_si128(cast[sse2.M128i](a), cast[sse2.M128i](allOnes)))

let x86_not_i32v128: proc (a: i32v128): i32v128 = 
  if sseExists(): not_i32v128_sse 
  else: not_i32v128_fallback 

func not_i32v128_neon(a: i32v128): i32v128 =
  return cast[i32v128](vmvnq_u32(cast[neon.Uint32x4](a)))

let arm_not_i32v128: proc (a: i32v128): i32v128 = if neonExists(): not_i32v128_neon else: not_i32v128_fallback

func not_i32v128_lsx(a: i32v128): i32v128 =
  return cast[i32v128](lsx_vnor_v(cast[lsx.M128i](a), cast[lsx.M128i](a)))

let la_not_i32v128: proc (a: i32v128): i32v128 = if lsxExists(): not_i32v128_lsx else: not_i32v128_fallback

template `not`*(a: i32v128): i32v128 = 
  when defined(amd64) or defined(i386): x86_not_i32v128(a)
  elif defined(aarch64): arm_not_i32v128(a)
  elif defined(loongarch): la_not_i32v128(a)

# ========================
# not / i64v128 / value
# ========================

func not_i64v128_fallback(a: i64v128): i64v128 = 
  var output: i64v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    unroll(i, 0, 1):
      pOut[i] = not pA[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    unroll(i, 0, 3):
      pOut[i] = not pA[i]
  return output

func not_i64v128_sse(a: i64v128): i64v128 = 
  let allOnes = mm_set1_epi32(-1)
  return cast[i64v128](mm_xor_si128(cast[sse2.M128i](a), cast[sse2.M128i](allOnes)))

let x86_not_i64v128: proc (a: i64v128): i64v128 = 
  if sseExists(): not_i64v128_sse 
  else: not_i64v128_fallback 

func not_i64v128_neon(a: i64v128): i64v128 =
  return cast[i64v128](vmvnq_u32(cast[neon.Uint32x4](a)))

let arm_not_i64v128: proc (a: i64v128): i64v128 = if neonExists(): not_i64v128_neon else: not_i64v128_fallback

func not_i64v128_lsx(a: i64v128): i64v128 =
  return cast[i64v128](lsx_vnor_v(cast[lsx.M128i](a), cast[lsx.M128i](a)))

let la_not_i64v128: proc (a: i64v128): i64v128 = if lsxExists(): not_i64v128_lsx else: not_i64v128_fallback

template `not`*(a: i64v128): i64v128 = 
  when defined(amd64) or defined(i386): x86_not_i64v128(a)
  elif defined(aarch64): arm_not_i64v128(a)
  elif defined(loongarch): la_not_i64v128(a)

# ========================
# not / u8v128 / value
# ========================

func not_u8v128_fallback(a: u8v128): u8v128 = 
  var output: u8v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    unroll(i, 0, 1):
      pOut[i] = not pA[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    unroll(i, 0, 3):
      pOut[i] = not pA[i]
  return output

func not_u8v128_sse(a: u8v128): u8v128 = 
  let allOnes = mm_set1_epi32(-1)
  return cast[u8v128](mm_xor_si128(cast[sse2.M128i](a), cast[sse2.M128i](allOnes)))

let x86_not_u8v128: proc (a: u8v128): u8v128 = 
  if sseExists(): not_u8v128_sse 
  else: not_u8v128_fallback 

func not_u8v128_neon(a: u8v128): u8v128 =
  return cast[u8v128](vmvnq_u8(cast[neon.Uint8x16](a)))

let arm_not_u8v128: proc (a: u8v128): u8v128 = if neonExists(): not_u8v128_neon else: not_u8v128_fallback

func not_u8v128_lsx(a: u8v128): u8v128 =
  return cast[u8v128](lsx_vnor_v(cast[lsx.M128i](a), cast[lsx.M128i](a)))

let la_not_u8v128: proc (a: u8v128): u8v128 = if lsxExists(): not_u8v128_lsx else: not_u8v128_fallback

template `not`*(a: u8v128): u8v128 = 
  when defined(amd64) or defined(i386): x86_not_u8v128(a)
  elif defined(aarch64): arm_not_u8v128(a)
  elif defined(loongarch): la_not_u8v128(a)

# ========================
# not / u16v128 / value
# ========================

func not_u16v128_fallback(a: u16v128): u16v128 = 
  var output: u16v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    unroll(i, 0, 1):
      pOut[i] = not pA[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    unroll(i, 0, 3):
      pOut[i] = not pA[i]
  return output

func not_u16v128_sse(a: u16v128): u16v128 = 
  let allOnes = mm_set1_epi32(-1)
  return cast[u16v128](mm_xor_si128(cast[sse2.M128i](a), cast[sse2.M128i](allOnes)))

let x86_not_u16v128: proc (a: u16v128): u16v128 = 
  if sseExists(): not_u16v128_sse 
  else: not_u16v128_fallback 

func not_u16v128_neon(a: u16v128): u16v128 =
  return cast[u16v128](vmvnq_u16(cast[neon.Uint16x8](a)))

let arm_not_u16v128: proc (a: u16v128): u16v128 = if neonExists(): not_u16v128_neon else: not_u16v128_fallback

func not_u16v128_lsx(a: u16v128): u16v128 =
  return cast[u16v128](lsx_vnor_v(cast[lsx.M128i](a), cast[lsx.M128i](a)))

let la_not_u16v128: proc (a: u16v128): u16v128 = if lsxExists(): not_u16v128_lsx else: not_u16v128_fallback

template `not`*(a: u16v128): u16v128 = 
  when defined(amd64) or defined(i386): x86_not_u16v128(a)
  elif defined(aarch64): arm_not_u16v128(a)
  elif defined(loongarch): la_not_u16v128(a)

# ========================
# not / u32v128 / value
# ========================

func not_u32v128_fallback(a: u32v128): u32v128 = 
  var output: u32v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    unroll(i, 0, 1):
      pOut[i] = not pA[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    unroll(i, 0, 3):
      pOut[i] = not pA[i]
  return output

func not_u32v128_sse(a: u32v128): u32v128 = 
  let allOnes = mm_set1_epi32(-1)
  return cast[u32v128](mm_xor_si128(cast[sse2.M128i](a), cast[sse2.M128i](allOnes)))

let x86_not_u32v128: proc (a: u32v128): u32v128 = 
  if sseExists(): not_u32v128_sse 
  else: not_u32v128_fallback 

func not_u32v128_neon(a: u32v128): u32v128 =
  return cast[u32v128](vmvnq_u32(cast[neon.Uint32x4](a)))

let arm_not_u32v128: proc (a: u32v128): u32v128 = if neonExists(): not_u32v128_neon else: not_u32v128_fallback

func not_u32v128_lsx(a: u32v128): u32v128 =
  return cast[u32v128](lsx_vnor_v(cast[lsx.M128i](a), cast[lsx.M128i](a)))

let la_not_u32v128: proc (a: u32v128): u32v128 = if lsxExists(): not_u32v128_lsx else: not_u32v128_fallback

template `not`*(a: u32v128): u32v128 = 
  when defined(amd64) or defined(i386): x86_not_u32v128(a)
  elif defined(aarch64): arm_not_u32v128(a)
  elif defined(loongarch): la_not_u32v128(a)

# ========================
# not / u64v128 / value
# ========================

func not_u64v128_fallback(a: u64v128): u64v128 = 
  var output: u64v128
  when CPUBits == 64:
    let pOut = cast[ptr array[2, uint64]](addr output)
    let pA   = cast[ptr array[2, uint64]](addr a)
    unroll(i, 0, 1):
      pOut[i] = not pA[i]
  else:
    let pOut = cast[ptr array[4, uint32]](addr output)
    let pA   = cast[ptr array[4, uint32]](addr a)
    unroll(i, 0, 3):
      pOut[i] = not pA[i]
  return output

func not_u64v128_sse(a: u64v128): u64v128 = 
  let allOnes = mm_set1_epi32(-1)
  return cast[u64v128](mm_xor_si128(cast[sse2.M128i](a), cast[sse2.M128i](allOnes)))

let x86_not_u64v128: proc (a: u64v128): u64v128 = 
  if sseExists(): not_u64v128_sse 
  else: not_u64v128_fallback 

func not_u64v128_neon(a: u64v128): u64v128 =
  return cast[u64v128](vmvnq_u32(cast[neon.Uint32x4](a)))

let arm_not_u64v128: proc (a: u64v128): u64v128 = if neonExists(): not_u64v128_neon else: not_u64v128_fallback

func not_u64v128_lsx(a: u64v128): u64v128 =
  return cast[u64v128](lsx_vnor_v(cast[lsx.M128i](a), cast[lsx.M128i](a)))

let la_not_u64v128: proc (a: u64v128): u64v128 = if lsxExists(): not_u64v128_lsx else: not_u64v128_fallback

template `not`*(a: u64v128): u64v128 = 
  when defined(amd64) or defined(i386): x86_not_u64v128(a)
  elif defined(aarch64): arm_not_u64v128(a)
  elif defined(loongarch): la_not_u64v128(a)