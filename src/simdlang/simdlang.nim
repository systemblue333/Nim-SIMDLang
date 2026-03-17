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

template `+`*[T](a, b: T): T

template `-`*[T](a, b: T): T

template `*`*[T](a, b: T): T

template `div`*[T](a, b: T): T

template `mod`*[T](a, b: T): T

template sqrt*[T](a: T): T

template abs*[T](a: T): T

template min*[T](a, b: T): T

template max*[T](a, b: T): T

template `and`*[T](a, b: T): T

template `or`*[T](a, b: T): T

template `xor`*[T](a, b: T): T

template `not`*[T](a: T): T

template `andnot`*[T](a, b: T): T

template `shl`*[T](a: T, count: static int): T

template `shr`*[T](a: T, count: static int): T

template `==`*[T](a, b: T): T

template `!=`*[T](a, b: T): T

template `>`*[T](a, b: T): T

template `<`*[T])a, b: T): T

template `>=`*[T](a, b: T): T

template `<=`*[T](a, b: T): T

template loadU*[T](p: pointer): T

template loadA*[T](p: pointer): T

template store*[T](p: pointer, a: T)

template set1*[V, T](val: T): V

template setzero*[T](): T

template blend*[T](a, b, mask: T): T

# (a AND mask) OR (b AND NOT mask)
template bitwiseSelect*[T](a, b, mask: T): T

template shuffle*[T](a: T, pattern: static int): T

template permute*[T](a, indices: T): T

template unpackHi*[T](a, b: T): T

template unpackLo*[T](a, b: T): T

template hadd*[T](a, b: T): T

template hsub*[T](a, b: T): T

template rsqrt*[T](a: T): T

template rcp*[T](a: T): T

template movemask*[T](a: T): int32

template broadcast*[V, T](val: T): V

template broadcast_load*[V](p: pointer): V

template bitcast*[TV, SV](v: SV): TV

template convert*[TV, SV](v: SV): TV

template `sar`*[T](a: T, count: static int): T

template extract*[V, T](a: V, index: static int): T

template gather*[V](base: pointer, indices: V, scale: static int): V

template scatter*[V](base: pointer, indices: V, v: V, scale: static int)

template addSat*[T](a, b: T): T

template subSat*[T](a, b: T): T

template sum*[T](a: T): T

template floor*[T](a: T): T

template ceil*[T](a: T): T

template round*[T](a: T): T

template fmadd*[T](a, b, c: T): T

template testZ*[T](a: T): bool

template testC*[T](a: T): bool

template maskLoad*[T](p: pointer, mask: T): T

template maskStore*[T](p: pointer, a: T, mask: T)
