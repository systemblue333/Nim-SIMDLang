
when defined(amd64) or defined(i386):
  proc cpuid(eax, ecx: int32): array[4, int32] {.importc: "__builtin_cpuid", nodecl.}
  
  var 
    initialized: bool = false
    hasSse, hasSse2, hasSse3, hasSsse3, hasSse41, hasSse42: bool
    hasAvx, hasAvx2, hasAvx512, hasAvx10: bool

  proc initX86CpuFeatures() =
    if initialized: return

    let res1 = cpuid(1, 0)
    let edx1 = res1[3]
    let ecx1 = res1[2]

    hasSse   = (edx1 and (1 shl 25)) != 0
    hasSse2  = (edx1 and (1 shl 26)) != 0
    hasSse3  = (ecx1 and (1 shl 0)) != 0
    hasSsse3 = (ecx1 and (1 shl 9)) != 0
    hasSse41 = (ecx1 and (1 shl 19)) != 0
    hasSse42 = (ecx1 and (1 shl 20)) != 0
    hasAvx   = (ecx1 and (1 shl 28)) != 0

    let res7 = cpuid(7, 0)
    let ebx7 = res7[1]
    let edx7 = res7[3]

    hasAvx2   = (ebx7 and (1 shl 5)) != 0
    hasAvx512 = (ebx7 and (1 shl 16)) != 0 
    let res7_1 = cpuid(7, 1)
    let edx7_1 = res7_1[3]
    hasAvx10   = (edx7_1 and (1 shl 19)) != 0

    initialized = true



proc sseExists*(): bool {.inline.} =
  when defined(amd64): true 
  elif defined(i386):
    initX86CpuFeatures()
    hasSse
  else: false

proc sse2Exists*(): bool {.inline.} =
  when defined(amd64): true
  elif defined(i386):
    initX86CpuFeatures()
    hasSse2
  else: false

proc sse3Exists*(): bool {.inline.} =
  when defined(amd64) or defined(i386):
    initX86CpuFeatures()
    hasSse3
  else: false

proc ssse3Exists*(): bool {.inline.} =
  when defined(amd64) or defined(i386):
    initX86CpuFeatures()
    hasSsse3
  else: false

proc sse41Exists*(): bool {.inline.} =
  when defined(amd64) or defined(i386):
    initX86CpuFeatures()
    hasSse41
  else: false

proc sse42Exists*(): bool {.inline.} =
  when defined(amd64) or defined(i386):
    initX86CpuFeatures()
    hasSse42
  else: false

proc avxExists*(): bool {.inline.} =
  when defined(amd64) or defined(i386):
    initX86CpuFeatures()
    hasAvx
  else: false

proc avx2Exists*(): bool {.inline.} =
  when defined(amd64) or defined(i386):
    initX86CpuFeatures()
    hasAvx2
  else: false

proc avx512Exists*(): bool {.inline.} =
  when defined(amd64) or defined(i386):
    initX86CpuFeatures()
    hasAvx512
  else: false

proc avx10Exists*(): bool {.inline.} =
  when defined(amd64) or defined(i386):
    initX86CpuFeatures()
    hasAvx10
  else: false

proc neonExists*(): bool {.inline.} =
  when defined(arm64): true 
  elif defined(arm):
    defined(neon) 
  else: false

proc lsxExists*(): bool {.inline.} =
  when defined(loongarch64):
    true 
  else: false

proc lasxExists*(): bool {.inline.} =
  when defined(loongarch64):
    defined(lasx)
  else: false