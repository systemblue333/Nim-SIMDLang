import neon

func vcvth_bf16_f32*(a: Float32_t): Bfloat16_t {.importc: "vcvth_bf16_f32".}

func vcvtah_f32_bf16*(a: Bfloat16_t): Float32_t {.importc: "vcvtah_f32_bf16".}
