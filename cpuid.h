#pragma once

#ifdef _MSC_VER
  #include<intrin.h>
  #define CPUID(info, x)  __cpuidex(reinterpret_cast<int *>(info), x, 0)
#else
  #include <cpuid.h>
  #define CPUID(info, x)  __cpuid_count(x, 0, info[0], info[1], info[2], info[3])
#endif

class SIMDFlags final {
 public:
  SIMDFlags(SIMDFlags &&) = delete;
  SIMDFlags(const SIMDFlags &) = delete;
  SIMDFlags &operator=(const SIMDFlags &) = delete;

  SIMDFlags() {
    unsigned int cpuInfo[4];
    // CPUID: https://en.wikipedia.org/wiki/CPUID
    CPUID(cpuInfo, 0x00000001);
    simd_flags_ |= cpuInfo[3] & (1 << 25) ? SIMD_SSE   : SIMD_NONE;
    simd_flags_ |= cpuInfo[3] & (1 << 26) ? SIMD_SSE2  : SIMD_NONE;
    simd_flags_ |= cpuInfo[2] & (1 << 0)  ? SIMD_SSE3  : SIMD_NONE;
    simd_flags_ |= cpuInfo[2] & (1 << 9)  ? SIMD_SSSE3 : SIMD_NONE;
    simd_flags_ |= cpuInfo[2] & (1 << 19) ? SIMD_SSE41 : SIMD_NONE;
    simd_flags_ |= cpuInfo[2] & (1 << 20) ? SIMD_SSE42 : SIMD_NONE;
    simd_flags_ |= cpuInfo[2] & (1 << 12) ? SIMD_FMA3  : SIMD_NONE;
    simd_flags_ |= cpuInfo[2] & (1 << 28) ? SIMD_AVX   : SIMD_NONE;

    CPUID(cpuInfo, 0x00000007);
    simd_flags_ |= cpuInfo[1] & (1 << 5)  ? SIMD_AVX2  : SIMD_NONE;
    simd_flags_ |= cpuInfo[1] & (1 << 16) ? SIMD_AVX512: SIMD_NONE;

    CPUID(cpuInfo, 0x80000001);
    simd_flags_ |= cpuInfo[2] & (1 << 16) ? SIMD_FMA4  : SIMD_NONE;
  }

  inline bool hasSSE()   const { return simd_flags_ & SIMD_SSE;   }
  inline bool hasSSE2()  const { return simd_flags_ & SIMD_SSE2;  }
  inline bool hasSSE3()  const { return simd_flags_ & SIMD_SSE3;  }
  inline bool hasSSSE3() const { return simd_flags_ & SIMD_SSSE3; }
  inline bool hasSSE41() const { return simd_flags_ & SIMD_SSE41; }
  inline bool hasSSE42() const { return simd_flags_ & SIMD_SSE42; }
  inline bool hasFMA3()  const { return simd_flags_ & SIMD_FMA3;  }
  inline bool hasFMA4()  const { return simd_flags_ & SIMD_FMA4;  }
  inline bool hasAVX()   const { return simd_flags_ & SIMD_AVX;   }
  inline bool hasAVX2()  const { return simd_flags_ & SIMD_AVX2;  }
  inline bool hasAVX512()const { return simd_flags_ & SIMD_AVX512;}

 private:
  enum simd_t {
    SIMD_NONE     = 0,        ///< None
    SIMD_SSE      = 1 << 0,   ///< SSE
    SIMD_SSE2     = 1 << 1,   ///< SSE 2
    SIMD_SSE3     = 1 << 2,   ///< SSE 3
    SIMD_SSSE3    = 1 << 3,   ///< SSSE 3
    SIMD_SSE41    = 1 << 4,   ///< SSE 4.1
    SIMD_SSE42    = 1 << 5,   ///< SSE 4.2
    SIMD_FMA3     = 1 << 6,   ///< FMA 3
    SIMD_FMA4     = 1 << 7,   ///< FMA 4
    SIMD_AVX      = 1 << 8,   ///< AVX
    SIMD_AVX2     = 1 << 9,   ///< AVX 2
    SIMD_AVX512   = 1 << 10,  ///< AVX 512
  };

  int simd_flags_ = SIMD_NONE;
};
