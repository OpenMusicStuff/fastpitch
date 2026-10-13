// fastpitch/pyin.hpp
//
// Header-only probabilistic YIN (pYIN) fundamental-frequency tracker.
//
//   Mauch & Dixon, "pYIN: A fundamental frequency estimator using probabilistic
//   threshold distributions", ICASSP 2014.
//
// The algorithm (difference function, threshold/Boltzmann priors, observation model,
// transition model and Viterbi decoding) reproduces librosa.pyin (librosa >= 1.0) so
// results can be compared frame-for-frame with the de-facto reference implementation.
//
// Performance techniques. None of them changes the decoded result: the observation model
// matches librosa to ~1e-15 and the Viterbi path is identical (see tests/).
//
//   YIN stage
//   * Lane-batched ("vertical") SIMD FFT: W frames are transformed at once, one per SIMD
//     lane, so every butterfly runs at full vector width without shuffles.
//   * Mixed-radix (2/3/4/5) Stockham FFT, cache-blocked (four-step) for large sizes, at the
//     smallest length >= N + max_period that yields an exact linear autocorrelation.
//   * The inverse transform of the (real, even) power spectrum is a DCT-I evaluated with a
//     half-length real FFT; its pre-processing is fused into the power-spectrum pass.
//   * Vectorised CMNDF (compensated sums), vectorised trough detection (compare+movemask),
//     O(1) threshold lookup and grouped threshold/Boltzmann prior evaluation.
//
//   Viterbi stage (S = 2 x pitch bins states, band of B transitions per state)
//   * The row-normalised triangular transition band turns the O(S^2) recursion into a
//     max-plus convolution over the band.
//   * Argmax for free: values live on a binary fixed-point grid inside doubles and every
//     kernel tap carries a unique code in the low bits, so sums are exact and one vmaxpd
//     yields the maximum and the first maximising predecessor (librosa's tie rule).
//   * The score matrix is Monge (strictly concave kernel), so the first argmax is monotone
//     in the destination; blocks of destinations only scan between neighbouring argmaxes.
//   * Provably exact pruning: voiced states without a candidate (log p = log tiny) are
//     dominated by the unvoiced state of the same bin, and states more than Lambda below the
//     frame maximum can never be on the optimal path (bounded, exact grid range).
//   * Back-pointers are stored as uint16 (librosa's own state type).
//
//   * Optional multithreading: frames are analysed in parallel while the decoder consumes
//     them in order; results are independent of the thread count.
//
// Requires C++17. Compile with -O2/-O3 and the widest SIMD you can target (-mavx2 -mfma /
// -march=native, or /arch:AVX2; SSE2 and NEON are used automatically). Do not compile with
// -ffast-math.
//
// SPDX-License-Identifier: MIT

#ifndef FASTPITCH_PYIN_HPP_INCLUDED
#define FASTPITCH_PYIN_HPP_INCLUDED

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#warning "fastpitch relies on IEEE-754 semantics (exact grid arithmetic, infinities, compensated sums); compile without -ffast-math / -ffinite-math-only"
#endif

#if defined(_MSC_VER)
#define FASTPITCH_INLINE __forceinline
#else
#define FASTPITCH_INLINE inline __attribute__((always_inline))
#endif

#if !defined(FASTPITCH_SIMD_SCALAR)
#if defined(__AVX2__) && (defined(__FMA__) || (defined(_MSC_VER) && !defined(__clang__)))
#define FASTPITCH_SIMD_AVX2 1
#include <immintrin.h>
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define FASTPITCH_SIMD_SSE2 1
#include <emmintrin.h>
#elif (defined(__aarch64__) || defined(_M_ARM64)) && (defined(__ARM_NEON) || defined(_M_ARM64))
#define FASTPITCH_SIMD_NEON 1
#include <arm_neon.h>
#endif
#endif

namespace fastpitch {

// ---------------------------------------------------------------------------------------
// Public configuration / result types
// ---------------------------------------------------------------------------------------

/// Parameters of the pYIN analysis. Defaults match librosa.pyin (fmin=C2, fmax=C7).
struct PyinOptions {
  double sample_rate = 22050.0;
  double fmin = 65.40639132514966;   ///< Hz (librosa.note_to_hz('C2'))
  double fmax = 2093.004522404789;   ///< Hz (librosa.note_to_hz('C7'))
  int frame_length = 2048;           ///< samples
  int hop_length = 0;                ///< samples; 0 -> frame_length / 4
  int n_thresholds = 100;            ///< number of YIN thresholds
  double beta_a = 2.0;               ///< beta prior over thresholds (shape a)
  double beta_b = 18.0;              ///< beta prior over thresholds (shape b)
  double boltzmann_parameter = 2.0;  ///< Boltzmann prior over troughs
  double resolution = 0.1;           ///< pitch-bin resolution in semitones
  double max_transition_rate = 35.92;///< octaves / second
  double switch_prob = 0.01;         ///< voiced <-> unvoiced switch probability
  double no_trough_prob = 0.01;      ///< mass given to the global minimum when no trough
  /// Transitions with probability below this are ignored by the Viterbi decoder
  /// (librosa >= 1.0 default). Set to 0 for exact (full) Viterbi inference.
  double transition_min_prob = 1e-4;
  bool center = true;                ///< pad frame_length/2 zeros on both sides (librosa)
  /// librosa (<= 1.0) drops the y[0]^2 energy term of the lag-1 difference d(1) (it zeroes
  /// the cumulative-energy row before reading it). true reproduces this so results match
  /// librosa frame-for-frame; false uses the exact definition d(k) = 2(r(0) - r(k)) -
  /// sum_{m<k} y[m]^2 for every lag.
  bool reproduce_librosa_quirks = true;
  bool fill_unvoiced = true;         ///< write fill_value to f0 of unvoiced frames
  /// Report the parabolically refined frequency of the YIN candidate that produced the
  /// decoded pitch bin instead of the bin centre (as the original pYIN does). Does not change
  /// the decoded path or voicing; off by default for librosa parity.
  bool refine_f0 = false;
  double fill_value = std::numeric_limits<double>::quiet_NaN();
  int num_threads = 1;               ///< 0 -> std::thread::hardware_concurrency()
};

/// Output of the analysis (one entry per frame).
template <class T>
struct PyinResult {
  std::vector<T> f0;                  ///< Hz (fill_value where unvoiced, if enabled)
  std::vector<std::uint8_t> voiced;   ///< 1 if the decoded state is voiced
  std::vector<T> voiced_prob;         ///< P(voiced) from the observation model
  double sample_rate = 0.0;
  int hop_length = 0;

  std::size_t size() const { return f0.size(); }
  /// Time (seconds) of frame i, matching librosa.times_like(f0, sr=sr, hop_length=hop).
  double time(std::size_t i) const { return double(i) * hop_length / sample_rate; }
};

namespace detail {

// ---------------------------------------------------------------------------------------
// Aligned storage
// ---------------------------------------------------------------------------------------

template <class T>
class AlignedBuffer {
 public:
  static constexpr std::size_t kAlign = 64;
  AlignedBuffer() = default;
  explicit AlignedBuffer(std::size_t n) { resize(n); }
  ~AlignedBuffer() { release(); }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
  AlignedBuffer(AlignedBuffer&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
  AlignedBuffer& operator=(AlignedBuffer&& o) noexcept {
    if (this != &o) {
      release();
      p_ = o.p_; n_ = o.n_;
      o.p_ = nullptr; o.n_ = 0;
    }
    return *this;
  }
  void resize(std::size_t n) {
    if (n == n_) return;
    release();
    if (n) {
      p_ = static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(kAlign)));
      n_ = n;
    }
  }
  void fill(T v) { std::fill(p_, p_ + n_, v); }
  T* data() { return p_; }
  const T* data() const { return p_; }
  std::size_t size() const { return n_; }
  T& operator[](std::size_t i) { return p_[i]; }
  const T& operator[](std::size_t i) const { return p_[i]; }

 private:
  void release() {
    if (p_) ::operator delete(p_, std::align_val_t(kAlign));
    p_ = nullptr;
    n_ = 0;
  }
  T* p_ = nullptr;
  std::size_t n_ = 0;
};

// ---------------------------------------------------------------------------------------
// SIMD abstraction. Each vector type V provides:
//   V::value_type, V::size, load/loadu/set1/iota, store/storeu, transpose (V x V block),
// and free functions + - *, fmadd/fmsub/fnmadd, vmax/vmin, cmpgt, select, hmax/hmin,
// store_i32 (truncating conversion).
// ---------------------------------------------------------------------------------------
namespace simd {

template <class T>
struct Scalar {
  using value_type = T;
  static constexpr int size = 1;
  T v;
  static FASTPITCH_INLINE Scalar load(const T* p) { return {*p}; }
  static FASTPITCH_INLINE Scalar loadu(const T* p) { return {*p}; }
  static FASTPITCH_INLINE Scalar set1(T x) { return {x}; }
  static FASTPITCH_INLINE Scalar iota() { return {T(0)}; }
  FASTPITCH_INLINE void store(T* p) const { *p = v; }
  FASTPITCH_INLINE void storeu(T* p) const { *p = v; }
  static FASTPITCH_INLINE void transpose(const T* const* rows, std::size_t col, T* out) {
    out[0] = rows[0][col];
  }
};
template <class T>
struct ScalarMask { bool m; };
template <class T> FASTPITCH_INLINE Scalar<T> operator+(Scalar<T> a, Scalar<T> b) { return {a.v + b.v}; }
template <class T> FASTPITCH_INLINE Scalar<T> operator-(Scalar<T> a, Scalar<T> b) { return {a.v - b.v}; }
template <class T> FASTPITCH_INLINE Scalar<T> operator*(Scalar<T> a, Scalar<T> b) { return {a.v * b.v}; }
template <class T> FASTPITCH_INLINE Scalar<T> fmadd(Scalar<T> a, Scalar<T> b, Scalar<T> c) { return {a.v * b.v + c.v}; }
template <class T> FASTPITCH_INLINE Scalar<T> fmsub(Scalar<T> a, Scalar<T> b, Scalar<T> c) { return {a.v * b.v - c.v}; }
template <class T> FASTPITCH_INLINE Scalar<T> fnmadd(Scalar<T> a, Scalar<T> b, Scalar<T> c) { return {c.v - a.v * b.v}; }
template <class T> FASTPITCH_INLINE Scalar<T> vmax(Scalar<T> a, Scalar<T> b) { return {a.v > b.v ? a.v : b.v}; }
template <class T> FASTPITCH_INLINE Scalar<T> vmin(Scalar<T> a, Scalar<T> b) { return {a.v < b.v ? a.v : b.v}; }
template <class T> FASTPITCH_INLINE ScalarMask<T> cmpgt(Scalar<T> a, Scalar<T> b) { return {a.v > b.v}; }
template <class T> FASTPITCH_INLINE ScalarMask<T> cmple(Scalar<T> a, Scalar<T> b) { return {a.v <= b.v}; }
template <class T> FASTPITCH_INLINE ScalarMask<T> mask_and(ScalarMask<T> a, ScalarMask<T> b) { return {a.m && b.m}; }
template <class T> FASTPITCH_INLINE int movemask(ScalarMask<T> a) { return a.m ? 1 : 0; }
template <class T> FASTPITCH_INLINE Scalar<T> operator/(Scalar<T> a, Scalar<T> b) { return {a.v / b.v}; }
template <class T> FASTPITCH_INLINE Scalar<T> select(ScalarMask<T> m, Scalar<T> a, Scalar<T> b) { return m.m ? a : b; }
template <class T> FASTPITCH_INLINE T hmax(Scalar<T> a) { return a.v; }
template <class T> FASTPITCH_INLINE T hmin(Scalar<T> a) { return a.v; }
template <class T> FASTPITCH_INLINE void store_i32(std::int32_t* p, Scalar<T> a) { *p = static_cast<std::int32_t>(a.v); }
template <class T> FASTPITCH_INLINE Scalar<T> round_nearest(Scalar<T> a) { return {std::nearbyint(a.v)}; }
template <class T> FASTPITCH_INLINE void store_u16(std::uint16_t* p, Scalar<T> a) { *p = static_cast<std::uint16_t>(a.v); }
FASTPITCH_INLINE Scalar<double> bit_and(Scalar<double> a, Scalar<double> b) {
  std::uint64_t x, y;
  std::memcpy(&x, &a.v, 8);
  std::memcpy(&y, &b.v, 8);
  x &= y;
  std::memcpy(&a.v, &x, 8);
  return a;
}
FASTPITCH_INLINE Scalar<double> bit_or(Scalar<double> a, Scalar<double> b) {
  std::uint64_t x, y;
  std::memcpy(&x, &a.v, 8);
  std::memcpy(&y, &b.v, 8);
  x |= y;
  std::memcpy(&a.v, &x, 8);
  return a;
}

#if defined(FASTPITCH_SIMD_AVX2)
struct F8 {
  using value_type = float;
  static constexpr int size = 8;
  __m256 v;
  static FASTPITCH_INLINE F8 load(const float* p) { return {_mm256_load_ps(p)}; }
  static FASTPITCH_INLINE F8 loadu(const float* p) { return {_mm256_loadu_ps(p)}; }
  static FASTPITCH_INLINE F8 set1(float x) { return {_mm256_set1_ps(x)}; }
  static FASTPITCH_INLINE F8 iota() { return {_mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7)}; }
  FASTPITCH_INLINE void store(float* p) const { _mm256_store_ps(p, v); }
  FASTPITCH_INLINE void storeu(float* p) const { _mm256_storeu_ps(p, v); }
  static FASTPITCH_INLINE void transpose(const float* const* rows, std::size_t col, float* out) {
    __m256 r0 = _mm256_loadu_ps(rows[0] + col), r1 = _mm256_loadu_ps(rows[1] + col);
    __m256 r2 = _mm256_loadu_ps(rows[2] + col), r3 = _mm256_loadu_ps(rows[3] + col);
    __m256 r4 = _mm256_loadu_ps(rows[4] + col), r5 = _mm256_loadu_ps(rows[5] + col);
    __m256 r6 = _mm256_loadu_ps(rows[6] + col), r7 = _mm256_loadu_ps(rows[7] + col);
    __m256 t0 = _mm256_unpacklo_ps(r0, r1), t1 = _mm256_unpackhi_ps(r0, r1);
    __m256 t2 = _mm256_unpacklo_ps(r2, r3), t3 = _mm256_unpackhi_ps(r2, r3);
    __m256 t4 = _mm256_unpacklo_ps(r4, r5), t5 = _mm256_unpackhi_ps(r4, r5);
    __m256 t6 = _mm256_unpacklo_ps(r6, r7), t7 = _mm256_unpackhi_ps(r6, r7);
    __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    _mm256_store_ps(out + 0, _mm256_permute2f128_ps(s0, s4, 0x20));
    _mm256_store_ps(out + 8, _mm256_permute2f128_ps(s1, s5, 0x20));
    _mm256_store_ps(out + 16, _mm256_permute2f128_ps(s2, s6, 0x20));
    _mm256_store_ps(out + 24, _mm256_permute2f128_ps(s3, s7, 0x20));
    _mm256_store_ps(out + 32, _mm256_permute2f128_ps(s0, s4, 0x31));
    _mm256_store_ps(out + 40, _mm256_permute2f128_ps(s1, s5, 0x31));
    _mm256_store_ps(out + 48, _mm256_permute2f128_ps(s2, s6, 0x31));
    _mm256_store_ps(out + 56, _mm256_permute2f128_ps(s3, s7, 0x31));
  }
};
struct F8M { __m256 m; };
FASTPITCH_INLINE F8 operator+(F8 a, F8 b) { return {_mm256_add_ps(a.v, b.v)}; }
FASTPITCH_INLINE F8 operator-(F8 a, F8 b) { return {_mm256_sub_ps(a.v, b.v)}; }
FASTPITCH_INLINE F8 operator*(F8 a, F8 b) { return {_mm256_mul_ps(a.v, b.v)}; }
FASTPITCH_INLINE F8 fmadd(F8 a, F8 b, F8 c) { return {_mm256_fmadd_ps(a.v, b.v, c.v)}; }
FASTPITCH_INLINE F8 fmsub(F8 a, F8 b, F8 c) { return {_mm256_fmsub_ps(a.v, b.v, c.v)}; }
FASTPITCH_INLINE F8 fnmadd(F8 a, F8 b, F8 c) { return {_mm256_fnmadd_ps(a.v, b.v, c.v)}; }
FASTPITCH_INLINE F8 vmax(F8 a, F8 b) { return {_mm256_max_ps(a.v, b.v)}; }
FASTPITCH_INLINE F8 vmin(F8 a, F8 b) { return {_mm256_min_ps(a.v, b.v)}; }
FASTPITCH_INLINE F8M cmpgt(F8 a, F8 b) { return {_mm256_cmp_ps(a.v, b.v, _CMP_GT_OQ)}; }
FASTPITCH_INLINE F8M cmple(F8 a, F8 b) { return {_mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ)}; }
FASTPITCH_INLINE F8M mask_and(F8M a, F8M b) { return {_mm256_and_ps(a.m, b.m)}; }
FASTPITCH_INLINE int movemask(F8M a) { return _mm256_movemask_ps(a.m); }
FASTPITCH_INLINE F8 operator/(F8 a, F8 b) { return {_mm256_div_ps(a.v, b.v)}; }
FASTPITCH_INLINE F8 select(F8M m, F8 a, F8 b) { return {_mm256_blendv_ps(b.v, a.v, m.m)}; }
FASTPITCH_INLINE float hmax(F8 a) {
  __m128 m = _mm_max_ps(_mm256_castps256_ps128(a.v), _mm256_extractf128_ps(a.v, 1));
  m = _mm_max_ps(m, _mm_movehl_ps(m, m));
  m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
  return _mm_cvtss_f32(m);
}
FASTPITCH_INLINE float hmin(F8 a) {
  __m128 m = _mm_min_ps(_mm256_castps256_ps128(a.v), _mm256_extractf128_ps(a.v, 1));
  m = _mm_min_ps(m, _mm_movehl_ps(m, m));
  m = _mm_min_ss(m, _mm_shuffle_ps(m, m, 1));
  return _mm_cvtss_f32(m);
}
FASTPITCH_INLINE void store_i32(std::int32_t* p, F8 a) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), _mm256_cvttps_epi32(a.v));
}

struct D4 {
  using value_type = double;
  static constexpr int size = 4;
  __m256d v;
  static FASTPITCH_INLINE D4 load(const double* p) { return {_mm256_load_pd(p)}; }
  static FASTPITCH_INLINE D4 loadu(const double* p) { return {_mm256_loadu_pd(p)}; }
  static FASTPITCH_INLINE D4 set1(double x) { return {_mm256_set1_pd(x)}; }
  static FASTPITCH_INLINE D4 iota() { return {_mm256_setr_pd(0, 1, 2, 3)}; }
  FASTPITCH_INLINE void store(double* p) const { _mm256_store_pd(p, v); }
  FASTPITCH_INLINE void storeu(double* p) const { _mm256_storeu_pd(p, v); }
  static FASTPITCH_INLINE void transpose(const double* const* rows, std::size_t col, double* out) {
    __m256d r0 = _mm256_loadu_pd(rows[0] + col), r1 = _mm256_loadu_pd(rows[1] + col);
    __m256d r2 = _mm256_loadu_pd(rows[2] + col), r3 = _mm256_loadu_pd(rows[3] + col);
    __m256d t0 = _mm256_unpacklo_pd(r0, r1), t1 = _mm256_unpackhi_pd(r0, r1);
    __m256d t2 = _mm256_unpacklo_pd(r2, r3), t3 = _mm256_unpackhi_pd(r2, r3);
    _mm256_store_pd(out + 0, _mm256_permute2f128_pd(t0, t2, 0x20));
    _mm256_store_pd(out + 4, _mm256_permute2f128_pd(t1, t3, 0x20));
    _mm256_store_pd(out + 8, _mm256_permute2f128_pd(t0, t2, 0x31));
    _mm256_store_pd(out + 12, _mm256_permute2f128_pd(t1, t3, 0x31));
  }
};
struct D4M { __m256d m; };
FASTPITCH_INLINE D4 operator+(D4 a, D4 b) { return {_mm256_add_pd(a.v, b.v)}; }
FASTPITCH_INLINE D4 operator-(D4 a, D4 b) { return {_mm256_sub_pd(a.v, b.v)}; }
FASTPITCH_INLINE D4 operator*(D4 a, D4 b) { return {_mm256_mul_pd(a.v, b.v)}; }
FASTPITCH_INLINE D4 fmadd(D4 a, D4 b, D4 c) { return {_mm256_fmadd_pd(a.v, b.v, c.v)}; }
FASTPITCH_INLINE D4 fmsub(D4 a, D4 b, D4 c) { return {_mm256_fmsub_pd(a.v, b.v, c.v)}; }
FASTPITCH_INLINE D4 fnmadd(D4 a, D4 b, D4 c) { return {_mm256_fnmadd_pd(a.v, b.v, c.v)}; }
FASTPITCH_INLINE D4 vmax(D4 a, D4 b) { return {_mm256_max_pd(a.v, b.v)}; }
FASTPITCH_INLINE D4 vmin(D4 a, D4 b) { return {_mm256_min_pd(a.v, b.v)}; }
FASTPITCH_INLINE D4M cmpgt(D4 a, D4 b) { return {_mm256_cmp_pd(a.v, b.v, _CMP_GT_OQ)}; }
FASTPITCH_INLINE D4M cmple(D4 a, D4 b) { return {_mm256_cmp_pd(a.v, b.v, _CMP_LE_OQ)}; }
FASTPITCH_INLINE D4M mask_and(D4M a, D4M b) { return {_mm256_and_pd(a.m, b.m)}; }
FASTPITCH_INLINE int movemask(D4M a) { return _mm256_movemask_pd(a.m); }
FASTPITCH_INLINE D4 operator/(D4 a, D4 b) { return {_mm256_div_pd(a.v, b.v)}; }
FASTPITCH_INLINE D4 select(D4M m, D4 a, D4 b) { return {_mm256_blendv_pd(b.v, a.v, m.m)}; }
FASTPITCH_INLINE double hmax(D4 a) {
  __m128d m = _mm_max_pd(_mm256_castpd256_pd128(a.v), _mm256_extractf128_pd(a.v, 1));
  return _mm_cvtsd_f64(_mm_max_sd(m, _mm_unpackhi_pd(m, m)));
}
FASTPITCH_INLINE double hmin(D4 a) {
  __m128d m = _mm_min_pd(_mm256_castpd256_pd128(a.v), _mm256_extractf128_pd(a.v, 1));
  return _mm_cvtsd_f64(_mm_min_sd(m, _mm_unpackhi_pd(m, m)));
}
FASTPITCH_INLINE void store_i32(std::int32_t* p, D4 a) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(p), _mm256_cvttpd_epi32(a.v));
}
FASTPITCH_INLINE void store_u16(std::uint16_t* p, D4 a) {
  const __m128i i = _mm256_cvttpd_epi32(a.v);
  _mm_storel_epi64(reinterpret_cast<__m128i*>(p), _mm_packus_epi32(i, i));
}
FASTPITCH_INLINE D4 round_nearest(D4 a) {
  return {_mm256_round_pd(a.v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)};
}
FASTPITCH_INLINE D4 bit_and(D4 a, D4 b) { return {_mm256_and_pd(a.v, b.v)}; }
FASTPITCH_INLINE D4 bit_or(D4 a, D4 b) { return {_mm256_or_pd(a.v, b.v)}; }

template <class T> struct native;
template <> struct native<float> { using type = F8; };
template <> struct native<double> { using type = D4; };
inline const char* isa_name() { return "AVX2+FMA"; }

#elif defined(FASTPITCH_SIMD_SSE2)
struct F4 {
  using value_type = float;
  static constexpr int size = 4;
  __m128 v;
  static FASTPITCH_INLINE F4 load(const float* p) { return {_mm_load_ps(p)}; }
  static FASTPITCH_INLINE F4 loadu(const float* p) { return {_mm_loadu_ps(p)}; }
  static FASTPITCH_INLINE F4 set1(float x) { return {_mm_set1_ps(x)}; }
  static FASTPITCH_INLINE F4 iota() { return {_mm_setr_ps(0, 1, 2, 3)}; }
  FASTPITCH_INLINE void store(float* p) const { _mm_store_ps(p, v); }
  FASTPITCH_INLINE void storeu(float* p) const { _mm_storeu_ps(p, v); }
  static FASTPITCH_INLINE void transpose(const float* const* rows, std::size_t col, float* out) {
    __m128 r0 = _mm_loadu_ps(rows[0] + col), r1 = _mm_loadu_ps(rows[1] + col);
    __m128 r2 = _mm_loadu_ps(rows[2] + col), r3 = _mm_loadu_ps(rows[3] + col);
    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
    _mm_store_ps(out + 0, r0);
    _mm_store_ps(out + 4, r1);
    _mm_store_ps(out + 8, r2);
    _mm_store_ps(out + 12, r3);
  }
};
struct F4M { __m128 m; };
FASTPITCH_INLINE F4 operator+(F4 a, F4 b) { return {_mm_add_ps(a.v, b.v)}; }
FASTPITCH_INLINE F4 operator-(F4 a, F4 b) { return {_mm_sub_ps(a.v, b.v)}; }
FASTPITCH_INLINE F4 operator*(F4 a, F4 b) { return {_mm_mul_ps(a.v, b.v)}; }
FASTPITCH_INLINE F4 fmadd(F4 a, F4 b, F4 c) { return {_mm_add_ps(_mm_mul_ps(a.v, b.v), c.v)}; }
FASTPITCH_INLINE F4 fmsub(F4 a, F4 b, F4 c) { return {_mm_sub_ps(_mm_mul_ps(a.v, b.v), c.v)}; }
FASTPITCH_INLINE F4 fnmadd(F4 a, F4 b, F4 c) { return {_mm_sub_ps(c.v, _mm_mul_ps(a.v, b.v))}; }
FASTPITCH_INLINE F4 vmax(F4 a, F4 b) { return {_mm_max_ps(a.v, b.v)}; }
FASTPITCH_INLINE F4 vmin(F4 a, F4 b) { return {_mm_min_ps(a.v, b.v)}; }
FASTPITCH_INLINE F4M cmpgt(F4 a, F4 b) { return {_mm_cmpgt_ps(a.v, b.v)}; }
FASTPITCH_INLINE F4M cmple(F4 a, F4 b) { return {_mm_cmple_ps(a.v, b.v)}; }
FASTPITCH_INLINE F4M mask_and(F4M a, F4M b) { return {_mm_and_ps(a.m, b.m)}; }
FASTPITCH_INLINE int movemask(F4M a) { return _mm_movemask_ps(a.m); }
FASTPITCH_INLINE F4 operator/(F4 a, F4 b) { return {_mm_div_ps(a.v, b.v)}; }
FASTPITCH_INLINE F4 select(F4M m, F4 a, F4 b) {
  return {_mm_or_ps(_mm_and_ps(m.m, a.v), _mm_andnot_ps(m.m, b.v))};
}
FASTPITCH_INLINE float hmax(F4 a) {
  __m128 m = _mm_max_ps(a.v, _mm_movehl_ps(a.v, a.v));
  m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
  return _mm_cvtss_f32(m);
}
FASTPITCH_INLINE float hmin(F4 a) {
  __m128 m = _mm_min_ps(a.v, _mm_movehl_ps(a.v, a.v));
  m = _mm_min_ss(m, _mm_shuffle_ps(m, m, 1));
  return _mm_cvtss_f32(m);
}
FASTPITCH_INLINE void store_i32(std::int32_t* p, F4 a) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(p), _mm_cvttps_epi32(a.v));
}

struct D2 {
  using value_type = double;
  static constexpr int size = 2;
  __m128d v;
  static FASTPITCH_INLINE D2 load(const double* p) { return {_mm_load_pd(p)}; }
  static FASTPITCH_INLINE D2 loadu(const double* p) { return {_mm_loadu_pd(p)}; }
  static FASTPITCH_INLINE D2 set1(double x) { return {_mm_set1_pd(x)}; }
  static FASTPITCH_INLINE D2 iota() { return {_mm_setr_pd(0, 1)}; }
  FASTPITCH_INLINE void store(double* p) const { _mm_store_pd(p, v); }
  FASTPITCH_INLINE void storeu(double* p) const { _mm_storeu_pd(p, v); }
  static FASTPITCH_INLINE void transpose(const double* const* rows, std::size_t col, double* out) {
    __m128d r0 = _mm_loadu_pd(rows[0] + col), r1 = _mm_loadu_pd(rows[1] + col);
    _mm_store_pd(out + 0, _mm_unpacklo_pd(r0, r1));
    _mm_store_pd(out + 2, _mm_unpackhi_pd(r0, r1));
  }
};
struct D2M { __m128d m; };
FASTPITCH_INLINE D2 operator+(D2 a, D2 b) { return {_mm_add_pd(a.v, b.v)}; }
FASTPITCH_INLINE D2 operator-(D2 a, D2 b) { return {_mm_sub_pd(a.v, b.v)}; }
FASTPITCH_INLINE D2 operator*(D2 a, D2 b) { return {_mm_mul_pd(a.v, b.v)}; }
FASTPITCH_INLINE D2 fmadd(D2 a, D2 b, D2 c) { return {_mm_add_pd(_mm_mul_pd(a.v, b.v), c.v)}; }
FASTPITCH_INLINE D2 fmsub(D2 a, D2 b, D2 c) { return {_mm_sub_pd(_mm_mul_pd(a.v, b.v), c.v)}; }
FASTPITCH_INLINE D2 fnmadd(D2 a, D2 b, D2 c) { return {_mm_sub_pd(c.v, _mm_mul_pd(a.v, b.v))}; }
FASTPITCH_INLINE D2 vmax(D2 a, D2 b) { return {_mm_max_pd(a.v, b.v)}; }
FASTPITCH_INLINE D2 vmin(D2 a, D2 b) { return {_mm_min_pd(a.v, b.v)}; }
FASTPITCH_INLINE D2M cmpgt(D2 a, D2 b) { return {_mm_cmpgt_pd(a.v, b.v)}; }
FASTPITCH_INLINE D2M cmple(D2 a, D2 b) { return {_mm_cmple_pd(a.v, b.v)}; }
FASTPITCH_INLINE D2M mask_and(D2M a, D2M b) { return {_mm_and_pd(a.m, b.m)}; }
FASTPITCH_INLINE int movemask(D2M a) { return _mm_movemask_pd(a.m); }
FASTPITCH_INLINE D2 operator/(D2 a, D2 b) { return {_mm_div_pd(a.v, b.v)}; }
FASTPITCH_INLINE D2 select(D2M m, D2 a, D2 b) {
  return {_mm_or_pd(_mm_and_pd(m.m, a.v), _mm_andnot_pd(m.m, b.v))};
}
FASTPITCH_INLINE double hmax(D2 a) { return _mm_cvtsd_f64(_mm_max_sd(a.v, _mm_unpackhi_pd(a.v, a.v))); }
FASTPITCH_INLINE double hmin(D2 a) { return _mm_cvtsd_f64(_mm_min_sd(a.v, _mm_unpackhi_pd(a.v, a.v))); }
FASTPITCH_INLINE void store_i32(std::int32_t* p, D2 a) {
  _mm_storel_epi64(reinterpret_cast<__m128i*>(p), _mm_cvttpd_epi32(a.v));
}
FASTPITCH_INLINE void store_u16(std::uint16_t* p, D2 a) {
  const __m128i i = _mm_cvttpd_epi32(a.v);
  p[0] = std::uint16_t(_mm_cvtsi128_si32(i));
  p[1] = std::uint16_t(_mm_cvtsi128_si32(_mm_srli_si128(i, 4)));
}
// Round to nearest-even for |x| < 2^51 (sufficient for all uses here).
FASTPITCH_INLINE D2 round_nearest(D2 a) {
  const __m128d m = _mm_set1_pd(6755399441055744.0);
  __m128d t = _mm_add_pd(a.v, m);
#if defined(__GNUC__) || defined(__clang__)
  __asm__("" : "+x"(t));  // opaque: (x + M) - M must not be simplified to x
#endif
  return {_mm_sub_pd(t, m)};
}
FASTPITCH_INLINE D2 bit_and(D2 a, D2 b) { return {_mm_and_pd(a.v, b.v)}; }
FASTPITCH_INLINE D2 bit_or(D2 a, D2 b) { return {_mm_or_pd(a.v, b.v)}; }

template <class T> struct native;
template <> struct native<float> { using type = F4; };
template <> struct native<double> { using type = D2; };
inline const char* isa_name() { return "SSE2"; }

#elif defined(FASTPITCH_SIMD_NEON)
struct F4 {
  using value_type = float;
  static constexpr int size = 4;
  float32x4_t v;
  static FASTPITCH_INLINE F4 load(const float* p) { return {vld1q_f32(p)}; }
  static FASTPITCH_INLINE F4 loadu(const float* p) { return {vld1q_f32(p)}; }
  static FASTPITCH_INLINE F4 set1(float x) { return {vdupq_n_f32(x)}; }
  static FASTPITCH_INLINE F4 iota() {
    static const float k[4] = {0, 1, 2, 3};
    return {vld1q_f32(k)};
  }
  FASTPITCH_INLINE void store(float* p) const { vst1q_f32(p, v); }
  FASTPITCH_INLINE void storeu(float* p) const { vst1q_f32(p, v); }
  static FASTPITCH_INLINE void transpose(const float* const* rows, std::size_t col, float* out) {
    float32x4x2_t p01 = vtrnq_f32(vld1q_f32(rows[0] + col), vld1q_f32(rows[1] + col));
    float32x4x2_t p23 = vtrnq_f32(vld1q_f32(rows[2] + col), vld1q_f32(rows[3] + col));
    vst1q_f32(out + 0, vcombine_f32(vget_low_f32(p01.val[0]), vget_low_f32(p23.val[0])));
    vst1q_f32(out + 4, vcombine_f32(vget_low_f32(p01.val[1]), vget_low_f32(p23.val[1])));
    vst1q_f32(out + 8, vcombine_f32(vget_high_f32(p01.val[0]), vget_high_f32(p23.val[0])));
    vst1q_f32(out + 12, vcombine_f32(vget_high_f32(p01.val[1]), vget_high_f32(p23.val[1])));
  }
};
struct F4M { uint32x4_t m; };
FASTPITCH_INLINE F4 operator+(F4 a, F4 b) { return {vaddq_f32(a.v, b.v)}; }
FASTPITCH_INLINE F4 operator-(F4 a, F4 b) { return {vsubq_f32(a.v, b.v)}; }
FASTPITCH_INLINE F4 operator*(F4 a, F4 b) { return {vmulq_f32(a.v, b.v)}; }
FASTPITCH_INLINE F4 fmadd(F4 a, F4 b, F4 c) { return {vfmaq_f32(c.v, a.v, b.v)}; }
FASTPITCH_INLINE F4 fmsub(F4 a, F4 b, F4 c) { return {vnegq_f32(vfmsq_f32(c.v, a.v, b.v))}; }
FASTPITCH_INLINE F4 fnmadd(F4 a, F4 b, F4 c) { return {vfmsq_f32(c.v, a.v, b.v)}; }
FASTPITCH_INLINE F4 vmax(F4 a, F4 b) { return {vmaxq_f32(a.v, b.v)}; }
FASTPITCH_INLINE F4 vmin(F4 a, F4 b) { return {vminq_f32(a.v, b.v)}; }
FASTPITCH_INLINE F4M cmpgt(F4 a, F4 b) { return {vcgtq_f32(a.v, b.v)}; }
FASTPITCH_INLINE F4M cmple(F4 a, F4 b) { return {vcleq_f32(a.v, b.v)}; }
FASTPITCH_INLINE F4M mask_and(F4M a, F4M b) { return {vandq_u32(a.m, b.m)}; }
FASTPITCH_INLINE int movemask(F4M a) {
  static const uint32_t w[4] = {1, 2, 4, 8};
  return int(vaddvq_u32(vandq_u32(a.m, vld1q_u32(w))));
}
FASTPITCH_INLINE F4 operator/(F4 a, F4 b) { return {vdivq_f32(a.v, b.v)}; }
FASTPITCH_INLINE F4 select(F4M m, F4 a, F4 b) { return {vbslq_f32(m.m, a.v, b.v)}; }
FASTPITCH_INLINE float hmax(F4 a) { return vmaxvq_f32(a.v); }
FASTPITCH_INLINE float hmin(F4 a) { return vminvq_f32(a.v); }
FASTPITCH_INLINE void store_i32(std::int32_t* p, F4 a) { vst1q_s32(p, vcvtq_s32_f32(a.v)); }

struct D2 {
  using value_type = double;
  static constexpr int size = 2;
  float64x2_t v;
  static FASTPITCH_INLINE D2 load(const double* p) { return {vld1q_f64(p)}; }
  static FASTPITCH_INLINE D2 loadu(const double* p) { return {vld1q_f64(p)}; }
  static FASTPITCH_INLINE D2 set1(double x) { return {vdupq_n_f64(x)}; }
  static FASTPITCH_INLINE D2 iota() {
    static const double k[2] = {0, 1};
    return {vld1q_f64(k)};
  }
  FASTPITCH_INLINE void store(double* p) const { vst1q_f64(p, v); }
  FASTPITCH_INLINE void storeu(double* p) const { vst1q_f64(p, v); }
  static FASTPITCH_INLINE void transpose(const double* const* rows, std::size_t col, double* out) {
    float64x2_t r0 = vld1q_f64(rows[0] + col), r1 = vld1q_f64(rows[1] + col);
    vst1q_f64(out + 0, vzip1q_f64(r0, r1));
    vst1q_f64(out + 2, vzip2q_f64(r0, r1));
  }
};
struct D2M { uint64x2_t m; };
FASTPITCH_INLINE D2 operator+(D2 a, D2 b) { return {vaddq_f64(a.v, b.v)}; }
FASTPITCH_INLINE D2 operator-(D2 a, D2 b) { return {vsubq_f64(a.v, b.v)}; }
FASTPITCH_INLINE D2 operator*(D2 a, D2 b) { return {vmulq_f64(a.v, b.v)}; }
FASTPITCH_INLINE D2 fmadd(D2 a, D2 b, D2 c) { return {vfmaq_f64(c.v, a.v, b.v)}; }
FASTPITCH_INLINE D2 fmsub(D2 a, D2 b, D2 c) { return {vnegq_f64(vfmsq_f64(c.v, a.v, b.v))}; }
FASTPITCH_INLINE D2 fnmadd(D2 a, D2 b, D2 c) { return {vfmsq_f64(c.v, a.v, b.v)}; }
FASTPITCH_INLINE D2 vmax(D2 a, D2 b) { return {vmaxq_f64(a.v, b.v)}; }
FASTPITCH_INLINE D2 vmin(D2 a, D2 b) { return {vminq_f64(a.v, b.v)}; }
FASTPITCH_INLINE D2M cmpgt(D2 a, D2 b) { return {vcgtq_f64(a.v, b.v)}; }
FASTPITCH_INLINE D2M cmple(D2 a, D2 b) { return {vcleq_f64(a.v, b.v)}; }
FASTPITCH_INLINE D2M mask_and(D2M a, D2M b) { return {vandq_u64(a.m, b.m)}; }
FASTPITCH_INLINE int movemask(D2M a) {
  return int((vgetq_lane_u64(a.m, 0) & 1) | ((vgetq_lane_u64(a.m, 1) & 1) << 1));
}
FASTPITCH_INLINE D2 operator/(D2 a, D2 b) { return {vdivq_f64(a.v, b.v)}; }
FASTPITCH_INLINE D2 select(D2M m, D2 a, D2 b) { return {vbslq_f64(m.m, a.v, b.v)}; }
FASTPITCH_INLINE double hmax(D2 a) { return vmaxvq_f64(a.v); }
FASTPITCH_INLINE double hmin(D2 a) { return vminvq_f64(a.v); }
FASTPITCH_INLINE void store_i32(std::int32_t* p, D2 a) {
  vst1_s32(p, vmovn_s64(vcvtq_s64_f64(a.v)));
}
FASTPITCH_INLINE D2 round_nearest(D2 a) { return {vrndnq_f64(a.v)}; }
FASTPITCH_INLINE void store_u16(std::uint16_t* p, D2 a) {
  p[0] = std::uint16_t(vgetq_lane_f64(a.v, 0));
  p[1] = std::uint16_t(vgetq_lane_f64(a.v, 1));
}
FASTPITCH_INLINE D2 bit_and(D2 a, D2 b) {
  return {vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(a.v), vreinterpretq_u64_f64(b.v)))};
}
FASTPITCH_INLINE D2 bit_or(D2 a, D2 b) {
  return {vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(a.v), vreinterpretq_u64_f64(b.v)))};
}

template <class T> struct native;
template <> struct native<float> { using type = F4; };
template <> struct native<double> { using type = D2; };
inline const char* isa_name() { return "NEON"; }

#else
template <class T> struct native { using type = Scalar<T>; };
inline const char* isa_name() { return "scalar"; }
#endif

template <class T>
using native_t = typename native<T>::type;

}  // namespace simd

// ---------------------------------------------------------------------------------------
// Lane-batched complex FFT (Stockham autosort, radices 2/3/4/5).
//
// W = V::size independent sequences are stored "vertically": element e of all sequences
// occupies 2*W consecutive scalars, real parts of the W lanes followed by imaginary parts.
// ---------------------------------------------------------------------------------------

template <class V>
class VecFFT {
 public:
  using T = typename V::value_type;
  static constexpr int W = V::size;
  static constexpr std::size_t E = 2 * W;  // scalars per vertical complex element

  static bool factorizable(int n) {
    if (n < 1) return false;
    for (int r : {2, 3, 5})
      while (n % r == 0) n /= r;
    return n == 1;
  }

  /// Relative cost estimate of a size-n transform (used to choose the padded length).
  static double cost(int n) {
    double c = 0;
    int m = n;
    while (m % 4 == 0) { c += 1.3; m /= 4; }
    while (m % 2 == 0) { c += 1.0; m /= 2; }
    while (m % 3 == 0) { c += 1.35; m /= 3; }
    while (m % 5 == 0) { c += 1.6; m /= 5; }
    return c * n;
  }

  /// Plans a size-n transform. Transforms whose working set exceeds the L1 cache are split
  /// as n = R * C (four-step): R-point column FFTs on gathered scratch, twiddles, then
  /// C-point row FFTs; this touches the large buffers twice instead of once per radix pass.
  void init(int n, bool allow_blocking = true) {
    init_flat(n);
    blocked_ = false;
    const std::size_t bytes = std::size_t(n) * E * sizeof(T);
    if (!allow_blocking || bytes <= 24 * 1024) return;
    int bestR = 0;
    double best = 0;
    for (int R = 8; R * 8 <= n; ++R) {
      if (n % R) continue;
      const int C = n / R;
      if (!factorizable(R) || !factorizable(C)) continue;
      const double c = cost(R) * C + cost(C) * R + 0.6 * n;  // + twiddle/copy overhead
      const double imb = std::abs(std::log(double(R) / C));     // prefer balanced splits
      const double score = c * (1.0 + 0.05 * imb);
      if (!bestR || score < best) { bestR = R; best = score; }
    }
    if (!bestR) return;
    blocked_ = true;
    R_ = bestR;
    C_ = n / bestR;
    col_.reset(new VecFFT());
    row_.reset(new VecFFT());
    col_->init(R_, false);
    row_->init(C_, false);
    scratch_.resize(std::size_t(std::max(R_, C_)) * 2 * E);
    tw4_.resize(std::size_t(C_) * R_ * 2);
    const double pi = 3.14159265358979323846;
    for (int n2 = 0; n2 < C_; ++n2)
      for (int k1 = 0; k1 < R_; ++k1) {
        const double a = -2.0 * pi * double(n2) * double(k1) / double(n);
        tw4_[(std::size_t(n2) * R_ + k1) * 2] = T(std::cos(a));
        tw4_[(std::size_t(n2) * R_ + k1) * 2 + 1] = T(std::sin(a));
      }
  }

  void init_flat(int n) {
    if (!factorizable(n)) throw std::invalid_argument("fastpitch: FFT size must be 2^a 3^b 5^c");
    n_ = n;
    stages_.clear();
    tw_.clear();
    std::vector<int> radices;
    int m = n;
    while (m % 4 == 0) { radices.push_back(4); m /= 4; }
    while (m % 2 == 0) { radices.push_back(2); m /= 2; }
    while (m % 3 == 0) { radices.push_back(3); m /= 3; }
    while (m % 5 == 0) { radices.push_back(5); m /= 5; }
    int l = n, s = 1;
    const double pi = 3.14159265358979323846;
    for (int r : radices) {
      Stage st;
      st.radix = r;
      st.m = l / r;
      st.s = s;
      st.tw = tw_.size();
      for (int p = 1; p < st.m; ++p)
        for (int k = 1; k < r; ++k) {
          const double a = -2.0 * pi * double(k) * double(p) / double(l);
          tw_.push_back(T(std::cos(a)));
          tw_.push_back(T(std::sin(a)));
        }
      stages_.push_back(st);
      l = st.m;
      s *= r;
    }
  }

  int size() const { return n_; }

  /// Forward DFT (sign -1, unnormalised). x holds the input; x and y are used as ping-pong
  /// buffers. Returns the buffer containing the result.
  T* forward(T* x, T* y) const {
    if (blocked_) return forward_blocked(x, y);
    return forward_flat(x, y);
  }

 private:
  // X[k1 + R k2] = sum_n2 W_C^(n2 k2) [ W_n^(n2 k1) sum_n1 x[C n1 + n2] W_R^(n1 k1) ]
  T* forward_blocked(T* x, T* y) const {
    const int R = R_, C = C_;
    T* s0 = scratch_.data();
    T* s1 = s0 + std::size_t(std::max(R, C)) * E;
    for (int n2 = 0; n2 < C; ++n2) {
      for (int n1 = 0; n1 < R; ++n1) {
        const T* src = x + (std::size_t(C) * n1 + n2) * E;
        T* dst = s0 + std::size_t(n1) * E;
        V::load(src).store(dst);
        V::load(src + W).store(dst + W);
        for (int q = 2 * V::size; q < int(E); q += V::size) V::load(src + q).store(dst + q);
      }
      const T* res = col_->forward_flat(s0, s1);
      const T* tw = tw4_.data() + std::size_t(n2) * R * 2;
      for (int k1 = 0; k1 < R; ++k1) {
        const T* sp = res + std::size_t(k1) * E;
        T* dst = x + (std::size_t(C) * k1 + n2) * E;
        V re = V::load(sp), im = V::load(sp + W);
        if (n2 && k1) cmul(re, im, V::set1(tw[2 * k1]), V::set1(tw[2 * k1 + 1]));
        re.store(dst);
        im.store(dst + W);
      }
    }
    for (int k1 = 0; k1 < R; ++k1) {
      T* row = x + std::size_t(C) * k1 * E;
      const T* res = row_->forward_flat(row, s0);
      for (int k2 = 0; k2 < C; ++k2) {
        const T* sp = res + std::size_t(k2) * E;
        T* dst = y + (std::size_t(k1) + std::size_t(R) * k2) * E;
        V::load(sp).store(dst);
        V::load(sp + W).store(dst + W);
      }
    }
    return y;
  }

 public:
  T* forward_flat(T* x, T* y) const {
    for (const Stage& st : stages_) {
      const T* tw = tw_.data() + st.tw;
      switch (st.radix) {
        case 4: stage4(x, y, st.m, st.s, tw); break;
        case 2: stage2(x, y, st.m, st.s, tw); break;
        case 3: stage3(x, y, st.m, st.s, tw); break;
        default: stage5(x, y, st.m, st.s, tw); break;
      }
      std::swap(x, y);
    }
    return x;
  }

 private:
  struct Stage {
    int radix, m, s;
    std::size_t tw;
  };

  static FASTPITCH_INLINE void ld(const T* p, V& re, V& im) {
    re = V::load(p);
    im = V::load(p + W);
  }
  static FASTPITCH_INLINE void st(T* p, V re, V im) {
    re.store(p);
    im.store(p + W);
  }
  // (re, im) *= (wr, wi)
  static FASTPITCH_INLINE void cmul(V& re, V& im, V wr, V wi) {
    const V r = fmsub(re, wr, im * wi);
    im = fmadd(re, wi, im * wr);
    re = r;
  }

  static void stage2(const T* x, T* y, int m, int s, const T* tw) {
    const std::size_t S = std::size_t(s) * E, half = std::size_t(m) * S;
    for (int p = 0; p < m; ++p) {
      const T* xp = x + std::size_t(p) * S;
      T* yp = y + std::size_t(2 * p) * S;
      if (p == 0) {
        for (std::size_t q = 0; q < S; q += E) {
          V ar, ai, br, bi;
          ld(xp + q, ar, ai);
          ld(xp + q + half, br, bi);
          st(yp + q, ar + br, ai + bi);
          st(yp + q + S, ar - br, ai - bi);
        }
      } else {
        const V wr = V::set1(tw[2 * (p - 1)]), wi = V::set1(tw[2 * (p - 1) + 1]);
        for (std::size_t q = 0; q < S; q += E) {
          V ar, ai, br, bi;
          ld(xp + q, ar, ai);
          ld(xp + q + half, br, bi);
          st(yp + q, ar + br, ai + bi);
          V dr = ar - br, di = ai - bi;
          cmul(dr, di, wr, wi);
          st(yp + q + S, dr, di);
        }
      }
    }
  }

  static FASTPITCH_INLINE void bfly4(V a0r, V a0i, V a1r, V a1i, V a2r, V a2i, V a3r, V a3i,
                                     V& b0r, V& b0i, V& b1r, V& b1i, V& b2r, V& b2i, V& b3r,
                                     V& b3i) {
    const V t0r = a0r + a2r, t0i = a0i + a2i;
    const V t1r = a0r - a2r, t1i = a0i - a2i;
    const V t2r = a1r + a3r, t2i = a1i + a3i;
    const V t3r = a1r - a3r, t3i = a1i - a3i;
    b0r = t0r + t2r; b0i = t0i + t2i;
    b2r = t0r - t2r; b2i = t0i - t2i;
    b1r = t1r + t3i; b1i = t1i - t3r;  // t1 - i*t3
    b3r = t1r - t3i; b3i = t1i + t3r;  // t1 + i*t3
  }

  static void stage4(const T* x, T* y, int m, int s, const T* tw) {
    const std::size_t S = std::size_t(s) * E, Q = std::size_t(m) * S;
    for (int p = 0; p < m; ++p) {
      const T* xp = x + std::size_t(p) * S;
      T* yp = y + std::size_t(4 * p) * S;
      if (p == 0) {
        for (std::size_t q = 0; q < S; q += E) {
          V a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i;
          ld(xp + q, a0r, a0i);
          ld(xp + q + Q, a1r, a1i);
          ld(xp + q + 2 * Q, a2r, a2i);
          ld(xp + q + 3 * Q, a3r, a3i);
          V b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i;
          bfly4(a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i, b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i);
          st(yp + q, b0r, b0i);
          st(yp + q + S, b1r, b1i);
          st(yp + q + 2 * S, b2r, b2i);
          st(yp + q + 3 * S, b3r, b3i);
        }
      } else {
        const T* w = tw + std::size_t(p - 1) * 6;
        const V w1r = V::set1(w[0]), w1i = V::set1(w[1]);
        const V w2r = V::set1(w[2]), w2i = V::set1(w[3]);
        const V w3r = V::set1(w[4]), w3i = V::set1(w[5]);
        for (std::size_t q = 0; q < S; q += E) {
          V a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i;
          ld(xp + q, a0r, a0i);
          ld(xp + q + Q, a1r, a1i);
          ld(xp + q + 2 * Q, a2r, a2i);
          ld(xp + q + 3 * Q, a3r, a3i);
          V b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i;
          bfly4(a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i, b0r, b0i, b1r, b1i, b2r, b2i, b3r, b3i);
          cmul(b1r, b1i, w1r, w1i);
          cmul(b2r, b2i, w2r, w2i);
          cmul(b3r, b3i, w3r, w3i);
          st(yp + q, b0r, b0i);
          st(yp + q + S, b1r, b1i);
          st(yp + q + 2 * S, b2r, b2i);
          st(yp + q + 3 * S, b3r, b3i);
        }
      }
    }
  }

  static void stage3(const T* x, T* y, int m, int s, const T* tw) {
    const std::size_t S = std::size_t(s) * E, Q = std::size_t(m) * S;
    const V half = V::set1(T(0.5));
    const V sn = V::set1(T(0.86602540378443864676));  // sin(2 pi / 3)
    for (int p = 0; p < m; ++p) {
      const T* xp = x + std::size_t(p) * S;
      T* yp = y + std::size_t(3 * p) * S;
      const T* w = tw + std::size_t(p > 0 ? p - 1 : 0) * 4;
      const V w1r = V::set1(p ? w[0] : T(1)), w1i = V::set1(p ? w[1] : T(0));
      const V w2r = V::set1(p ? w[2] : T(1)), w2i = V::set1(p ? w[3] : T(0));
      for (std::size_t q = 0; q < S; q += E) {
        V a0r, a0i, a1r, a1i, a2r, a2i;
        ld(xp + q, a0r, a0i);
        ld(xp + q + Q, a1r, a1i);
        ld(xp + q + 2 * Q, a2r, a2i);
        const V t1r = a1r + a2r, t1i = a1i + a2i;
        const V t2r = fnmadd(half, t1r, a0r), t2i = fnmadd(half, t1i, a0i);
        // t3 = -i * sin(2pi/3) * (a1 - a2)
        const V t3r = sn * (a1i - a2i), t3i = sn * (a2r - a1r);
        V b1r = t2r + t3r, b1i = t2i + t3i;
        V b2r = t2r - t3r, b2i = t2i - t3i;
        st(yp + q, a0r + t1r, a0i + t1i);
        if (p) {
          cmul(b1r, b1i, w1r, w1i);
          cmul(b2r, b2i, w2r, w2i);
        }
        st(yp + q + S, b1r, b1i);
        st(yp + q + 2 * S, b2r, b2i);
      }
    }
  }

  static void stage5(const T* x, T* y, int m, int s, const T* tw) {
    const std::size_t S = std::size_t(s) * E, Q = std::size_t(m) * S;
    const V c1 = V::set1(T(0.30901699437494742410));   // cos(2pi/5)
    const V c2 = V::set1(T(-0.80901699437494742410));  // cos(4pi/5)
    const V s1 = V::set1(T(0.95105651629515357212));   // sin(2pi/5)
    const V s2 = V::set1(T(0.58778525229247312917));   // sin(4pi/5)
    for (int p = 0; p < m; ++p) {
      const T* xp = x + std::size_t(p) * S;
      T* yp = y + std::size_t(5 * p) * S;
      const T* w = tw + std::size_t(p > 0 ? p - 1 : 0) * 8;
      V wr[4], wi[4];
      for (int k = 0; k < 4; ++k) {
        wr[k] = V::set1(p ? w[2 * k] : T(1));
        wi[k] = V::set1(p ? w[2 * k + 1] : T(0));
      }
      for (std::size_t q = 0; q < S; q += E) {
        V a0r, a0i, a1r, a1i, a2r, a2i, a3r, a3i, a4r, a4i;
        ld(xp + q, a0r, a0i);
        ld(xp + q + Q, a1r, a1i);
        ld(xp + q + 2 * Q, a2r, a2i);
        ld(xp + q + 3 * Q, a3r, a3i);
        ld(xp + q + 4 * Q, a4r, a4i);
        const V t1r = a1r + a4r, t1i = a1i + a4i;
        const V t2r = a2r + a3r, t2i = a2i + a3i;
        const V t3r = a1r - a4r, t3i = a1i - a4i;
        const V t4r = a2r - a3r, t4i = a2i - a3i;
        const V u1r = fmadd(c2, t2r, fmadd(c1, t1r, a0r)), u1i = fmadd(c2, t2i, fmadd(c1, t1i, a0i));
        const V u2r = fmadd(c1, t2r, fmadd(c2, t1r, a0r)), u2i = fmadd(c1, t2i, fmadd(c2, t1i, a0i));
        const V v1r = fmadd(s2, t4r, s1 * t3r), v1i = fmadd(s2, t4i, s1 * t3i);
        const V v2r = fnmadd(s1, t4r, s2 * t3r), v2i = fnmadd(s1, t4i, s2 * t3i);
        // b1 = u1 - i v1, b4 = u1 + i v1, b2 = u2 - i v2, b3 = u2 + i v2
        V br[4] = {u1r + v1i, u2r + v2i, u2r - v2i, u1r - v1i};
        V bi[4] = {u1i - v1r, u2i - v2r, u2i + v2r, u1i + v1r};
        st(yp + q, a0r + t1r + t2r, a0i + t1i + t2i);
        for (int k = 0; k < 4; ++k) {
          if (p) cmul(br[k], bi[k], wr[k], wi[k]);
          st(yp + q + std::size_t(k + 1) * S, br[k], bi[k]);
        }
      }
    }
  }

  int n_ = 0;
  std::vector<Stage> stages_;
  std::vector<T> tw_;
  bool blocked_ = false;
  int R_ = 0, C_ = 0;
  std::unique_ptr<VecFFT> col_, row_;
  mutable AlignedBuffer<T> scratch_;
  std::vector<T> tw4_;
};

// ---------------------------------------------------------------------------------------
// Special functions
// ---------------------------------------------------------------------------------------

inline double betacf(double a, double b, double x) {
  const double kEps = 1e-16, kFpMin = 1e-300;
  const double qab = a + b, qap = a + 1.0, qam = a - 1.0;
  double c = 1.0, d = 1.0 - qab * x / qap;
  if (std::fabs(d) < kFpMin) d = kFpMin;
  d = 1.0 / d;
  double h = d;
  for (int m = 1; m <= 10000; ++m) {
    const double m2 = 2.0 * m;
    double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (std::fabs(d) < kFpMin) d = kFpMin;
    c = 1.0 + aa / c;
    if (std::fabs(c) < kFpMin) c = kFpMin;
    d = 1.0 / d;
    h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (std::fabs(d) < kFpMin) d = kFpMin;
    c = 1.0 + aa / c;
    if (std::fabs(c) < kFpMin) c = kFpMin;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1.0) < kEps) break;
  }
  return h;
}

/// Regularised incomplete beta function I_x(a, b) (CDF of Beta(a, b)).
inline double beta_cdf(double x, double a, double b) {
  if (!(x > 0.0)) return 0.0;
  if (x >= 1.0) return 1.0;
  const double lbt = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) +
                     b * std::log1p(-x);
  const double bt = std::exp(lbt);
  if (x < (a + 1.0) / (a + b + 2.0)) return bt * betacf(a, b, x) / a;
  return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

// ---------------------------------------------------------------------------------------
// Shared, immutable analysis model (derived from PyinOptions)
// ---------------------------------------------------------------------------------------

struct Cand {
  std::int32_t bin;
  double logp;  ///< log(p + tiny)
  double f0;    ///< refined candidate frequency (Hz) before quantisation to the bin
};

struct FrameObs {
  double voiced_prob;
  double log_unvoiced;  ///< log((1 - voiced_prob) / n_bins + tiny)
  std::uint32_t cbegin, cend;  ///< candidate range (bins strictly decreasing)
};

struct Model {
  PyinOptions opt;
  int N = 0, hop = 0;
  int pmin = 0, pmax = 0, K = 0;  // period range and number of YIN lags kept
  int L = 0, M = 0;               // linear-correlation FFT length (real) and M = L / 2
  int nthr = 0;
  std::vector<double> thr;        // thresholds[0..nthr]
  std::vector<double> beta;       // beta[k] = mass of threshold k (k = 1..nthr)
  std::vector<double> cumbeta;    // cumbeta[k] = sum_{j<=k} beta[j]
  std::vector<double> tailbeta;   // tailbeta[k] = sum_{j>=k} beta[j] (accurate for tiny tails)
  std::vector<double> bz_exp;     // exp(-lambda * pos)
  std::vector<double> bz_fact;    // (1 - exp(-lambda)) / (1 - exp(-lambda * n))
  int nbps = 0, nbins = 0;
  double bin_scale = 0;           // 12 * nbps
  double tiny = std::numeric_limits<double>::min();
  double log_tiny = 0;
  std::vector<double> freqs;      // centre frequency of each pitch bin

  // Transition model (see Decoder)
  int dmin = 0, dmax = 0;         // support of the kernel in source offset d = src - dst
  std::vector<double> gk;         // log tri(-d), index d - dmin
  std::vector<double> logZ;       // log row-normaliser of each source bin
  double logZmax = 0, g0 = 0;
  double ls_stay = 0, ls_switch = 0, lpi = 0;
  bool exact = false;             // full transition matrix (log(0 + tiny) entries allowed)
  bool prunable = false;
  int s_lo = 0, s_hi = -1, c_lo = 0, c_hi = -1;  // interior allowed d ranges (same / cross)
  std::vector<int> src_s_lo, src_s_hi, src_c_lo, src_c_hi;  // per-source allowed d ranges
  struct Extra { int src, lo, hi; };
  std::vector<Extra> extra_s, extra_c;  // allowed d outside the interior ranges

  explicit Model(const PyinOptions& o) : opt(o) { build(); }

  std::int64_t frame_start(std::size_t f) const {
    const std::int64_t s = std::int64_t(f) * hop;
    return opt.center ? s - N / 2 : s;
  }

  std::size_t num_frames(std::size_t n) const {
    const std::size_t padded = opt.center ? n + 2 * std::size_t(N / 2) : n;
    if (padded < std::size_t(N)) return 0;
    return 1 + (padded - std::size_t(N)) / std::size_t(hop);
  }

 private:
  static void fail(const std::string& m) { throw std::invalid_argument("fastpitch: " + m); }

  void build() {
    const double sr = opt.sample_rate;
    N = opt.frame_length;
    hop = opt.hop_length > 0 ? opt.hop_length : N / 4;
    if (!(sr > 0)) fail("sample_rate must be positive");
    if (N < 4) fail("frame_length too small");
    if (hop < 1) fail("hop_length must be positive");
    if (!(opt.fmin > 0)) fail("fmin must be strictly positive");
    if (!(opt.fmin < opt.fmax)) fail("fmin must be less than fmax");
    if (opt.fmax > sr / 2) fail("fmax cannot exceed the Nyquist frequency");
    if (sr / opt.fmin >= N - 1) fail("fmin is too small for frame_length and sample_rate");
    if (opt.n_thresholds < 1) fail("n_thresholds must be positive");
    if (!(opt.resolution > 0 && opt.resolution < 1)) fail("resolution must be in (0, 1)");
    if (!(opt.switch_prob > 0 && opt.switch_prob < 1)) fail("switch_prob must be in (0, 1)");
    if (!(opt.max_transition_rate > 0)) fail("max_transition_rate must be positive");
    if (!(opt.transition_min_prob >= 0)) fail("transition_min_prob must be >= 0");

    pmin = int(std::floor(sr / opt.fmax));
    pmax = std::min(int(std::ceil(sr / opt.fmin)), N - 1);
    K = pmax - pmin + 1;
    if (pmin < 1 || K < 2) fail("invalid period range");

    // Smallest cheap FFT length giving an exact linear autocorrelation up to lag pmax.
    {
      const int need = N + pmax;
      int bestM = 0;
      double bestCost = 0;
      for (int m = (need + 1) / 2; m <= 2 * need; ++m) {
        using F = VecFFT<simd::native_t<float>>;
        if (m % 2 || !F::factorizable(m) || !F::factorizable(m / 2)) continue;
        const double c = F::cost(m) + F::cost(m / 2);
        if (!bestM || c < bestCost) { bestM = m; bestCost = c; }
        if (m > 2 * bestM && bestM) break;
      }
      M = bestM;
      L = 2 * M;
    }

    // Threshold prior (np.linspace(0, 1, n + 1) and its beta-CDF increments).
    nthr = opt.n_thresholds;
    thr.resize(nthr + 1);
    const double step = 1.0 / nthr;
    for (int k = 0; k <= nthr; ++k) thr[k] = double(k) * step;
    thr[nthr] = 1.0;
    std::vector<double> cdf(nthr + 1);
    for (int k = 0; k <= nthr; ++k) cdf[k] = beta_cdf(thr[k], opt.beta_a, opt.beta_b);
    beta.assign(nthr + 2, 0.0);
    cumbeta.assign(nthr + 2, 0.0);
    for (int k = 1; k <= nthr; ++k) {
      beta[k] = cdf[k] - cdf[k - 1];
      cumbeta[k] = cumbeta[k - 1] + beta[k];
    }
    cumbeta[nthr + 1] = cumbeta[nthr];
    tailbeta.assign(nthr + 2, 0.0);
    for (int k = nthr; k >= 1; --k) tailbeta[k] = tailbeta[k + 1] + beta[k];

    // Boltzmann prior tables: pmf(k; lambda, n) = (1-e^-l) e^(-l k) / (1-e^(-l n))
    const double lam = opt.boltzmann_parameter;
    bz_exp.resize(K + 2);
    bz_fact.resize(K + 2);
    for (int i = 0; i < K + 2; ++i) {
      bz_exp[i] = std::exp(-lam * i);
      bz_fact[i] = i ? (1.0 - std::exp(-lam)) / (1.0 - std::exp(-lam * i)) : 0.0;
    }

    // Pitch bins.
    nbps = int(std::ceil(1.0 / opt.resolution));
    bin_scale = double(12 * nbps);
    nbins = int(std::floor(bin_scale * std::log2(opt.fmax / opt.fmin))) + 1;
    if (nbins < 2) fail("pitch range too small");
    if (2 * nbins > 65535) fail("too many pitch bins");
    log_tiny = std::log(tiny);
    freqs.resize(nbins);
    for (int i = 0; i < nbins; ++i)
      freqs[i] = opt.fmin * std::pow(2.0, double(i) / double(12 * nbps));

    build_transitions();
  }

  void build_transitions() {
    const int n = nbins;
    const double ms = std::nearbyint(opt.max_transition_rate * 12.0 * hop / opt.sample_rate);
    const int width = int(ms) * nbps + 1;
    if (width > n) fail("transition width exceeds the number of pitch bins");

    // scipy.signal.windows.triang(width, sym=True)
    std::vector<double> win(width);
    {
      const int half = (width + 1) / 2;
      std::vector<double> w(half);
      for (int i = 1; i <= half; ++i)
        w[i - 1] = (width % 2 == 0) ? (2.0 * i - 1.0) / width : 2.0 * i / (width + 1.0);
      for (int i = 0; i < half; ++i) win[i] = w[i];
      if (width % 2 == 0) {
        for (int i = 0; i < half; ++i) win[half + i] = w[half - 1 - i];
      } else {
        for (int i = 0; i + 1 < half; ++i) win[half + i] = w[half - 2 - i];
      }
    }
    // librosa.sequence.transition_local: value at target offset delta = dst - src.
    const int lpad = (n - width) / 2;
    const int hw = width / 2;
    auto tri = [&](int delta) -> double {
      if (delta < -hw || delta > hw) return 0.0;
      int q = (delta - lpad - n / 2 - 1) % n;
      if (q < 0) q += n;
      return q < width ? win[q] : 0.0;
    };
    // Support in source offset d = -delta.
    int dlo = 1, dhi = -1;
    for (int d = -hw; d <= hw; ++d)
      if (tri(-d) > 0) {
        if (dlo > dhi) dlo = d;
        dhi = d;
      }
    for (int d = dlo; d <= dhi; ++d)
      if (!(tri(-d) > 0)) fail("unsupported transition window");
    dmin = dlo;
    dmax = dhi;
    gk.resize(dmax - dmin + 1);
    for (int d = dmin; d <= dmax; ++d) gk[d - dmin] = std::log(tri(-d));
    g0 = (0 >= dmin && 0 <= dmax) ? gk[-dmin] : -std::numeric_limits<double>::infinity();

    std::vector<double> Z(n, 0.0);
    logZ.resize(n);
    logZmax = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
      for (int d = dmin; d <= dmax; ++d) {
        const int j = i - d;
        if (j >= 0 && j < n) Z[i] += tri(-d);
      }
      logZ[i] = std::log(Z[i]);
      logZmax = std::max(logZmax, logZ[i]);
    }

    // librosa.sequence.transition_loop(2, 1 - switch_prob)
    const double p_stay = 1.0 - opt.switch_prob;
    const double p_switch = (1.0 - p_stay) / 1.0;
    ls_stay = std::log(p_stay);
    ls_switch = std::log(p_switch);
    lpi = std::log(1.0 / (2.0 * n) + tiny);
    exact = !(opt.transition_min_prob > 0);

    src_s_lo.assign(n, 0);
    src_s_hi.assign(n, -1);
    src_c_lo.assign(n, 0);
    src_c_hi.assign(n, -1);
    const double lthr = exact ? -std::numeric_limits<double>::infinity()
                              : std::log(opt.transition_min_prob + tiny);
    for (int i = 0; i < n; ++i) {
      for (int blk = 0; blk < 2; ++blk) {
        const double c = blk == 0 ? p_stay : p_switch;
        int lo = 1, hi = -1;
        for (int d = dmin; d <= dmax; ++d) {
          const double t = tri(-d) / Z[i];
          if (std::log(c * t + tiny) >= lthr) {
            if (lo > hi) lo = d;
            hi = d;
          }
        }
        (blk == 0 ? src_s_lo : src_c_lo)[i] = lo;
        (blk == 0 ? src_s_hi : src_c_hi)[i] = hi;
      }
    }
    // Every destination needs at least one predecessor (librosa raises otherwise).
    {
      std::vector<char> has(2 * n, 0);
      for (int i = 0; i < n; ++i)
        for (int blk = 0; blk < 2; ++blk) {
          const int lo = blk ? src_c_lo[i] : src_s_lo[i], hi = blk ? src_c_hi[i] : src_s_hi[i];
          for (int d = lo; d <= hi; ++d) {
            const int j = i - d;
            if (j < 0 || j >= n) continue;
            has[j] = 1;      // into the same voicing block of the source...
            has[n + j] = 1;  // ...or the other one (each source exists in both blocks)
          }
        }
      // Each block receives from both blocks (same from its own, cross from the other),
      // so a destination has a predecessor iff some source reaches its bin.
      for (int j = 0; j < 2 * n; ++j)
        if (!has[j]) fail("empty transition set; reduce transition_min_prob");
    }
    auto interior = [&](const std::vector<int>& lo, const std::vector<int>& hi, int& ilo,
                        int& ihi) {
      ilo = std::numeric_limits<int>::min();
      ihi = std::numeric_limits<int>::max();
      for (int i = 0; i < n; ++i) {
        ilo = std::max(ilo, lo[i]);
        ihi = std::min(ihi, hi[i]);
      }
    };
    interior(src_s_lo, src_s_hi, s_lo, s_hi);
    interior(src_c_lo, src_c_hi, c_lo, c_hi);
    auto extras = [&](const std::vector<int>& lo, const std::vector<int>& hi, int ilo, int ihi,
                      std::vector<Extra>& out) {
      out.clear();
      for (int i = 0; i < n; ++i) {
        if (lo[i] > hi[i]) continue;
        // valid destinations: j = i - d in [0, n)  <=>  d in [i - n + 1, i]
        const int vlo = std::max(lo[i], i - n + 1), vhi = std::min(hi[i], i);
        if (ilo > ihi) {
          if (vlo <= vhi) out.push_back({i, vlo, vhi});
          continue;
        }
        if (vlo < ilo) {
          const int h = std::min(ilo - 1, vhi);
          if (vlo <= h) out.push_back({i, vlo, h});
        }
        if (vhi > ihi) {
          const int l = std::max(ihi + 1, vlo);
          if (l <= vhi) out.push_back({i, l, vhi});
        }
      }
    };
    extras(src_s_lo, src_s_hi, s_lo, s_hi, extra_s);
    extras(src_c_lo, src_c_hi, c_lo, c_hi, extra_c);
    prunable = s_lo <= 0 && s_hi >= 0 && c_lo <= 0 && c_hi >= 0;
  }
};

// ---------------------------------------------------------------------------------------
// YIN stage: difference function, CMNDF, troughs, threshold/Boltzmann priors and the
// sparse observation model for a batch of W frames.
// ---------------------------------------------------------------------------------------

template <int Lanes>
struct Batch {
  FrameObs frames[Lanes];
  int count = 0;
  std::vector<Cand> cands;
};

template <class T>
class YinWorker {
 public:
  using V = simd::native_t<T>;
  static constexpr int W = V::size;
  using BatchT = Batch<W>;

  explicit YinWorker(const Model& m) : m_(m) {
    fft_.init(m.M);
    fft2_.init(m.M / 2);
    const std::size_t E = 2 * W;
    a_.resize(std::size_t(m.M) * E);
    b_.resize(std::size_t(m.M) * E);
    energy_.resize(std::size_t(m.pmax + 1) * W);
    yin_.resize(std::size_t(m.K) * W);
    invk_.resize(m.pmax + 1);
    for (int k = 1; k <= m.pmax; ++k) invk_[k] = T(1.0 / k);
    tcap_ = m.K / 2 + 2;
    tlist_.resize(std::size_t(W) * tcap_);
    rowbuf_.resize(std::size_t(W) * m.N);
    zero_.assign(std::size_t(m.N) + W, T(0));
    const int half = m.M / 2;
    twc_.resize(half + 1);
    tws_.resize(half + 1);
    const double pi = 3.14159265358979323846;
    for (int k = 0; k <= half; ++k) {
      const double a = 2.0 * pi * k / m.L;
      twc_[k] = T(std::cos(a));
      tws_[k] = T(std::sin(a));
    }
    // DCT-I tables: sin/cos(pi k / M) and the size-M real-FFT post-processing twiddles.
    dsin_.resize(half + 1);
    dcos_.resize(half + 1);
    for (int k = 0; k <= half; ++k) {
      dsin_[k] = T(std::sin(pi * k / m.M));
      dcos_[k] = T(std::cos(pi * k / m.M));
    }
    jmax_ = (m.pmax + 1) / 2 + 1;
    rtc_.resize(jmax_ + 1);
    rts_.resize(jmax_ + 1);
    for (int j = 0; j <= jmax_; ++j) {
      rtc_[j] = T(std::cos(2.0 * pi * j / m.M));
      rts_[j] = T(std::sin(2.0 * pi * j / m.M));
    }
    acf_.resize(std::size_t(2 * jmax_ + 2) * W);
    tr_h_.resize(m.K);
    tr_k_.resize(m.K);
    tr_p_.resize(m.K);
    kvals_.resize(m.K);
    kbits_.assign(std::size_t(m.nthr) / 64 + 1, 0);
  }

  /// Normalised difference function of lane l from the last run() (diagnostics/tests):
  /// out[i] = yin[min_period + i].
  void yin_values(int l, std::vector<double>& out) const {
    out.resize(std::size_t(m_.K));
    for (int i = 0; i < m_.K; ++i) out[i] = double(yin_[std::size_t(i) * W + l]);
  }

  /// Analyse frames [f0, f0 + count) of signal x (length len) into out.
  void run(const T* x, std::size_t len, std::size_t f0, int count, BatchT& out) {
    const Model& m = m_;
    const int N = m.N;
    out.count = count;
    out.cands.clear();

    // 1. Gather W frames (zero padded where needed) and transpose into lane layout.
    const T* rows[W];
    for (int l = 0; l < W; ++l) {
      if (l >= count) { rows[l] = zero_.data(); continue; }
      const std::int64_t s = m.frame_start(f0 + l);
      if (s >= 0 && s + N <= std::int64_t(len)) {
        rows[l] = x + s;
      } else {
        T* r = rowbuf_.data() + std::size_t(l) * N;
        for (int i = 0; i < N; ++i) {
          const std::int64_t k = s + i;
          r[i] = (k >= 0 && k < std::int64_t(len)) ? x[k] : T(0);
        }
        rows[l] = r;
      }
    }
    T* a = a_.data();
    const int Nb = N - N % W;
    for (int s = 0; s < Nb; s += W) V::transpose(rows, std::size_t(s), a + std::size_t(s) * W);
    for (int s = Nb; s < N; ++s)
      for (int l = 0; l < W; ++l) a[std::size_t(s) * W + l] = rows[l][s];
    std::fill(a + std::size_t(N) * W, a + std::size_t(m.L) * W, T(0));

    // 2. Energy prefix E[k] = sum_{i<k} y[i]^2, accumulated in double precision.
    T* E = energy_.data();
    {
      double acc[W];
      for (int l = 0; l < W; ++l) {
        acc[l] = 0.0;
        E[l] = T(0);
      }
      for (int k = 1; k <= m.pmax; ++k) {
        const T* y = a + std::size_t(k - 1) * W;
        T* e = E + std::size_t(k) * W;
        for (int l = 0; l < W; ++l) {
          acc[l] += double(y[l]) * double(y[l]);
          e[l] = T(acc[l]);
        }
        if (k == 1 && m.opt.reproduce_librosa_quirks)
          for (int l = 0; l < W; ++l) e[l] = T(0);
      }
    }

    // 3. Linear autocorrelation r[k] = sum_m y[m] y[m+k], k <= max_period:
    //    real FFT (size L) -> power spectrum P -> r = (2/L) * DCT-I(P[0..M]); the DCT-I runs
    //    as a size-M real FFT (complex M/2) whose input is formed in the power pass.
    T* z = fft_.forward(a_.data(), b_.data());
    T* u = z == a_.data() ? b_.data() : a_.data();
    V y1 = power_pass(z, u);
    T* z2 = fft2_.forward(u, z);
    dct_post(z2, y1);
    const T* r = acf_.data();

    // 4. Cumulative mean normalised difference (librosa >= 1.0 definition):
    //      d[k] = 2 (acf[0] - acf[k]) - sum_{i<k} y[i]^2,  yin[k] = d[k] / (mean(d[1..k]) + tiny)
    //    The running sum uses compensated (Kahan) summation.
    {
      const V two = V::set1(T(2));
      const V tiny = V::set1(std::numeric_limits<T>::min());
      const V r0 = V::load(r);
      V cum = V::set1(T(0)), comp = cum;
      T* Y = yin_.data();
      for (int k = 1; k <= m.pmax; ++k) {
        const V d = fmsub(two, r0 - V::load(r + std::size_t(k) * W), V::load(E + std::size_t(k) * W));
        const V yk = d - comp;
        const V t = cum + yk;
        comp = (t - cum) - yk;
        cum = t;
        if (k >= m.pmin) (d / fmadd(cum, V::set1(invk_[k]), tiny)).store(Y + std::size_t(k - m.pmin) * W);
      }
    }

    // 5. Troughs (librosa.util.localmin, is_trough[0] = y[0] < y[1]) for all lanes at once.
    {
      const T* Y = yin_.data();
      const int K = m.K;
      const int live = count >= W ? -1 : (1 << count) - 1;
      for (int l = 0; l < W; ++l) tn_[l] = 0;
      auto emit = [&](int bits, int i) {
        bits &= live;
        while (bits) {
          const int l = ctz(bits);
          bits &= bits - 1;
          tlist_[std::size_t(l) * tcap_ + tn_[l]++] = i;
        }
      };
      V prev = V::load(Y), cur = V::load(Y + W);
      emit(movemask(cmpgt(cur, prev)), 0);
      for (int i = 1; i + 1 < K; ++i) {
        const V next = V::load(Y + std::size_t(i + 1) * W);
        const int bits = movemask(mask_and(cmpgt(prev, cur), cmple(cur, next)));
        if (bits) emit(bits, i);
        prev = cur;
        cur = next;
      }
      emit(movemask(cmpgt(prev, cur)), K - 1);
    }

    // 6. Per-frame threshold/Boltzmann priors and observation model.
    for (int l = 0; l < count; ++l) frame_obs(l, out);
  }

 private:
  // In:  z = DFT_M(y[2n] + i y[2n+1])  (frame zero-padded to L = 2M).
  // Out: u[0..M-1] (real, written in sample layout = M/2 packed complex elements) with
  //        u_0 = (P_0 + P_M)/2,  u_k = (P_k + P_{M-k})/2 + (P_k - P_{M-k}) sin(pi k/M),
  //      where P_k = |X_k|^2 is the power spectrum of the zero-padded frame, and the return
  //      value y_1 = (P_0 - P_M)/2 + sum_{k=1}^{M-1} P_k cos(pi k/M) (DCT-I output 1).
  //      Then Re DFT_M(u)_j = y_{2j} and Im DFT_M(u)_j = y_{2j+1} - y_{2j-1}, where
  //      y_n = (P_0 + (-1)^n P_M)/2 + sum_{k=1}^{M-1} P_k cos(pi k n / M) and r[n] = (2/L) y_n.
  V power_pass(const T* z, T* u) const {
    const int M = m_.M;
    const std::size_t E = 2 * W;
    const V half = V::set1(T(0.5)), quarter = V::set1(T(0.25));
    V y1, y1c = V::set1(T(0));  // y1 with Kahan compensation
    {
      const V zr = V::load(z), zi = V::load(z + W);
      const V x0 = zr + zi, xm = zr - zi;  // X_0 and X_M (real)
      const V p0 = x0 * x0, pm = xm * xm;
      (half * (p0 + pm)).store(u);
      y1 = half * (p0 - pm);
    }
    for (int k = 1; k <= M / 2; ++k) {
      const T* pa = z + std::size_t(k) * E;
      const T* pc = z + std::size_t(M - k) * E;
      const V ar = V::load(pa), ai = V::load(pa + W);
      const V cr = V::load(pc), ci = V::load(pc + W);
      const V er = ar + cr, ei = ai - ci;  // 2 Xe = A + conj(C)
      const V xr = ai + ci, xi = cr - ar;  // 2 Xo = -i (A - conj(C))
      const V c = V::set1(twc_[k]), s = V::set1(tws_[k]);
      const V wr = fmadd(c, xr, s * xi);   // 2 w Xo, w = exp(-2 pi i k / L)
      const V wi = fnmadd(s, xr, c * xi);
      const V sr = er + wr, si = ei + wi;  // 2 X_k
      const V dr = er - wr, di = ei - wi;  // 2 conj(X_{M-k})
      const V pk = quarter * fmadd(sr, sr, si * si);
      const V pmk = quarter * fmadd(dr, dr, di * di);
      const V avg = half * (pk + pmk), dif = pk - pmk;
      const V sn = V::set1(dsin_[k]);
      fmadd(dif, sn, avg).store(u + std::size_t(k) * W);
      if (M - k != k) fnmadd(dif, sn, avg).store(u + std::size_t(M - k) * W);
      const V term = dif * V::set1(dcos_[k]) - y1c;
      const V t = y1 + term;
      y1c = (t - y1) - term;
      y1 = t;
    }
    return y1;
  }

  // From Z2 = DFT_{M/2}(u[2n] + i u[2n+1]) produce r[0..2*jmax+1] into acf_ (lane layout).
  void dct_post(const T* z2, V y1) {
    const int Mh = m_.M / 2;
    const std::size_t E = 2 * W;
    const V half = V::set1(T(0.5)), scale = V::set1(T(2.0 / m_.L));
    T* out = acf_.data();
    {
      const V zr = V::load(z2), zi = V::load(z2 + W);
      (scale * (zr + zi)).store(out);  // y_0 = U_0 = Re Z_0 + Im Z_0
      (scale * y1).store(out + W);
    }
    V yodd = y1, comp = V::set1(T(0));  // running sum with Kahan compensation
    for (int j = 1; j <= jmax_; ++j) {
      const T* pa = z2 + std::size_t(j) * E;
      const T* pc = z2 + std::size_t(Mh - j) * E;
      const V ar = V::load(pa), ai = V::load(pa + W);
      const V cr = V::load(pc), ci = V::load(pc + W);
      // U_j = (A + conj C)/2 + exp(-2 pi i j/M) (A - conj C)/(2i)
      const V er = ar + cr, ei = ai - ci;
      const V xr = ai + ci, xi = cr - ar;
      const V c = V::set1(rtc_[j]), s = V::set1(rts_[j]);
      const V ur = half * (er + fmadd(c, xr, s * xi));
      const V ui = half * (ei + fnmadd(s, xr, c * xi));
      const V term = ui - comp;
      const V t = yodd + term;
      comp = (t - yodd) - term;
      yodd = t;
      (scale * ur).store(out + std::size_t(2 * j) * W);
      (scale * yodd).store(out + std::size_t(2 * j + 1) * W);
    }
  }

  static FASTPITCH_INLINE int ctz64(std::uint64_t x) {
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long r;
    _BitScanForward64(&r, x);
    return int(r);
#else
    return __builtin_ctzll(x);
#endif
  }

  static FASTPITCH_INLINE int ctz(unsigned x) {
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long r;
    _BitScanForward(&r, x);
    return int(r);
#else
    return __builtin_ctz(x);
#endif
  }

  void frame_obs(int l, BatchT& out) {
    const Model& m = m_;
    const int K = m.K;
    const T* Y = yin_.data() + l;
    auto y = [&](int i) { return double(Y[std::size_t(i) * W]); };
    const int* tidx = tlist_.data() + std::size_t(l) * tcap_;
    const int nt = tn_[l];

    FrameObs& fo = out.frames[l];
    fo.cbegin = fo.cend = std::uint32_t(out.cands.size());
    if (nt == 0) {
      fo.voiced_prob = 0.0;
      fo.log_unvoiced = std::log(1.0 / m.nbins + m.tiny);
      return;
    }

    // First threshold index each trough lies below (nthr + 1 if none).
    const double* thr = m.thr.data();
    const int nthr = m.nthr;
    int nk = 0;
    int gmin = 0;
    for (int t = 0; t < nt; ++t) {
      const double h = y(tidx[t]);
      tr_h_[t] = h;
      tr_p_[t] = 0.0;
      int k;
      if (!(h < 1.0)) {
        k = nthr + 1;  // also NaN
      } else if (h < thr[1]) {
        k = 1;
      } else {
        k = std::min(std::max(int(h * nthr) + 1, 1), nthr);
        while (k > 1 && h < thr[k - 1]) --k;
        while (k <= nthr && !(h < thr[k])) ++k;
      }
      tr_k_[t] = k;
      if (k <= nthr) kbits_[std::size_t(k) >> 6] |= std::uint64_t(1) << (k & 63);
      if (h < tr_h_[gmin]) gmin = t;
    }
    // Distinct threshold indices in ascending order (bitmap scan, clears the bitmap).
    for (std::size_t w = 0; w < kbits_.size(); ++w) {
      std::uint64_t b = kbits_[w];
      kbits_[w] = 0;
      while (b) {
        kvals_[nk++] = int(w * 64) + ctz64(b);
        b &= b - 1;
      }
    }

    // Thresholds are grouped by the set of troughs below them; within a group the
    // Boltzmann prior of each trough is constant.
    for (int g = 0; g < nk; ++g) {
      const int u = kvals_[g];
      const int unext = g + 1 < nk ? kvals_[g + 1] : m.nthr + 1;
      const double B = m.tailbeta[u] - m.tailbeta[unext];  // mass of thresholds [u, unext)
      int cnt = 0;
      for (int t = 0; t < nt; ++t) cnt += tr_k_[t] <= u;
      const double f = B * m.bz_fact[cnt];
      int pos = 0;
      for (int t = 0; t < nt; ++t)
        if (tr_k_[t] <= u) tr_p_[t] += f * m.bz_exp[pos++];
    }
    tr_p_[gmin] += m.opt.no_trough_prob * m.cumbeta[tr_k_[gmin] - 1];

    // Candidates: parabolic refinement, pitch bin, observation probability.
    for (int t = 0; t < nt; ++t) {
      const double p = tr_p_[t];
      if (!(p != 0.0)) continue;
      const int i = tidx[t];
      double shift = 0.0;
      if (i > 0 && i < K - 1) {
        const double ym = y(i - 1), y0 = y(i), yp = y(i + 1);
        const double aa = yp + ym - 2.0 * y0;
        const double bb = (yp - ym) / 2.0;
        shift = std::fabs(bb) >= std::fabs(aa) ? 0.0 : -bb / aa;
      }
      const double period = double(m.pmin + i) + shift;
      const double f0 = m.opt.sample_rate / period;
      double bin = std::nearbyint(m.bin_scale * std::log2(f0 / m.opt.fmin));
      if (bin < 0) bin = 0;
      if (bin >= m.nbins) continue;  // index n_bins is overwritten by the unvoiced fill
      const int b = int(bin);
      // Bins are non-increasing along the troughs; on a collision the later candidate
      // replaces the earlier one (librosa's fancy-index assignment: last write wins).
      if (out.cands.size() > fo.cbegin && out.cands.back().bin == b) {
        out.cands.back().logp = p;
        out.cands.back().f0 = f0;
      } else {
        out.cands.push_back({b, p, f0});
      }
    }
    fo.cend = std::uint32_t(out.cands.size());
    double vp = 0.0;
    for (std::uint32_t c = fo.cbegin; c < fo.cend; ++c) vp += out.cands[c].logp;
    for (std::uint32_t c = fo.cbegin; c < fo.cend; ++c)
      out.cands[c].logp = std::log(out.cands[c].logp + m.tiny);
    vp = std::min(std::max(vp, 0.0), 1.0);
    fo.voiced_prob = vp;
    fo.log_unvoiced = std::log((1.0 - vp) / m.nbins + m.tiny);
  }

  const Model& m_;
  VecFFT<V> fft_, fft2_;
  std::vector<T> dsin_, dcos_, rtc_, rts_;
  AlignedBuffer<T> acf_;
  int jmax_ = 0;
  AlignedBuffer<T> a_, b_;
  AlignedBuffer<T> energy_, yin_;
  std::vector<T> invk_;
  std::vector<int> tlist_;
  int tn_[W] = {};
  int tcap_ = 0;
  std::vector<T> rowbuf_, zero_;
  std::vector<T> twc_, tws_;
  std::vector<int> tr_k_, kvals_;
  std::vector<std::uint64_t> kbits_;
  std::vector<double> tr_h_, tr_p_;
};

// ---------------------------------------------------------------------------------------
// Viterbi decoder for the pYIN HMM.
//
// States: voiced bins 0..n-1 (global index j) and unvoiced bins (global index n + j).
// log A[(blk_a, i) -> (blk_b, j)] = log(c_ab) + log tri(j - i) - log Z_i inside the allowed
// band. For each destination the predecessor maximising value + log A is selected, ties
// going to the lowest global index exactly like librosa's dense recursion.
//
// Max-plus convolution with free argmax ("grid mode"):
//   Source values f = value - log Z are rounded to a binary fixed-point grid G = q * 2^b
//   held in doubles, and every kernel tap d carries a unique code c(d) < 2^b in the
//   q-units below the grid, decreasing with d. Because every operand is an integer multiple
//   of q well inside 2^51 q, f + g_enc(d) is computed exactly, so a single vmax per tap
//   yields both the maximum and (from the low bits) the first maximising tap. Grid rounding
//   (~4e-9 nats) is far below any meaningful log-probability difference.
//
// Bounded dynamic range: a state whose value falls more than Lambda below the frame maximum
//   can never lie on the optimal path, since the maximum state reaches any state within K0
//   steps, each costing at most max|log A| + |log tiny|. Such states are set to a floor that
//   behaves as -inf (it can never win against a live predecessor), which keeps all
//   arithmetic inside the exact grid range without changing the decoded path.
//
// Voiced-state pruning: voiced bins without a YIN candidate have log p = log(tiny). When the
//   unvoiced observation exceeds that by more than the worst-case gap they are strictly
//   dominated by the unvoiced state of the same bin and are skipped (see push()).
// ---------------------------------------------------------------------------------------

class Decoder {
 public:
  using VT = double;
  using VV = simd::native_t<double>;
  static constexpr int W = VV::size;
  static constexpr int kUnroll = 4;

  explicit Decoder(const Model& m) : m_(m), n_(m.nbins) {
    const int span = std::max(std::abs(m.dmin), std::abs(m.dmax));
    const int blk = kUnroll * W;
    pad_ = ((span + blk + W - 1) / W) * W;
    nr_ = ((n_ + blk - 1) / blk) * blk;
    setup_grid();
    setup_tables();
    fpad_.resize(std::size_t(nr_) + 2 * pad_);
    fpad_.fill(neg_);
    logZ_.resize(nr_);
    logZ_.fill(0.0);
    for (int i = 0; i < n_; ++i) logZ_[i] = m.logZ[i];
    for (auto* b : {&U_, &V_, &Un_, &Vn_, &obsV_}) {
      b->resize(nr_);
      b->fill(neg_);
    }
    for (auto* b : {&cUs_, &dUs_, &cUc_, &dUc_, &cVs_, &dVs_, &cVc_, &dVc_}) {
      b->resize(nr_);
      b->fill(0.0);
    }
    iV_.resize(nr_);
    const double gap = m.logZmax - m.g0;
    const double minls = std::min(m.ls_stay, m.ls_switch);
    prune_c0_ = gap - minls;
    prune_c1_ = (gap - m.ls_stay) + (gap - minls);
    oob_c_ = gap - minls;
  }

  void reset(std::size_t n_frames) {
    t_ = 0;
    recs_.clear();
    recs_.reserve(n_frames);
    pool_.clear();
    pool_.reserve(n_frames * std::size_t(n_ + 16));
    live_.clear();
    prev_dense_ = true;
  }

  void push(const FrameObs& fo, const Cand* cands) {
    const Model& m = m_;
    const int n = n_;
    const VT L0 = m.log_tiny;
    const VT obsU = fo.log_unvoiced;
    const std::uint32_t nc = fo.cend - fo.cbegin;
    cands += fo.cbegin;

    Rec rec;
    rec.off = pool_.size();

    if (t_ == 0) {
      const VT lpi = m.lpi;
      const bool dense = !(m.prunable && obsU - L0 > prune_c0_ + kMargin);
      for (int j = 0; j < n; ++j) Un_[j] = obsU + lpi;
      for (int j = 0; j < n; ++j) Vn_[j] = dense ? L0 + lpi : neg_;
      live_n_.clear();
      for (std::uint32_t c = nc; c-- > 0;) {
        Vn_[cands[c].bin] = cands[c].logp + lpi;
        live_n_.push_back(cands[c].bin);
      }
      rec.dense = dense;
      rec.nlive = 0;
      recs_.push_back(rec);
      VT mx = -std::numeric_limits<VT>::infinity(), mn = std::numeric_limits<VT>::infinity();
      for (int j = 0; j < n; ++j) {
        mx = std::max(mx, Un_[j]);
        mn = std::min(mn, Un_[j]);
      }
      if (dense) {
        for (int j = 0; j < n; ++j) mx = std::max(mx, Vn_[j]);
      } else {
        for (int j : live_n_) mx = std::max(mx, Vn_[j]);
      }
      finish_frame(dense, mx, mn);
      ++t_;
      return;
    }

    // ---- pruning decision for this frame ------------------------------------------------
    // A voiced state without candidate (log p = L0) at t is beaten, for every successor x,
    // by the unvoiced state U[bin(x)] at t whenever
    //   obsU - L0 > spread(t-1) + 2 (logZmax - g0) - ls_stay - min(ls_stay, ls_switch).
    const bool dense = !(m.prunable && obsU - L0 > spread_ + prune_c1_ + kMargin);
    const bool oob = m.exact && !(spread_ + oob_c_ + kMargin < -L0);

    // ---- in-band maxima from the previous frame ---------------------------------------
    // Same-voicing maxima are needed for every destination; cross-voicing maxima into the
    // voiced block only at live voiced destinations.
    build_f(U_.data());
    conv_band(m.s_lo, m.s_hi, cUs_.data(), dUs_.data());
    if (dense || !grid_) {
      conv_band(m.c_lo, m.c_hi, cUc_.data(), dUc_.data());
    } else {
      for (std::uint32_t c = 0; c < nc; ++c) row_max(cands[c].bin, m.c_lo, m.c_hi, cUc_.data(), dUc_.data());
    }
    apply_extras(cUs_.data(), dUs_.data(), cUc_.data(), dUc_.data(), dense, cands, nc);
    if (prev_dense_) {
      build_f(V_.data());
      conv_band(m.c_lo, m.c_hi, cVc_.data(), dVc_.data());
      if (dense || !grid_) {
        conv_band(m.s_lo, m.s_hi, cVs_.data(), dVs_.data());
      } else {
        for (std::uint32_t c = 0; c < nc; ++c) row_max(cands[c].bin, m.s_lo, m.s_hi, cVs_.data(), dVs_.data());
      }
      run_extras(m.extra_c, cVc_.data(), dVc_.data());
      if (dense) apply_extras_same_dense();
      else apply_extras_same_sparse(cands, nc);
    } else {
      scatter_live(dense, cands, nc);
    }

    // ---- destinations: values, back-pointers (stored directly as uint16) and max/min -----
    rec.dense = dense;
    const std::size_t o = pool_.size();
    pool_.resize(o + std::size_t(n) + std::size_t(dense ? nr_ : nr_ - n));
    std::uint16_t* pu = pool_.data() + o;
    VT mx, mnU;
    // Unvoiced destinations: voiced sources (cross) have the lower global indices.
    combine(Un_.data(), cVc_.data(), dVc_.data(), m.ls_switch, 0, cUs_.data(), dUs_.data(),
            m.ls_stay, n, nullptr, obsU, pu, mx, mnU);
    live_n_.clear();
    if (dense) {
      for (int j = 0; j < n; ++j) obsV_[j] = L0;
      for (std::uint32_t c = 0; c < nc; ++c) obsV_[cands[c].bin] = cands[c].logp;
      VT mxv, mnv;
      combine(Vn_.data(), cVs_.data(), dVs_.data(), m.ls_stay, 0, cUc_.data(), dUc_.data(),
              m.ls_switch, n, obsV_.data(), 0, pu + n, mxv, mnv);
      mx = std::max(mx, mxv);
    } else {
      const VT ls = m.ls_switch, lt = m.ls_stay;
      for (std::uint32_t c = nc; c-- > 0;) {  // ascending bins
        const int j = cands[c].bin;
        const VT a = lt + cVs_[j], b = ls + cUc_[j];
        if (b > a) {
          Vn_[j] = b + cands[c].logp;
          iV_[j] = n + j + std::int32_t(dUc_[j]);
        } else {
          Vn_[j] = a + cands[c].logp;
          iV_[j] = j + std::int32_t(dVs_[j]);
        }
        mx = std::max(mx, Vn_[j]);
        live_n_.push_back(j);
      }
    }
    if (oob) {
      apply_oob(dense, obsU, cands, nc, pu);
      mx = -std::numeric_limits<VT>::infinity();
      mnU = std::numeric_limits<VT>::infinity();
      for (int k = 0; k < n; ++k) {
        mx = std::max(mx, Un_[k]);
        mnU = std::min(mnU, Un_[k]);
      }
      if (dense) {
        for (int k = 0; k < n; ++k) mx = std::max(mx, Vn_[k]);
      } else {
        for (int k : live_n_) mx = std::max(mx, Vn_[k]);
      }
    }
    if (dense) {
      pool_.resize(o + 2 * std::size_t(n));
      rec.nlive = 0;
    } else {
      pool_.resize(o + std::size_t(n));
      rec.nlive = std::uint32_t(live_n_.size());
      for (int jj : live_n_) {
        pool_.push_back(std::uint16_t(jj));
        pool_.push_back(std::uint16_t(iV_[jj]));
      }
    }
    recs_.push_back(rec);
    finish_frame(dense, mx, mnU);
    ++t_;
  }

  /// Back-track. states[t] is the global state index (bin, or n + bin if unvoiced).
  void finish(std::int32_t* states) const {
    if (t_ == 0) return;
    const int n = n_;
    VT best = -std::numeric_limits<VT>::infinity();
    int arg = n;
    if (prev_dense_) {
      for (int j = 0; j < n; ++j)
        if (V_[j] > best) { best = V_[j]; arg = j; }
    } else {
      for (int j : live_)
        if (V_[j] > best) { best = V_[j]; arg = j; }
    }
    for (int j = 0; j < n; ++j)
      if (U_[j] > best) { best = U_[j]; arg = n + j; }
    int s = arg;
    for (std::size_t t = t_; t-- > 0;) {
      states[t] = s;
      if (t == 0) break;
      const Rec& r = recs_[t];
      const std::uint16_t* p = pool_.data() + r.off;
      if (s >= n) {
        s = p[s - n];
      } else if (r.dense) {
        s = p[n + s];
      } else {
        const std::uint16_t* lv = p + n;
        int found = 0;
        for (std::uint32_t k = 0; k < r.nlive; ++k)
          if (lv[2 * k] == s) { found = lv[2 * k + 1]; break; }
        s = found;  // decoded voiced states are always live
      }
    }
  }

 private:
  struct Rec {
    std::size_t off;
    std::uint32_t nlive;
    bool dense;
  };
  struct BI {
    VT v;
    int i;
  };
  static constexpr VT kMargin = 1.0;
  static constexpr int kMonoMinTaps = 48;  // narrower bands: plain scan is cheaper

  static VT bits_to_double(std::uint64_t u) {
    VT d;
    std::memcpy(&d, &u, sizeof d);
    return d;
  }

  void setup_tables() {
    const Model& m = m_;
    const int n = n_;
    // Extra (edge-source) transitions indexed by destination, in ascending source order.
    auto csr = [&](const std::vector<Model::Extra>& ex, std::vector<int>& off,
                   std::vector<int>& dd) {
      off.assign(n + 1, 0);
      for (const auto& e : ex)
        for (int d = e.lo; d <= e.hi; ++d) ++off[e.src - d + 1];
      for (int j = 0; j < n; ++j) off[j + 1] += off[j];
      dd.assign(off[n], 0);
      std::vector<int> pos(off.begin(), off.end() - 1);
      for (const auto& e : ex)
        for (int d = e.lo; d <= e.hi; ++d) dd[pos[e.src - d]++] = d;
    };
    csr(m.extra_c, xc_off_, xc_d_);
    csr(m.extra_s, xs_off_, xs_d_);
    // Reversed kernel gt[e] = g[-e] for e in [-dmax, -dmin], padded for vector loads.
    const int taps = m.dmax - m.dmin + 1;
    grev_off_ = W + m.dmax;
    grev_.assign(std::size_t(taps + 2 * W + 1), neg_);
    for (int d = m.dmin; d <= m.dmax; ++d) grev_[std::size_t(grev_off_ - d)] = gplain_[d - m.dmin];
  }

  void setup_grid() {
    const Model& m = m_;
    const int n = n_;
    const double L0abs = -m.log_tiny;
    taps_ = m.dmax - m.dmin + 1;
    gplain_.assign(m.gk.begin(), m.gk.end());
    // Worst one-step transition cost.
    double amax = 0;
    if (m.exact) {
      amax = L0abs;
    } else {
      for (int i = 0; i < n; ++i)
        for (int blk = 0; blk < 2; ++blk) {
          const double lc = blk ? m.ls_switch : m.ls_stay;
          const int lo = blk ? m.src_c_lo[i] : m.src_s_lo[i];
          const int hi = blk ? m.src_c_hi[i] : m.src_s_hi[i];
          for (int d = lo; d <= hi; ++d)
            amax = std::max(amax, -(lc + m.gk[d - m.dmin] - m.logZ[i]));
        }
    }
    // Number of steps within which the maximum state can reach any state at an exactly
    // prescribed time: exact mode allows every transition (log tiny), otherwise move at
    // most r bins per step inside the unvoiced block (self transitions pad the route),
    // plus one switch out of and one switch back into the voiced block.
    int k0 = 0;
    if (m.exact) {
      k0 = 1;
    } else if (m.prunable) {
      const int r = std::min(-m.s_lo, m.s_hi);
      if (r > 0) k0 = (n - 1 + r - 1) / r + 2;
    }
    grid_ = false;
    neg_ = -1e30;
    if (k0 > 0) {
      lambda_ = k0 * (amax + L0abs) + 64.0;
      const double need = lambda_ + 2.0 * (amax + L0abs) + 1024.0;
      const int e = int(std::ceil(std::log2(need)));
      int b = 0;
      while ((1 << b) < taps_) ++b;
      const int qexp = e + 2 - 51;  // |values| < 2^(e+1) <= 2^51 q / 2
      const double G = std::ldexp(1.0, qexp + b);
      if (G <= 1e-6 && b <= 20) {
        grid_ = true;
        q_ = std::ldexp(1.0, qexp);
        invq_ = std::ldexp(1.0, -qexp);
        G_ = G;
        invG_ = std::ldexp(1.0, -(qexp + b));
        neg_ = -std::ldexp(1.0, e);
        lowmask_ = (std::uint64_t(1) << b) - 1;
        gplain_.resize(taps_);
        genc_.resize(taps_);
        for (int k = 0; k < taps_; ++k) {
          const double gg = std::nearbyint(m.gk[k] * invG_) * G_;
          gplain_[k] = gg;
          genc_[k] = gg + double(taps_ - 1 - k) * q_;
        }
        dbase_ = double(m.dmin + taps_ - 1);
        // Monotone argmax search requires a strictly concave (grid) kernel.
        mono_ = true;
        for (int k = 1; k + 1 < taps_; ++k)
          if (!(gplain_[k - 1] + gplain_[k + 1] - 2.0 * gplain_[k] < 0.0)) mono_ = false;
        abnd_.resize(std::size_t(nr_ / (kUnroll * W)) + 1);
      }
    }
  }

  // f = value - offset - log Z (offset = previous frame maximum). In grid mode f is rounded
  // to the grid and states more than Lambda below the maximum become the dead floor.
  void build_f(const VT* src) {
    VT* f = fpad_.data() + pad_;
    const VV off = VV::set1(off_);
    if (grid_) {
      const VV ig = VV::set1(invG_), g = VV::set1(G_);
      const VV lam = VV::set1(-lambda_), neg = VV::set1(neg_);
      for (int j = 0; j < nr_; j += W) {
        const VV x = VV::load(src + j) - off;
        const VV r = round_nearest((x - VV::load(logZ_.data() + j)) * ig) * g;
        select(cmpgt(lam, x), neg, r).storeu(f + j);
      }
    } else {
      for (int j = 0; j < nr_; j += W)
        (VV::load(src + j) - off - VV::load(logZ_.data() + j)).storeu(f + j);
    }
    for (int j = n_; j < nr_; ++j) f[j] = neg_;
  }


  // dst[j] = obs_j + max(la + ca[j], lb + cb[j]); a-sources have the lower global indices so
  // they win ties. Back-pointers (global source index) are written as uint16 to ptr; the
  // maximum and minimum over the n valid destinations are returned in mx / mn.
  void combine(VT* dst, const VT* ca, const VT* da, VT la, int aoff, const VT* cb,
               const VT* db, VT lb, int boff, const VT* obs, VT obsc, std::uint16_t* ptr,
               VT& mx, VT& mn) const {
    const VV vla = VV::set1(la), vlb = VV::set1(lb), voc = VV::set1(obsc);
    const VV va = VV::set1(VT(aoff)), vb = VV::set1(VT(boff)), step = VV::set1(VT(W));
    VV jv = VV::iota();
    VV vmx = VV::set1(-std::numeric_limits<VT>::infinity());
    VV vmn = VV::set1(std::numeric_limits<VT>::infinity());
    const int n = n_;
    int j = 0;
    auto body = [&](int k) {
      const VV a = vla + VV::load(ca + k);
      const VV b = vlb + VV::load(cb + k);
      const auto mk = cmpgt(b, a);
      const VV v = select(mk, b, a) + (obs ? VV::load(obs + k) : voc);
      v.store(dst + k);
      store_u16(ptr + k, select(mk, VV::load(db + k) + vb, VV::load(da + k) + va) + jv);
      jv = jv + step;
      return v;
    };
    for (; j + W <= n; j += W) {
      const VV v = body(j);
      vmx = vmax(vmx, v);
      vmn = vmin(vmn, v);
    }
    mx = hmax(vmx);
    mn = hmin(vmn);
    if (j < n) {
      body(j);
      for (; j < n; ++j) {
        mx = std::max(mx, dst[j]);
        mn = std::min(mn, dst[j]);
      }
    }
  }

  // Max-plus convolution over source offsets [lo, hi] for all destinations.
  void conv_band(int lo, int hi, VT* c, VT* d) {
    if (lo > hi) {  // empty band: no predecessor through this block
      for (int j = 0; j < nr_; ++j) {
        c[j] = neg_ * 2;
        d[j] = 0;
      }
      return;
    }
    const VT* f = fpad_.data() + pad_;
    if (!grid_) {
      blend_conv(f, gplain_.data() - m_.dmin, lo, hi, c, d);
    } else if (mono_ && hi - lo + 1 >= kMonoMinTaps) {
      mono_conv(lo, hi, c, d);
    } else {
      grid_conv(lo, hi, c);
      decode(c, d);
    }
  }

  // Encoded maxima over taps [lo, hi] for all destinations (plain scan).
  void grid_conv(int lo, int hi, VT* out) const {
    constexpr int U = kUnroll;
    const VT* f = fpad_.data() + pad_;
    const VT* g = genc_.data() - m_.dmin;
    const VV neg = VV::set1(neg_ * 2);
    for (int j = 0; j < nr_; j += U * W) {
      const VT* fj = f + j;
      VV a[U];
      for (int u = 0; u < U; ++u) a[u] = neg;
      for (int d = lo; d <= hi; ++d) {
        const VV gd = VV::set1(g[d]);
        for (int u = 0; u < U; ++u) a[u] = vmax(a[u], VV::loadu(fj + u * W + d) + gd);
      }
      for (int u = 0; u < U; ++u) a[u].store(out + j + u * W);
    }
  }

  // Encoded maximum over taps [lo, hi] for a single destination j (SIMD along taps).
  VT row_max_enc(int j, int lo, int hi) const {
    const VT* f = fpad_.data() + pad_ + j;
    const VT* g = genc_.data() - m_.dmin;
    VV a0 = VV::set1(neg_ * 2), a1 = a0, a2 = a0, a3 = a0;
    int d = lo;
    for (; d + 4 * W - 1 <= hi; d += 4 * W) {
      a0 = vmax(a0, VV::loadu(f + d) + VV::loadu(g + d));
      a1 = vmax(a1, VV::loadu(f + d + W) + VV::loadu(g + d + W));
      a2 = vmax(a2, VV::loadu(f + d + 2 * W) + VV::loadu(g + d + 2 * W));
      a3 = vmax(a3, VV::loadu(f + d + 3 * W) + VV::loadu(g + d + 3 * W));
    }
    for (; d + W - 1 <= hi; d += W) a0 = vmax(a0, VV::loadu(f + d) + VV::loadu(g + d));
    VT r = hmax(vmax(vmax(a0, a1), vmax(a2, a3)));
    for (; d <= hi; ++d) r = std::max(r, f[d] + g[d]);
    return r;
  }

  // Code of an encoded value -> source offset d (and the plain value).
  FASTPITCH_INLINE int decode_one(VT v, VT* value) const {
    const VT y = v * invq_ + 6755399441055744.0;
    std::uint64_t u;
    std::memcpy(&u, &y, 8);
    const int code = int(u & lowmask_);
    if (value) *value = v - VT(code) * q_;
    return int(dbase_) - code;
  }

  void row_max(int j, int lo, int hi, VT* c, VT* dd) const {
    VT v;
    const int d = decode_one(row_max_enc(j, lo, hi), &v);
    c[j] = v;
    dd[j] = VT(d);
  }

  // Max-plus convolution over taps [lo, hi] for all destinations, exploiting that the first
  // maximising source a(j) is non-decreasing in j: the score matrix f[i] + g(i - j) is Monge
  // because the (grid-exact) log-triangle kernel is strictly concave, and the tap codes add
  // a term linear in i - j, which preserves the property. Exact a(j) is computed for every
  // block start; a block then only scans sources in [a(j0), a(j0 + B)].
  void mono_conv(int lo, int hi, VT* out, VT* dout) {
    constexpr int B = kUnroll * W;
    const int nb = nr_ / B;
    const VT* f = fpad_.data() + pad_;
    const VT* g = genc_.data() - m_.dmin;
    int* a = abnd_.data();
    // Independent rows (no carried dependency) so they overlap in the out-of-order core.
    for (int b = 0; b < nb; ++b) a[b] = b * B + decode_one(row_max_enc(b * B, lo, hi), nullptr);
    const VV neg = VV::set1(neg_ * 2);
    for (int b = 0; b < nb; ++b) {
      const int j0 = b * B;
      const int dlo = std::max(lo, a[b] - j0 - (B - 1));
      const int dhi = b + 1 < nb ? std::min(hi, a[b + 1] - j0) : hi;
      const VT* fj = f + j0;
      VV acc[kUnroll];
      for (int u = 0; u < kUnroll; ++u) acc[u] = neg;
      for (int d = dlo; d <= dhi; ++d) {
        const VV gd = VV::set1(g[d]);
        for (int u = 0; u < kUnroll; ++u) acc[u] = vmax(acc[u], VV::loadu(fj + u * W + d) + gd);
      }
      for (int u = 0; u < kUnroll; ++u) acc[u].store(out + j0 + u * W);
    }
    decode(out, dout);
  }

  // Split encoded maxima into value (in place) and source offset d.
  void decode(VT* c, VT* dout) const {
    const VV invq = VV::set1(invq_), q = VV::set1(q_);
    const VV magic = VV::set1(6755399441055744.0);  // 1.5 * 2^52
    const VV two52 = VV::set1(4503599627370496.0);  // 2^52
    const VV mask = VV::set1(bits_to_double(lowmask_));
    const VV base = VV::set1(dbase_);
    for (int j = 0; j < nr_; j += W) {
      const VV v = VV::load(c + j);
      const VV y = v * invq + magic;  // exact: integer-valued, |.| < 2^51
      const VV code = bit_or(bit_and(y, mask), two52) - two52;
      fnmadd(code, q, v).store(c + j);
      (base - code).store(dout + j);
    }
  }

  // Exact fallback (no grid): first-index argmax via compare/blend.
  void blend_conv(const VT* f, const VT* g, int lo, int hi, VT* out, VT* od) const {
    for (int j = 0; j < nr_; j += 2 * W) {
      VV b0 = VV::set1(neg_ * 2), b1 = b0, i0 = VV::set1(0.0), i1 = i0;
      for (int d = lo; d <= hi; ++d) {
        const VV gd = VV::set1(g[d]), dd = VV::set1(VT(d));
        const VV x0 = VV::loadu(f + j + d) + gd;
        const VV x1 = VV::loadu(f + j + W + d) + gd;
        const auto m0 = cmpgt(x0, b0), m1 = cmpgt(x1, b1);
        b0 = select(m0, x0, b0);
        i0 = select(m0, dd, i0);
        b1 = select(m1, x1, b1);
        i1 = select(m1, dd, i1);
      }
      b0.store(out + j);
      b1.store(out + j + W);
      i0.store(od + j);
      i1.store(od + j + W);
    }
  }

  // Edge sources whose (smaller) row normaliser admits extra transitions. Cross-voicing
  // results are only needed at the live voiced destinations when the frame is pruned.
  void apply_extras(VT* cs, VT* ds, VT* cc, VT* dc, bool dense, const Cand* cands,
                    std::uint32_t nc) {
    const Model& m = m_;
    const VT* f = fpad_.data() + pad_;
    const VT* g = gplain_.data() - m.dmin;
    run_extras(m.extra_s, cs, ds);
    if (dense) {
      run_extras(m.extra_c, cc, dc);
      return;
    }
    for (std::uint32_t c = 0; c < nc; ++c) {
      const int j = cands[c].bin;
      for (int k = xc_off_[j]; k < xc_off_[j + 1]; ++k) {
        const int d = xc_d_[k];
        const VT fv = f[j + d];
        if (!(fv > neg_)) continue;
        const VT v = fv + g[d];
        if (v > cc[j] || (v == cc[j] && VT(d) < dc[j])) {
          cc[j] = v;
          dc[j] = VT(d);
        }
      }
    }
  }

  void run_extras(const std::vector<Model::Extra>& ex, VT* c, VT* dd) const {
    const VT* f = fpad_.data() + pad_;
    const VT* g = gplain_.data() - m_.dmin;
    for (const auto& e : ex) {
      const VT fv = f[e.src];
      if (!(fv > neg_)) continue;
      for (int d = e.lo; d <= e.hi; ++d) {
        const int j = e.src - d;
        const VT v = fv + g[d];
        if (v > c[j] || (v == c[j] && VT(d) < dd[j])) {
          c[j] = v;
          dd[j] = VT(d);
        }
      }
    }
  }

  // V->V (same-voicing) extras of a dense previous frame.
  void apply_extras_same_dense() {
    const Model& m = m_;
    const VT* f = fpad_.data() + pad_;
    const VT* g = gplain_.data() - m.dmin;
    for (const auto& e : m.extra_s) {
      const VT fv = f[e.src];
      if (!(fv > neg_)) continue;
      for (int d = e.lo; d <= e.hi; ++d) {
        const int j = e.src - d;
        const VT v = fv + g[d];
        if (v > cVs_[j] || (v == cVs_[j] && VT(d) < dVs_[j])) {
          cVs_[j] = v;
          dVs_[j] = VT(d);
        }
      }
    }
  }
  void apply_extras_same_sparse(const Cand* cands, std::uint32_t nc) {
    const Model& m = m_;
    const VT* f = fpad_.data() + pad_;
    const VT* g = gplain_.data() - m.dmin;
    for (std::uint32_t c = 0; c < nc; ++c) {
      const int j = cands[c].bin;
      for (int k = xs_off_[j]; k < xs_off_[j + 1]; ++k) {
        const int d = xs_d_[k];
        const VT fv = f[j + d];
        if (!(fv > neg_)) continue;
        const VT v = fv + g[d];
        if (v > cVs_[j] || (v == cVs_[j] && VT(d) < dVs_[j])) {
          cVs_[j] = v;
          dVs_[j] = VT(d);
        }
      }
    }
  }

  // Previous frame was pruned: only live voiced sources contribute.
  void scatter_live(bool dense, const Cand* cands, std::uint32_t nc) {
    const Model& m = m_;
    const int n = n_;
    const VV ng = VV::set1(neg_ * 2);
    for (int j = 0; j < nr_; j += W) ng.store(cVc_.data() + j);
    if (dense)
      for (int j = 0; j < nr_; j += W) ng.store(cVs_.data() + j);
    const VT* gt = grev_.data() + grev_off_;  // gt[e] = g[-e]
    const VV step = VV::set1(VT(W));
    for (int c : live_) {  // ascending source order -> first maximum wins
      const VT pvc = prev_value(V_, c);
      if (!(pvc > -std::numeric_limits<VT>::infinity())) continue;
      VT fv = pvc - logZ_[c];
      if (grid_) fv = std::nearbyint(fv * invG_) * G_;
      const VV fvv = VV::set1(fv), cv = VV::set1(VT(c));
      for (int blk = dense ? 0 : 1; blk < 2; ++blk) {
        VT* cb = blk ? cVc_.data() : cVs_.data();
        VT* db = blk ? dVc_.data() : dVs_.data();
        const int lo = std::max(blk ? m.src_c_lo[c] : m.src_s_lo[c], c - n + 1);
        const int hi = std::min(blk ? m.src_c_hi[c] : m.src_s_hi[c], c);
        // destinations j = c - d for d in [lo, hi], i.e. j in [c - hi, c - lo]
        int j = c - hi;
        const int jend = c - lo;
        VV jv = VV::iota() + VV::set1(VT(j));
        for (; j + W - 1 <= jend; j += W) {
          const VV x = fvv + VV::loadu(gt + (j - c));
          const VV cur = VV::loadu(cb + j);
          const auto mk = cmpgt(x, cur);
          select(mk, x, cur).storeu(cb + j);
          select(mk, cv - jv, VV::loadu(db + j)).storeu(db + j);
          jv = jv + step;
        }
        for (; j <= jend; ++j) {
          const VT x = fv + gt[j - c];
          if (x > cb[j]) {
            cb[j] = x;
            db[j] = VT(c - j);
          }
        }
      }
    }
    if (!dense) {
      // V->V maxima are only needed at the live destinations of this frame.
      const VT* g = gplain_.data() - m.dmin;
      for (std::uint32_t k = 0; k < nc; ++k) {
        const int j = cands[k].bin;
        VT best = neg_ * 2, bd = 0;
        for (int c : live_) {
          const VT pvc = prev_value(V_, c);
          if (!(pvc > -std::numeric_limits<VT>::infinity())) continue;
          const int d = c - j;
          if (d < m.src_s_lo[c] || d > m.src_s_hi[c]) continue;
          VT fv = pvc - logZ_[c];
          if (grid_) fv = std::nearbyint(fv * invG_) * G_;
          const VT x = fv + g[d];
          if (x > best) {
            best = x;
            bd = VT(d);
          }
        }
        cVs_[j] = best;
        dVs_[j] = bd;
      }
    }
  }

  // Exact mode only: transitions with probability 0 still carry log(tiny).
  void apply_oob(bool dense, VT obsU, const Cand* cands, std::uint32_t nc, std::uint16_t* pu) {
    const Model& m = m_;
    const int n = n_;
    const VT L0 = m.log_tiny;
    auto better = [](VT v, int i, const BI& b) { return v > b.v || (v == b.v && i < b.i); };
    std::vector<BI>& pre = oob_pre_;
    std::vector<BI>& suf = oob_suf_;
    const BI none{-std::numeric_limits<VT>::infinity(), 1 << 30};
    pre.assign(n + 1, none);
    suf.assign(n + 1, none);
    const VT ninf = -std::numeric_limits<VT>::infinity();
    std::vector<VT>& pv = oob_pv_;
    std::vector<VT>& pw = oob_pw_;
    pv.assign(n, ninf);
    pw.resize(n);
    if (prev_dense_) {
      for (int i = 0; i < n; ++i) pv[i] = prev_value(V_, i);
    } else {
      for (int i : live_) pv[i] = prev_value(V_, i);
    }
    for (int i = 0; i < n; ++i) pw[i] = prev_value(U_, i);
    for (int i = 0; i < n; ++i) {
      BI b = pre[i];
      if (pv[i] > ninf && better(pv[i], i, b)) b = {pv[i], i};
      if (pw[i] > ninf && better(pw[i], n + i, b)) b = {pw[i], n + i};
      pre[i + 1] = b;
    }
    for (int i = n - 1; i >= 0; --i) {
      BI b = suf[i + 1];
      if (pv[i] > ninf && better(pv[i], i, b)) b = {pv[i], i};
      if (pw[i] > ninf && better(pw[i], n + i, b)) b = {pw[i], n + i};
      suf[i] = b;
    }
    // Sources i with i - j outside [dmin, dmax] are out of band for destination j.
    auto oob_best = [&](int j) {
      BI b = none;
      const int hiPre = j + m.dmin;
      const int loSuf = j + m.dmax + 1;
      if (hiPre > 0) {
        const BI& p = pre[std::min(hiPre, n)];
        if (better(p.v, p.i, b)) b = p;
      }
      if (loSuf < n) {
        const BI& s = suf[std::max(loSuf, 0)];
        if (better(s.v, s.i, b)) b = s;
      }
      return b;
    };
    for (int j = 0; j < n; ++j) {
      const BI o = oob_best(j);
      if (o.i == none.i) continue;
      const VT v = o.v + L0 + obsU;
      if (better(v, o.i, BI{Un_[j], int(pu[j])})) {
        Un_[j] = v;
        pu[j] = std::uint16_t(o.i);
      }
    }
    auto upd_v = [&](int j, VT obs) {
      const BI o = oob_best(j);
      if (o.i == none.i) return;
      const VT v = o.v + L0 + obs;
      const int cur = dense ? int(pu[n + j]) : iV_[j];
      if (better(v, o.i, BI{Vn_[j], cur})) {
        Vn_[j] = v;
        if (dense) pu[n + j] = std::uint16_t(o.i);
        else iV_[j] = o.i;
      }
    };
    if (dense) {
      for (int j = 0; j < n; ++j) upd_v(j, obsV_[j]);
    } else {
      for (std::uint32_t c = 0; c < nc; ++c) upd_v(cands[c].bin, cands[c].logp);
    }
  }

  // Values are kept un-normalised; the next build_f() subtracts off_ = this frame's max.
  void finish_frame(bool dense, VT mx, VT mnU) {
    std::swap(U_, Un_);
    std::swap(V_, Vn_);
    off_ = mx;
    spread_ = mx - mnU;
    std::sort(live_n_.begin(), live_n_.end());
    live_.swap(live_n_);
    if (dense) live_.clear();
    prev_dense_ = dense;
  }

  // Normalised previous-frame value, -inf if dead (or not live in a pruned frame).
  VT prev_value(const AlignedBuffer<VT>& b, int i) const {
    const VT v = b[i] - off_;
    return (grid_ && v < -lambda_) ? -std::numeric_limits<VT>::infinity() : v;
  }

  const Model& m_;
  int n_, nr_, pad_;
  int taps_ = 0;
  bool grid_ = false, mono_ = false;
  std::vector<int> xc_off_, xc_d_, xs_off_, xs_d_;
  std::vector<VT> grev_;
  int grev_off_ = 0;
  std::vector<int> abnd_;
  VT neg_ = -1e30, lambda_ = 0, q_ = 0, invq_ = 0, G_ = 0, invG_ = 0, dbase_ = 0;
  std::uint64_t lowmask_ = 0;
  std::vector<VT> gplain_, genc_;
  AlignedBuffer<VT> fpad_, logZ_;
  AlignedBuffer<VT> U_, V_, Un_, Vn_, obsV_;
  AlignedBuffer<VT> cUs_, dUs_, cUc_, dUc_, cVs_, dVs_, cVc_, dVc_;
  AlignedBuffer<std::int32_t> iV_;
  std::vector<int> live_, live_n_;
  bool prev_dense_ = true;
  VT spread_ = 0, off_ = 0;
  VT prune_c0_ = 0, prune_c1_ = 0, oob_c_ = 0;
  std::size_t t_ = 0;
  std::vector<Rec> recs_;
  std::vector<std::uint16_t> pool_;  // back-pointers (global state index < 65536)
  std::vector<BI> oob_pre_, oob_suf_;
  std::vector<VT> oob_pv_, oob_pw_;
};
}  // namespace detail

// ---------------------------------------------------------------------------------------
// Public analyser
// ---------------------------------------------------------------------------------------

/// pYIN analyser. Construct once per configuration and reuse; buffers are kept between
/// calls. A single instance must not be used from several threads at the same time.
///
/// T : sample / FFT precision (float or double). Decoding always uses double precision.
template <class T = float>
class Pyin {
  static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                "fastpitch::Pyin supports float or double samples");

 public:
  explicit Pyin(const PyinOptions& opt = PyinOptions()) : model_(opt), decoder_(model_) {}

  const PyinOptions& options() const { return model_.opt; }
  int hop_length() const { return model_.hop; }
  int n_pitch_bins() const { return model_.nbins; }
  const std::vector<double>& bin_frequencies() const { return model_.freqs; }
  std::size_t num_frames(std::size_t n_samples) const { return model_.num_frames(n_samples); }
  /// Internal FFT length used for the autocorrelation (for diagnostics).
  int fft_length() const { return model_.L; }
  static const char* simd_isa() { return detail::simd::isa_name(); }

  PyinResult<T> operator()(const T* x, std::size_t n) {
    PyinResult<T> r;
    analyze(x, n, r);
    return r;
  }
  PyinResult<T> operator()(const std::vector<T>& x) { return (*this)(x.data(), x.size()); }

  void analyze(const T* x, std::size_t n, PyinResult<T>& out) {
    const detail::Model& m = model_;
    const std::size_t nf = m.num_frames(n);
    out.sample_rate = m.opt.sample_rate;
    out.hop_length = m.hop;
    out.f0.resize(nf);
    out.voiced.resize(nf);
    out.voiced_prob.resize(nf);
    if (nf == 0) return;
    constexpr int W = Worker::W;
    const std::size_t nb = (nf + W - 1) / W;
    decoder_.reset(nf);

    int threads = m.opt.num_threads;
    if (threads <= 0) threads = int(std::max(1u, std::thread::hardware_concurrency()));
    threads = int(std::min<std::size_t>(std::size_t(threads), nb));
    while (int(workers_.size()) < threads) workers_.emplace_back(new Worker(m));

    const bool refine = m.opt.refine_f0;
    if (refine) {
      refine_off_.assign(nf + 1, 0);
      refine_.clear();
    }
    auto consume = [&](const Batch& b, std::size_t f0) {
      for (int l = 0; l < b.count; ++l) {
        const detail::FrameObs& fo = b.frames[l];
        out.voiced_prob[f0 + l] = T(fo.voiced_prob);
        decoder_.push(fo, b.cands.data());
        if (refine) {
          for (std::uint32_t c = fo.cbegin; c < fo.cend; ++c)
            refine_.push_back({b.cands[c].bin, 0.0, b.cands[c].f0});
          refine_off_[f0 + l + 1] = refine_.size();
        }
      }
    };

    if (threads <= 1) {
      Batch batch;
      for (std::size_t bi = 0; bi < nb; ++bi) {
        const std::size_t f0 = bi * W;
        workers_[0]->run(x, n, f0, int(std::min<std::size_t>(W, nf - f0)), batch);
        consume(batch, f0);
      }
    } else {
      // Workers analyse batches in parallel (claimed in increasing order) while this thread
      // feeds the sequential decoder, analysing batches itself whenever it would wait.
      batches_.resize(nb);
      std::unique_ptr<std::atomic<int>[]> ready(new std::atomic<int>[nb]);
      for (std::size_t i = 0; i < nb; ++i) ready[i].store(0, std::memory_order_relaxed);
      std::atomic<std::size_t> next{0};
      std::atomic<bool> failed{false};
      std::exception_ptr error;
      std::mutex error_mutex;
      auto produce = [&](Worker& w) -> bool {
        const std::size_t bi = next.fetch_add(1, std::memory_order_relaxed);
        if (bi >= nb) return false;
        const std::size_t f0 = bi * W;
        try {
          w.run(x, n, f0, int(std::min<std::size_t>(W, nf - f0)), batches_[bi]);
        } catch (...) {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (!error) error = std::current_exception();
          failed.store(true, std::memory_order_release);
          next.store(nb, std::memory_order_relaxed);
        }
        ready[bi].store(1, std::memory_order_release);
        return true;
      };
      std::vector<std::thread> pool;
      for (int t = 1; t < threads; ++t)
        pool.emplace_back([&, t] { while (produce(*workers_[t])) {} });
      for (std::size_t bi = 0; bi < nb && !failed.load(std::memory_order_acquire); ++bi) {
        while (!ready[bi].load(std::memory_order_acquire)) {
          if (!produce(*workers_[0])) std::this_thread::yield();
        }
        if (!failed.load(std::memory_order_acquire)) consume(batches_[bi], bi * W);
      }
      for (auto& th : pool) th.join();
      if (error) std::rethrow_exception(error);
    }

    states_.resize(nf);
    decoder_.finish(states_.data());
    const int nbins = m.nbins;
    for (std::size_t t = 0; t < nf; ++t) {
      const int s = states_[t];
      const bool v = s < nbins;
      out.voiced[t] = v;
      double f = m.freqs[v ? s : s - nbins];
      if (v && refine)
        for (std::size_t c = refine_off_[t]; c < refine_off_[t + 1]; ++c)
          if (refine_[c].bin == s) f = refine_[c].f0;
      out.f0[t] = (!v && m.opt.fill_unvoiced) ? T(m.opt.fill_value) : T(f);
    }
  }

 private:
  using Worker = detail::YinWorker<T>;
  using Batch = typename Worker::BatchT;
  detail::Model model_;
  detail::Decoder decoder_;
  std::vector<std::unique_ptr<Worker>> workers_;
  std::vector<Batch> batches_;
  std::vector<std::int32_t> states_;
  std::vector<detail::Cand> refine_;
  std::vector<std::size_t> refine_off_;
};

/// One-shot convenience wrapper.
template <class T>
PyinResult<T> pyin(const T* x, std::size_t n, const PyinOptions& opt = PyinOptions()) {
  Pyin<T> p(opt);
  return p(x, n);
}
template <class T>
PyinResult<T> pyin(const std::vector<T>& x, const PyinOptions& opt = PyinOptions()) {
  return pyin(x.data(), x.size(), opt);
}

}  // namespace fastpitch

#endif  // FASTPITCH_PYIN_HPP_INCLUDED
