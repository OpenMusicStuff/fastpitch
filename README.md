# fastpitch

**Header-only C++17 pYIN pitch tracker.** It produces the same result as `librosa.pyin`
on every frame, and runs **~10× faster than the original C++ pYIN, 64–100× faster than
librosa** on one core, and more with threads.

```cpp
#include <fastpitch/pyin.hpp>

fastpitch::PyinOptions opt;          // librosa.pyin defaults (fmin=C2, fmax=C7, 2048/512)
opt.sample_rate = 44100;
fastpitch::Pyin<float> tracker(opt);  // reusable, keeps its buffers
auto r = tracker(samples.data(), samples.size());
// r.f0[i] (Hz, NaN if unvoiced), r.voiced[i], r.voiced_prob[i], r.time(i)
```

One file (`include/fastpitch/pyin.hpp`), no dependencies, MIT licensed. It has AVX2/FMA,
SSE2 and NEON backends plus a portable scalar fallback, and builds warning-free on
MSVC 2019, GCC 16 and Clang 23.

---

## Results

Measured on an Intel Core Ultra 7 270K Plus (Arrow Lake), Windows 11, Clang 23 `-O3
-march=native` (AVX2+FMA). The input is 300 s of speech (librosa's `libri1` example,
tiled). Times are the best of several runs of the full pipeline (framing → f0 track). All
C/C++ competitors were built with the same compiler flags (libpyin with its own `-Ofast`
release flags). Reproduce with `bench/run_benchmarks.py`.

### librosa defaults: sr = 22 050 Hz, frame 2048, hop 512

| implementation | HMM size | time | × realtime | fastpitch (1 thread) is faster by |
|---|---|---:|---:|---:|
| **fastpitch, 1 thread** | 601 bins, ±50 bins/frame (librosa) | **49 ms** | **6 088×** | — |
| **fastpitch, 8 threads** | same | **24 ms** | **12 572×** | — |
| **fastpitch, 1 thread, c4dm-sized HMM** | 346 bins, ±5 bins/frame | **37 ms** | **8 209×** | — |
| c4dm/pyin (Mauch & Dixon, original C++) | 345 bins, ±5 bins/frame | 556 ms | 540× | **11.3×** (15.2× same-size HMM) |
| Sleepwalking/libpyin (C) | own variant | 453 ms¹ | 662× | **9.2×** |
| librosa 1.0.0 `pyin` (NumPy/Numba) | 601 bins, ±50 bins/frame | 4 994 ms | 60× | **101×** |

### pYIN Vamp framing: sr = 44 100 Hz, frame 2048, hop 256

| implementation | HMM size | time | × realtime | fastpitch (1 thread) is faster by |
|---|---|---:|---:|---:|
| **fastpitch, 1 thread** | 601 bins, ±15 bins/frame (librosa) | **209 ms** | **1 433×** | — |
| **fastpitch, 8 threads** | same | **82 ms** | **3 650×** | — |
| **fastpitch, 1 thread, c4dm-sized HMM** | 346 bins, ±5 bins/frame | **168 ms** | **1 790×** | — |
| c4dm/pyin (original C++) | 345 bins, ±5 bins/frame | 2 190 ms | 137× | **10.5×** (13.1× same-size HMM) |
| Sleepwalking/libpyin (C) | own variant | 1 830 ms¹ | 164× | **8.7×** |
| librosa 1.0.0 `pyin` | 601 bins, ±15 bins/frame | 13 405 ms | 22× | **64×** |

¹ libpyin crashes intermittently (an upstream bug) with these frequency ranges. The
timings are from runs that completed.

With threads the YIN stage scales across cores. The Viterbi decoder is inherently
sequential and then bounds throughput at ~1.7 µs/frame. To process many files, run one
`Pyin` per thread.

Also considered: [sevagh/pitch-detection](https://github.com/sevagh/pitch-detection)'s
pYIN returns one pitch per buffer (it does not track frames) and needs mlpack and ffts.
[LibPyin](https://github.com/xstreck1/LibPyin) packages the same c4dm code shown above.

### Accuracy vs. ground truth

Synthetic harmonic signals with known f0 (vibrato, glides, gaps) at 22 050 Hz with hop 256.
Six 20 s signals per SNR. RPA is the fraction of voiced frames within 50 cents.

| implementation | SNR | RPA | voicing recall | voicing false alarm | median error |
|---|---:|---:|---:|---:|---:|
| **fastpitch** | 30 dB | **90.87%** | **92.68%** | 19.37% | 6.48 ¢ |
| librosa 1.0.0 | 30 dB | 90.87% | 92.68% | 19.37% | 6.48 ¢ |
| c4dm/pyin | 30 dB | 80.04% | 86.41% | 5.43% | 2.59 ¢ |
| Sleepwalking/libpyin | 30 dB | 74.62% | 88.13% | 11.79% | 12.40 ¢ |
| **fastpitch** | 10 dB | **90.30%** | **92.40%** | 19.36% | 7.37 ¢ |
| librosa 1.0.0 | 10 dB | 90.30% | 92.40% | 19.36% | 7.37 ¢ |
| c4dm/pyin | 10 dB | 79.15% | 82.67% | 4.82% | 3.87 ¢ |
| Sleepwalking/libpyin | 10 dB | 67.78% | 75.66% | 4.60% | 14.20 ¢ |

fastpitch and librosa are identical, as they should be. The variants differ by design:
c4dm uses a shorter YIN window and reports the raw candidate frequency, while librosa
reports 0.1-semitone bin centres. Setting `refine_f0 = true` reports the refined candidate
frequency of the decoded bin. That lowers the error a little without changing the path.

## Exactness

The speed comes from algorithmic restructuring, not approximation. Each claim below is
tested:

* **Frame-for-frame identical to `librosa.pyin` (librosa 1.0).** The test set covers 9
  signals (tones, chirps, vibrato, noisy note sequences, noise, silence, librosa's trumpet
  and speech examples), both Viterbi modes, plus a 90-case sweep. The sweep varies sample
  rate (16–48 kHz), frame and hop sizes, pitch resolution (0.05–⅓ semitone, including
  even-width transition kernels), transition rate, switch probability, threshold count and
  priors. Every case gives identical voicing and pitch bin in every frame, with
  `voiced_prob` within 2·10⁻¹⁵ (`tests/compare_librosa.py`).
* **Decoder vs. an independent dense Viterbi.** `tests/test_pyin.cpp` includes a
  transcription of librosa's O(T·S²) recursion (full transition matrix, `log(p + tiny)`,
  thresholded predecessor sets, first-index ties). It runs on 84 randomised and adversarial
  observation sequences: runs of `voiced_prob == 1`, silence, candidates at edge bins,
  probabilities of 10⁻⁴⁰ and octave jumps. Every decoded path has exactly the optimal
  log-likelihood. Where states differ, it is because the paths are exactly tied in real
  arithmetic: the dense recursion breaks such ties by floating-point rounding order, while
  fastpitch's exact grid arithmetic applies librosa's first-index rule.
* **The YIN stage matches a direct O(N²) float64 evaluation to ~10⁻¹⁵.**
* **librosa bug found along the way.** `librosa.core.pitch._cumulative_mean_normalized_difference`
  zeroes the cumulative-energy row before reading it. As a result d(1) misses the y[0]²
  term, which can shift YIN values by up to ~7·10⁻⁴. fastpitch reproduces this by default
  (`reproduce_librosa_quirks = true`) so results match librosa. Set it to `false` for the
  mathematically correct definition.
* `Pyin<double>` matches librosa's float64 pipeline exactly. `Pyin<float>` (the default) is
  as accurate as librosa's float32 pipeline: compared with the float64 truth it differed in
  1 of 1 715 frames. That frame's decision sat 1.8·10⁻⁵ of a bin from a rounding boundary,
  a tie at float32 resolution.
* Results do not depend on the thread count, SIMD backend (AVX2, SSE2, scalar all tested)
  or compiler (MSVC, GCC, Clang all tested).

## How it is fast

**YIN stage (~2 µs/frame at 22 kHz)**

* **Lane-batched FFT.** 4–8 frames are transformed at once, one per SIMD lane, so every
  butterfly runs at full vector width without shuffles. It is a Stockham mixed-radix
  (2/3/4/5) FFT with cache blocking for larger sizes, and runs at ~75 GFLOP/s per core.
  That is why there is no external FFT dependency.
* **Exact linear autocorrelation at minimal size.** The FFT length is the cheapest
  2ᵃ3ᵇ5ᶜ ≥ N + max_period (librosa pads to ~2N). The inverse transform of the real, even
  power spectrum is a DCT-I computed with a *half-length* real FFT, with compensated
  summation. Its pre-processing is fused into the power-spectrum pass.
* **Vectorised CMNDF and troughs.** The CMNDF is vectorised across frames, and trough
  detection uses compare + movemask.
* **Cheap priors.** Threshold lookup is O(1), and the threshold/Boltzmann priors are
  evaluated once per distinct set of troughs (via a bitmap) rather than once per threshold.

**Viterbi stage (~1.7 µs/frame for librosa's 1 202-state HMM)**

* **Banded max-plus convolution.** The row-normalised triangular transition band turns
  the O(S²) recursion into a max-plus convolution over the band.
* **Argmax for free.** State values live on a binary fixed-point grid inside doubles, and
  every kernel tap carries a unique code in the low bits. Sums are therefore exact, and a
  single `vmaxpd` yields both the maximum and the first maximising predecessor (librosa's
  tie rule). Grid rounding is ~4·10⁻⁹ nats.
* **Monotone argmax.** The log-triangle kernel is strictly concave, so the score matrix is
  Monge. The first argmax is then monotone in the destination bin, so blocks of
  destinations only scan sources between their neighbours' argmaxes. This cuts the work
  to ~30% of the band.
* **Provably exact pruning.**
  * Voiced states without a YIN candidate (`log p = log tiny`) are dominated by the
    unvoiced state of the same bin. They are skipped, with a rigorous margin test that
    falls back to dense frames when it does not hold.
  * States more than Λ below the frame maximum can never be on the optimal path. Λ is
    derived from the worst-case reachability cost, and this bound keeps the grid range
    exact.
  * Cross-voicing maxima are computed only at live voiced destinations.
* **Compact bookkeeping.** Destination values, back-pointers (stored as `uint16`) and
  normalisation are fused into one pass.

## Usage

### API

```cpp
namespace fastpitch {
struct PyinOptions {
  double sample_rate = 22050;
  double fmin = 65.406..., fmax = 2093.004...;  // C2, C7
  int frame_length = 2048, hop_length = 0;      // 0 -> frame_length / 4
  int n_thresholds = 100;
  double beta_a = 2, beta_b = 18, boltzmann_parameter = 2;
  double resolution = 0.1;                      // semitones per pitch bin
  double max_transition_rate = 35.92;           // octaves per second
  double switch_prob = 0.01, no_trough_prob = 0.01;
  double transition_min_prob = 1e-4;            // librosa >= 1.0 default; 0 = exact Viterbi
  bool center = true;                           // zero-pad frame_length/2 on both sides
  bool reproduce_librosa_quirks = true;         // see "Exactness"
  bool fill_unvoiced = true;                    // f0 = fill_value (NaN) when unvoiced
  double fill_value = NaN;
  bool refine_f0 = false;                       // candidate frequency instead of bin centre
  int num_threads = 1;                          // 0 = all hardware threads
};

template <class T> struct PyinResult {          // one entry per frame
  std::vector<T> f0;  std::vector<uint8_t> voiced;  std::vector<T> voiced_prob;
  double time(size_t i) const;                   // = librosa.times_like
};

template <class T = float> class Pyin {          // T = float or double samples
 public:
  explicit Pyin(const PyinOptions&);             // validates, throws std::invalid_argument
  PyinResult<T> operator()(const T* x, size_t n);
  PyinResult<T> operator()(const std::vector<T>& x);
  void analyze(const T* x, size_t n, PyinResult<T>& out);   // reuses out's storage
  size_t num_frames(size_t n_samples) const;
  int n_pitch_bins() const;  const std::vector<double>& bin_frequencies() const;
};

template <class T> PyinResult<T> pyin(const std::vector<T>& x, const PyinOptions& = {});
}
```

The parameters and their defaults have exactly librosa's meaning, so
`librosa.pyin(y, fmin=..., fmax=..., sr=..., frame_length=..., hop_length=..., ...)` maps
one-to-one. A `Pyin` instance is not thread-safe; create one per thread. `num_threads`
parallelises a single call internally.

### Building

Copy `include/fastpitch/pyin.hpp`, or use CMake (`find_package(fastpitch)` after install,
or `add_subdirectory`, then link `fastpitch::fastpitch`). Compile with optimisation and
the widest SIMD you can target:

| compiler | flags | backend |
|---|---|---|
| GCC / Clang | `-O3 -march=native` (or `-mavx2 -mfma`) | AVX2 + FMA |
| MSVC | `/O2 /arch:AVX2` | AVX2 + FMA |
| any x86-64 | (default) | SSE2 (~1.5× slower) |
| AArch64 | (default) | NEON (compile-checked; not run on hardware here) |
| any | `-DFASTPITCH_SIMD_SCALAR` | portable scalar (~3.5× slower) |

Do **not** compile with `-ffast-math`. The exact grid arithmetic, infinities and
compensated sums need IEEE semantics, and the header warns if fast-math is on.

### Memory

The Viterbi back-pointers take 2 bytes per pitch bin per frame: ~1.2 KB/frame with the
default 601 bins, or ~190 MB for an hour of audio at the default hop. librosa's dense
decoder stores more than an order of magnitude more (a float64 value and a pointer for
every state, plus the dense observation matrix).

## Repository layout

```
include/fastpitch/pyin.hpp   the library
examples/basic.cpp           minimal usage
tools/fastpitch_cli.cpp      raw-float32 CLI used by the scripts
tests/test_pyin.cpp          unit tests incl. dense reference Viterbi (ctest)
tests/test_fft.cpp           FFT vs naive DFT, all radices, flat and blocked (ctest)
tests/compare_librosa.py     frame-by-frame parity with librosa (+ --sweep)
bench/run_benchmarks.py      speed + accuracy tables above
bench/compete_*.c(pp)        harnesses for the original C++ pYIN and libpyin
bench/fetch_competitors.sh   fetches those (pinned revisions; GPL code, not part of fastpitch)
bench/stage_profile.cpp      per-stage timing
```

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build
pip install librosa && python tests/compare_librosa.py build/fastpitch_cli --sweep
sh bench/fetch_competitors.sh
cmake -S . -B build -DFASTPITCH_BUILD_COMPETITORS=ON && cmake --build build
python bench/run_benchmarks.py --fastpitch build/fastpitch_cli --c4dm build/compete_c4dm \
       --sw build/compete_sw --threads 8
```

## References

* M. Mauch and S. Dixon, "pYIN: A fundamental frequency estimator using probabilistic
  threshold distributions," ICASSP 2014.
* A. de Cheveigné and H. Kawahara, "YIN, a fundamental frequency estimator for speech and
  music," JASA 111(4), 2002.
* librosa `pyin` (librosa ≥ 1.0) is the behavioural reference.
