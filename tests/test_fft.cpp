// Correctness (vs. naive DFT) and throughput of the lane-batched FFT.
#include <fastpitch/pyin.hpp>

#include <chrono>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

using namespace fastpitch::detail;

template <class T>
static int check(int n, bool blocked, bool bench) {
  using V = simd::native_t<T>;
  constexpr int W = V::size;
  VecFFT<V> fft;
  fft.init(n, blocked);
  AlignedBuffer<T> a(std::size_t(n) * 2 * W), b(std::size_t(n) * 2 * W), in(std::size_t(n) * 2 * W);
  std::mt19937 rng(n);
  std::normal_distribution<double> nd;
  for (std::size_t i = 0; i < in.size(); ++i) in[i] = T(nd(rng));
  std::copy(in.data(), in.data() + in.size(), a.data());
  const T* r = fft.forward(a.data(), b.data());
  double err = 0, ref = 0;
  const double pi = 3.14159265358979323846;
  for (int l = 0; l < W; ++l)
    for (int k = 0; k < n; ++k) {
      std::complex<double> acc = 0;
      for (int j = 0; j < n; ++j) {
        const std::complex<double> xj(in[std::size_t(j) * 2 * W + l], in[std::size_t(j) * 2 * W + W + l]);
        acc += xj * std::polar(1.0, -2 * pi * double(j) * k / n);
      }
      const std::complex<double> got(r[std::size_t(k) * 2 * W + l], r[std::size_t(k) * 2 * W + W + l]);
      err = std::max(err, std::abs(got - acc));
      ref = std::max(ref, std::abs(acc));
    }
  const double rel = err / ref;
  const double tol = sizeof(T) == 4 ? 2e-6 : 1e-12;
  double ns = 0;
  if (bench) {
    const int reps = 2000;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
      T* x = i & 1 ? b.data() : a.data();
      fft.forward(x, x == a.data() ? b.data() : a.data());
    }
    auto t1 = std::chrono::steady_clock::now();
    ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / reps / W;
  }
  std::printf("%-6s n=%5d %-7s rel.err=%.2e %s  %8.1f ns/transform/lane  %.2f ns/(point*log2n)\n",
              sizeof(T) == 4 ? "float" : "double", n, blocked ? "blocked" : "flat", rel,
              rel < tol ? "OK " : "BAD", ns, ns / (n * std::log2(double(n))));
  return rel < tol ? 0 : 1;
}

int main(int argc, char**) {
  const bool bench = argc > 1;
  int fails = 0;
  for (int n : {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 25, 27, 30, 32, 45, 60, 64, 100, 125,
                128, 243, 256, 360, 512, 625, 1000, 1024, 1200, 1280, 1440, 1500, 1536, 2048}) {
    for (bool blk : {false, true}) {
      fails += check<float>(n, blk, bench && n >= 512);
      if (n <= 1500) fails += check<double>(n, blk, false);
    }
  }
  std::printf(fails ? "FAILED (%d)\n" : "ALL FFT TESTS PASSED\n", fails);
  return fails != 0;
}
