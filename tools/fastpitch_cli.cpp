// Command-line front end used by the validation scripts.
//
//   fastpitch_cli <in.f32> <out.bin> [key=value ...]
//
// in.f32  : raw little-endian float32 mono samples
// out.bin : int64 n_frames, then f0[n] (f64), voiced_prob[n] (f64), voiced[n] (u8)
// keys    : sr fmin fmax frame hop thr (transition_min_prob) threads repeat double=0|1
#include <fastpitch/pyin.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

template <class T>
static int run(const std::vector<float>& xf, const fastpitch::PyinOptions& opt, int repeat,
               const char* out_path) {
  std::vector<T> x(xf.begin(), xf.end());
  fastpitch::Pyin<T> p(opt);
  fastpitch::PyinResult<T> r;
  double best = 1e30;
  for (int i = 0; i < repeat; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    p.analyze(x.data(), x.size(), r);
    const auto t1 = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
  }
  const double dur = double(x.size()) / opt.sample_rate;
  std::fprintf(stderr, "isa=%s fft=%d frames=%zu time=%.3f ms  (%.0fx realtime)\n",
               fastpitch::Pyin<T>::simd_isa(), p.fft_length(), r.size(), best * 1e3, dur / best);
  FILE* f = std::fopen(out_path, "wb");
  if (!f) return 1;
  const long long n = (long long)r.size();
  std::fwrite(&n, 8, 1, f);
  for (auto v : r.f0) { double d = v; std::fwrite(&d, 8, 1, f); }
  for (auto v : r.voiced_prob) { double d = v; std::fwrite(&d, 8, 1, f); }
  std::fwrite(r.voiced.data(), 1, r.voiced.size(), f);
  std::fclose(f);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s in.f32 out.bin [sr= fmin= fmax= frame= hop= thr= threads= repeat= double=]\n", argv[0]);
    return 2;
  }
  FILE* f = std::fopen(argv[1], "rb");
  if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
  std::vector<float> x;
  float buf[4096];
  size_t k;
  while ((k = std::fread(buf, 4, 4096, f)) > 0) x.insert(x.end(), buf, buf + k);
  std::fclose(f);

  fastpitch::PyinOptions opt;
  int repeat = 1, dbl = 0;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    const size_t eq = a.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = a.substr(0, eq);
    const double v = std::atof(a.c_str() + eq + 1);
    if (key == "sr") opt.sample_rate = v;
    else if (key == "fmin") opt.fmin = v;
    else if (key == "fmax") opt.fmax = v;
    else if (key == "frame") opt.frame_length = int(v);
    else if (key == "hop") opt.hop_length = int(v);
    else if (key == "thr") opt.transition_min_prob = v;
    else if (key == "threads") opt.num_threads = int(v);
    else if (key == "repeat") repeat = int(v);
    else if (key == "double") dbl = int(v);
    else if (key == "center") opt.center = v != 0;
    else if (key == "res") opt.resolution = v;
    else if (key == "rate") opt.max_transition_rate = v;
    else if (key == "switch") opt.switch_prob = v;
    else if (key == "nthr") opt.n_thresholds = int(v);
    else if (key == "ba") opt.beta_a = v;
    else if (key == "bb") opt.beta_b = v;
    else if (key == "boltz") opt.boltzmann_parameter = v;
    else if (key == "notrough") opt.no_trough_prob = v;
    else if (key == "refine") opt.refine_f0 = v != 0;
    else if (key == "quirks") opt.reproduce_librosa_quirks = v != 0;
    else { std::fprintf(stderr, "unknown key %s\n", key.c_str()); return 2; }
  }
  try {
    return dbl ? run<double>(x, opt, repeat, argv[2]) : run<float>(x, opt, repeat, argv[2]);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
