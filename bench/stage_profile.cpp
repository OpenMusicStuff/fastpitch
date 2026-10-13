// Per-stage timing of the fastpitch pipeline (YIN stage vs. Viterbi decoder).
#include <fastpitch/pyin.hpp>

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "build/long.f32";
  FILE* f = std::fopen(path, "rb");
  if (!f) return 1;
  std::vector<float> x;
  float buf[4096];
  size_t k;
  while ((k = std::fread(buf, 4, 4096, f)) > 0) x.insert(x.end(), buf, buf + k);
  std::fclose(f);

  fastpitch::PyinOptions opt;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    const double v = std::atof(a.c_str() + a.find('=') + 1);
    if (a.rfind("thr=", 0) == 0) opt.transition_min_prob = v;
    if (a.rfind("sr=", 0) == 0) opt.sample_rate = v;
    if (a.rfind("hop=", 0) == 0) opt.hop_length = int(v);
    if (a.rfind("frame=", 0) == 0) opt.frame_length = int(v);
  }
  using T = float;
  fastpitch::detail::Model m(opt);
  fastpitch::detail::YinWorker<T> w(m);
  using B = fastpitch::detail::YinWorker<T>::BatchT;
  constexpr int W = fastpitch::detail::YinWorker<T>::W;
  const size_t nf = m.num_frames(x.size()), nb = (nf + W - 1) / W;
  std::vector<B> batches(nb);
  double tyin = 1e30, tvit = 1e30;
  fastpitch::detail::Decoder dec(m);
  for (int rep = 0; rep < 5; ++rep) {
    auto t0 = clk::now();
    for (size_t b = 0; b < nb; ++b)
      w.run(x.data(), x.size(), b * W, int(std::min<size_t>(W, nf - b * W)), batches[b]);
    auto t1 = clk::now();
    dec.reset(nf);
    for (size_t b = 0; b < nb; ++b)
      for (int l = 0; l < batches[b].count; ++l) dec.push(batches[b].frames[l], batches[b].cands.data());
    std::vector<std::int32_t> st(nf);
    dec.finish(st.data());
    auto t2 = clk::now();
    tyin = std::min(tyin, std::chrono::duration<double>(t1 - t0).count());
    tvit = std::min(tvit, std::chrono::duration<double>(t2 - t1).count());
  }
  size_t cands = 0;
  for (auto& b : batches) cands += b.cands.size();
  std::printf("frames=%zu  L=%d  nbins=%d  cands/frame=%.2f\n", nf, m.L, m.nbins, double(cands) / nf);
  std::printf("yin    : %8.3f ms  (%.3f us/frame)\n", tyin * 1e3, tyin * 1e6 / nf);
  std::printf("viterbi: %8.3f ms  (%.3f us/frame)\n", tvit * 1e3, tvit * 1e6 / nf);
}
