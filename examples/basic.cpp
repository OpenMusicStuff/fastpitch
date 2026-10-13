// Minimal fastpitch example: track the pitch of a synthetic vibrato tone.
#include <fastpitch/pyin.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
  const double sr = 22050.0;
  std::vector<float> x(std::size_t(2.0 * sr));
  double phase = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double t = double(i) / sr;
    const double f = 220.0 * std::pow(2.0, 0.5 / 12.0 * std::sin(2 * 3.14159265358979 * 5.0 * t));
    phase += 2 * 3.14159265358979 * f / sr;
    x[i] = float(0.5 * std::sin(phase) + 0.25 * std::sin(2 * phase));
  }

  fastpitch::PyinOptions opt;  // librosa.pyin defaults: fmin=C2, fmax=C7, frame=2048, hop=512
  opt.sample_rate = sr;
  opt.num_threads = 0;         // use all cores (results do not depend on the thread count)

  fastpitch::Pyin<float> tracker(opt);  // reusable; keeps its buffers between calls
  const fastpitch::PyinResult<float> r = tracker(x);

  for (std::size_t i = 0; i < r.size(); i += 4)
    std::printf("t=%6.3f s  f0=%8.2f Hz  voiced=%d  p(voiced)=%.3f\n", r.time(i), r.f0[i],
                int(r.voiced[i]), r.voiced_prob[i]);
}
