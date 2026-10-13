// Benchmark harness for the original C++ pYIN (c4dm/pyin, Mauch & Dixon), pitch-track path.
// Mirrors PYinVamp::process() + getRemainingFeatures() (smoothed pitch track, full Viterbi).
//
//   compete_c4dm <in.f32> <out.f64> sr=<sr> [frame=2048] [hop=256] [repeat=N]
#include "Yin.h"
#include "MonoPitchHMM.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  FILE* f = std::fopen(argv[1], "rb");
  if (!f) return 1;
  std::vector<float> x;
  float buf[4096];
  size_t k;
  while ((k = std::fread(buf, 4, 4096, f)) > 0) x.insert(x.end(), buf, buf + k);
  std::fclose(f);
  double sr = 44100;
  int block = 2048, step = 256, repeat = 1;
  for (int i = 3; i < argc; ++i) {
    std::string a = argv[i];
    const double v = std::atof(a.c_str() + a.find('=') + 1);
    if (a.rfind("sr=", 0) == 0) sr = v;
    if (a.rfind("frame=", 0) == 0) block = int(v);
    if (a.rfind("hop=", 0) == 0) step = int(v);
    if (a.rfind("repeat=", 0) == 0) repeat = int(v);
  }
  const float lowAmp = 0.1f;  // PYinVamp default
  std::vector<float> track;
  double best = 1e30;
  for (int rep = 0; rep < repeat; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    Yin yin(block, size_t(sr), 0.0, true);
    yin.setThresholdDistr(2.0f);
    yin.setFrameSize(block);
    yin.setFast(true);
    MonoPitchHMM hmm(0);
    std::vector<std::vector<std::pair<double, double>>> pitchProb;
    std::vector<double> frame(block);
    // Vamp hosts feed zero-padded blocks starting at sample 0.
    for (size_t start = 0; start < x.size(); start += step) {
      float rms = 0;
      for (int i = 0; i < block; ++i) {
        const float s = start + i < x.size() ? x[start + i] : 0.f;
        frame[i] = s;
        rms += s * s;
      }
      rms = std::sqrt(rms / block);
      const bool low = rms < lowAmp;
      Yin::YinOutput yo = yin.processProbabilisticYin(frame.data());
      std::vector<std::pair<double, double>> pp;
      for (auto& c : yo.freqProb) {
        const double pitch = 12 * std::log(c.first / 440) / std::log(2.) + 69;
        if (!low) pp.push_back({pitch, c.second});
        else pp.push_back({pitch, c.second * ((rms + 0.01 * lowAmp) / (1.01 * lowAmp))});
      }
      std::vector<double> obs = hmm.calculateObsProb(pp);
      if (pitchProb.empty()) hmm.initialise(obs);
      else hmm.process(obs);
      pitchProb.push_back(pp);
    }
    std::vector<int> path = hmm.track();
    track.clear();
    for (size_t i = 0; i < path.size(); ++i) track.push_back(hmm.nearestFreq(path[i], pitchProb[i]));
    const auto t1 = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
  }
  std::fprintf(stderr, "c4dm-pyin frames=%zu time=%.3f ms (%.0fx realtime)\n", track.size(),
               best * 1e3, x.size() / sr / best);
  FILE* o = std::fopen(argv[2], "wb");
  for (float v : track) { double d = v; std::fwrite(&d, 8, 1, o); }
  std::fclose(o);
}
