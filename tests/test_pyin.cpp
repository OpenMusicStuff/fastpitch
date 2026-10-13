// Unit tests for fastpitch (no external dependencies).
//
//  * Decoder vs. an independent dense O(T S^2) Viterbi reference (librosa's recursion) on
//    randomised and adversarial observation sequences, in both transition modes.
//  * Pitch accuracy on synthetic tones, determinism across thread counts and instance reuse,
//    float/double agreement, edge cases (empty/short input, silence, NaN), option validation.
#include <fastpitch/pyin.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

int g_failures = 0, g_checks = 0;

#define CHECK(cond)                                                                   \
  do {                                                                                \
    ++g_checks;                                                                       \
    if (!(cond)) {                                                                    \
      ++g_failures;                                                                   \
      std::printf("  FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);                 \
    }                                                                                 \
  } while (0)

void run(const char* name, const std::function<void()>& f) {
  const int before = g_failures;
  f();
  std::printf("[%s] %s\n", g_failures == before ? " ok " : "FAIL", name);
}

using fastpitch::PyinOptions;
using fastpitch::detail::Cand;
using fastpitch::detail::FrameObs;
using fastpitch::detail::Model;

// ---------------------------------------------------------------------------------------
// Reference: dense Viterbi exactly as librosa.sequence.viterbi (log(p + tiny) everywhere,
// predecessor sets thresholded on log transition probability, first-index ties).
// ---------------------------------------------------------------------------------------
std::vector<int> reference_viterbi(const Model& m, const std::vector<FrameObs>& obs,
                                   const std::vector<Cand>& cands,
                                   const std::vector<int>* score_of = nullptr,
                                   double* scores = nullptr) {
  const int n = m.nbins, S = 2 * n;
  const double tiny = std::numeric_limits<double>::min();
  // Transition matrix rebuilt independently from librosa's definition.
  const double ms = std::nearbyint(m.opt.max_transition_rate * 12.0 * m.hop / m.opt.sample_rate);
  const int width = int(ms) * m.nbps + 1;
  std::vector<double> win(width);
  for (int i = 0; i < width; ++i) {
    // scipy.signal.windows.triang(width, sym=True)
    const int half = (width + 1) / 2;
    const int k = i < half ? i + 1 : width - i;
    win[i] = (width % 2 == 0) ? (2.0 * k - 1.0) / width : 2.0 * k / (width + 1.0);
  }
  std::vector<double> T(std::size_t(n) * n, 0.0);
  for (int i = 0; i < n; ++i) {
    // pad_center + roll + knock out off-band entries
    std::vector<double> row(n, 0.0);
    const int lpad = (n - width) / 2;
    for (int q = 0; q < width; ++q) row[(lpad + q + n / 2 + i + 1) % n] = win[q];
    for (int j = std::min(n, i + width / 2 + 1); j < n; ++j) row[j] = 0;
    for (int j = 0; j < std::max(0, i - width / 2); ++j) row[j] = 0;
    double z = 0;
    for (double v : row) z += v;
    for (int j = 0; j < n; ++j) T[std::size_t(i) * n + j] = row[j] / z;
  }
  const double p_stay = 1.0 - m.opt.switch_prob, p_switch = (1.0 - p_stay) / 1.0;
  std::vector<double> lt(std::size_t(S) * S);
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b)
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
          lt[std::size_t(a * n + i) * S + (b * n + j)] =
              std::log((a == b ? p_stay : p_switch) * T[std::size_t(i) * n + j] + tiny);
  const bool thresholded = m.opt.transition_min_prob > 0;
  const double lthr = std::log(m.opt.transition_min_prob + tiny);
  const double lpi = std::log(1.0 / S + tiny);
  const std::size_t nt = obs.size();
  std::vector<double> lp(S), val(S), nv(S);
  std::vector<std::vector<int>> ptr(nt, std::vector<int>(S, 0));
  auto load = [&](std::size_t t) {
    const FrameObs& fo = obs[t];
    for (int j = 0; j < n; ++j) lp[j] = std::log(0.0 + tiny);
    for (std::uint32_t c = fo.cbegin; c < fo.cend; ++c) lp[cands[c].bin] = cands[c].logp;
    for (int j = 0; j < n; ++j) lp[n + j] = fo.log_unvoiced;
  };
  load(0);
  for (int s = 0; s < S; ++s) val[s] = lp[s] + lpi;
  for (std::size_t t = 1; t < nt; ++t) {
    load(t);
    for (int j = 0; j < S; ++j) {
      double best = -std::numeric_limits<double>::infinity();
      int arg = 0;
      for (int k = 0; k < S; ++k) {
        const double l = lt[std::size_t(k) * S + j];
        if (thresholded && !(l >= lthr)) continue;
        const double c = val[k] + l;
        if (c > best) {
          best = c;
          arg = k;
        }
      }
      ptr[t][j] = arg;
      nv[j] = lp[j] + best;
    }
    val.swap(nv);
  }
  std::vector<int> states(nt);
  int s = int(std::max_element(val.begin(), val.end()) - val.begin());
  for (std::size_t t = nt; t-- > 0;) {
    states[t] = s;
    if (t) s = ptr[t][s];
  }
  // Log-likelihood of a path under this (dense, librosa) model; -inf if a transition is
  // not admissible.
  if (score_of) {
    for (int which = 0; which < 2; ++which) {
      const std::vector<int>& path = which ? *score_of : states;
      double sc = 0;
      for (std::size_t t = 0; t < nt; ++t) {
        load(t);
        sc += lp[path[t]];
        if (t == 0) {
          sc += lpi;
        } else {
          const double l = lt[std::size_t(path[t - 1]) * S + path[t]];
          if (thresholded && !(l >= lthr)) sc = -std::numeric_limits<double>::infinity();
          sc += l;
        }
      }
      (which ? scores[1] : scores[0]) = sc;
    }
  }
  return states;
}

std::vector<int> fast_viterbi(const Model& m, const std::vector<FrameObs>& obs,
                              const std::vector<Cand>& cands) {
  fastpitch::detail::Decoder dec(m);
  dec.reset(obs.size());
  for (const FrameObs& fo : obs) dec.push(fo, cands.data());
  std::vector<std::int32_t> st(obs.size());
  dec.finish(st.data());
  return std::vector<int>(st.begin(), st.end());
}

// Random observation sequences with the structure produced by the YIN stage, plus
// adversarial frames (voiced_prob == 1, no candidates, edge bins, tiny probabilities).
void make_sequence(const Model& m, std::mt19937& rng, int frames, int style,
                   std::vector<FrameObs>& obs, std::vector<Cand>& cands) {
  const int n = m.nbins;
  const double tiny = std::numeric_limits<double>::min();
  std::uniform_real_distribution<double> U(0, 1);
  obs.clear();
  cands.clear();
  double pitch = U(rng) * (n - 1);
  for (int t = 0; t < frames; ++t) {
    FrameObs fo{};
    fo.cbegin = std::uint32_t(cands.size());
    int kind = int(U(rng) * 10);
    if (style == 1) kind = (t / 7) % 3 == 0 ? 9 : kind;  // runs of voiced_prob == 1
    if (style == 2) kind = U(rng) < 0.5 ? 0 : kind;      // many silent frames
    std::vector<std::pair<int, double>> c;
    if (kind != 0) {
      pitch += (U(rng) - 0.5) * (style == 3 ? 400 : 30);
      pitch = std::min(std::max(pitch, 0.0), double(n - 1));
      const int nc = 1 + int(U(rng) * 6);
      double total = kind == 9 ? 1.0 : U(rng);
      for (int k = 0; k < nc; ++k) {
        int b = k == 0 ? int(pitch) : int(U(rng) * n);
        if (U(rng) < 0.05) b = U(rng) < 0.5 ? 0 : n - 1;
        double p = k == 0 ? total * (0.5 + 0.5 * U(rng)) : total * U(rng) * 0.3;
        if (U(rng) < 0.05) p = 1e-40;
        c.push_back({b, p});
      }
    }
    // bins strictly decreasing, unique (last write wins like librosa)
    std::sort(c.begin(), c.end(), [](auto& a, auto& b) { return a.first > b.first; });
    double vp = 0;
    for (std::size_t k = 0; k < c.size(); ++k) {
      if (k + 1 < c.size() && c[k + 1].first == c[k].first) continue;
      cands.push_back({c[k].first, std::log(c[k].second + tiny), 0.0});
      vp += c[k].second;
    }
    if (kind == 9) vp = 1.0;
    vp = std::min(std::max(vp, 0.0), 1.0);
    fo.cend = std::uint32_t(cands.size());
    fo.voiced_prob = vp;
    fo.log_unvoiced = std::log((1.0 - vp) / n + tiny);
    obs.push_back(fo);
  }
}

std::vector<float> tone(double f0, double sr, double seconds, double amp = 0.4) {
  std::vector<float> x(std::size_t(sr * seconds));
  double ph = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double v = std::sin(ph) + 0.5 * std::sin(2 * ph) + 0.25 * std::sin(3 * ph);
    x[i] = float(amp * v / 1.75);
    ph += 2 * 3.14159265358979323846 * f0 / sr;
  }
  return x;
}

}  // namespace

int main() {
  std::printf("fastpitch tests (SIMD: %s)\n", fastpitch::Pyin<float>::simd_isa());

  run("decoder == dense reference Viterbi (randomised, adversarial)", [] {
    std::mt19937 rng(12345);
    struct Cfg { double sr, fmin, fmax; int hop; double res, rate, sw, thr; };
    const Cfg cfgs[] = {
        {22050, 200, 800, 512, 0.1, 35.92, 0.01, 1e-4},    // default model, narrower range
        {22050, 200, 800, 512, 0.1, 35.92, 0.01, 0.0},     // exact (full) Viterbi
        {44100, 100, 1000, 256, 0.2, 35.92, 0.01, 1e-4},   // narrow band
        {22050, 150, 600, 512, 1 / 3., 35.92, 0.01, 1e-4}, // even transition width
        {22050, 150, 600, 512, 1 / 3., 35.92, 0.01, 0.0},
        {22050, 300, 900, 300, 0.1, 20.0, 0.2, 1e-4},      // frequent switching
        {16000, 80, 400, 160, 0.05, 35.92, 0.01, 1e-4},    // wide band (~31 bins/frame)
    };
    int frames_checked = 0, mismatches = 0, observable = 0, suboptimal = 0;
    for (const Cfg& c : cfgs) {
      PyinOptions o;
      o.sample_rate = c.sr;
      o.fmin = c.fmin;
      o.fmax = c.fmax;
      o.frame_length = 2048;
      o.hop_length = c.hop;
      o.resolution = c.res;
      o.max_transition_rate = c.rate;
      o.switch_prob = c.sw;
      o.transition_min_prob = c.thr;
      const Model m(o);
      for (int style = 0; style < 4; ++style)
        for (int rep = 0; rep < 3; ++rep) {
          std::vector<FrameObs> obs;
          std::vector<Cand> cands;
          make_sequence(m, rng, 60, style, obs, cands);
          const auto got = fast_viterbi(m, obs, cands);
          double sc[2];
          const auto ref = reference_viterbi(m, obs, cands, &got, sc);
          frames_checked += int(ref.size());
          int mm = 0, obsdiff = 0;
          for (std::size_t t = 0; t < ref.size(); ++t) {
            mm += ref[t] != got[t];
            const bool vr = ref[t] < m.nbins, vg = got[t] < m.nbins;
            obsdiff += vr != vg || (vr && ref[t] != got[t]);
          }
          mismatches += mm;
          observable += obsdiff;
          // Paths may differ only through exact ties: both must be optimal.
          const double tol = 1e-9 * std::max(1.0, std::abs(sc[0]));
          const bool optimal = std::abs(sc[0] - sc[1]) <= tol;
          suboptimal += !optimal;
          if ((mm || !optimal) && std::getenv("FP_VERBOSE"))
            std::printf("    sr=%g res=%g thr=%g sw=%g style=%d rep=%d: %d differing states, "
                        "loglik ref %.12f fast %.12f\n",
                        c.sr, c.res, c.thr, c.sw, style, rep, mm, sc[0], sc[1]);
        }
    }
    std::printf("    %d frames: %d states differ from the reference (%d observable), "
                "%d sequences with a non-optimal path\n",
                frames_checked, mismatches, observable, suboptimal);
    CHECK(suboptimal == 0);
  });

  run("pitch accuracy on harmonic tones (C2..C7)", [] {
    for (double sr : {22050.0, 44100.0}) {
      PyinOptions o;
      o.sample_rate = sr;
      o.hop_length = sr > 30000 ? 256 : 512;
      fastpitch::Pyin<float> p(o);
      for (double midi = 38; midi <= 94; midi += 4.37) {
        const double f = 440.0 * std::pow(2.0, (midi - 69) / 12);
        auto x = tone(f, sr, 1.0);
        auto r = p(x);
        int voiced = 0, inside = 0;
        double worst = 0;
        for (std::size_t t = 4; t + 4 < r.size(); ++t) {
          ++inside;
          if (!r.voiced[t]) continue;
          ++voiced;
          worst = std::max(worst, std::abs(1200 * std::log2(r.f0[t] / f)));
        }
        CHECK(voiced == inside);
        // Bin quantisation (+-5 cents) plus pYIN's own estimation error on low tones.
        CHECK(worst <= 25.0);
      }
    }
  });

  run("results independent of thread count and instance reuse", [] {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0, 0.05f);
    auto x = tone(173.0, 22050, 6.0);
    auto y = tone(331.0, 22050, 3.0);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += nd(rng);
    x.insert(x.end(), y.begin(), y.end());
    PyinOptions o;
    fastpitch::Pyin<float> base(o);
    const auto ref = base(x);
    for (int th : {2, 3, 8, 0}) {
      o.num_threads = th;
      fastpitch::Pyin<float> p(o);
      for (int rep = 0; rep < 2; ++rep) {
        const auto r = p(x);
        CHECK(r.voiced == ref.voiced);
        CHECK(r.voiced_prob == ref.voiced_prob);
        bool same = r.size() == ref.size();
        for (std::size_t t = 0; same && t < r.size(); ++t)
          same = (std::isnan(r.f0[t]) && std::isnan(ref.f0[t])) || r.f0[t] == ref.f0[t];
        CHECK(same);
        p(y);  // interleave a different length
      }
    }
  });

  run("refine_f0: same path, frequencies closer to the truth", [] {
    double worst_bin = 0, worst_ref = 0;
    for (double f : {82.4, 146.8, 261.6, 523.3, 987.8}) {
      auto x = tone(f, 22050, 1.0);
      PyinOptions o;
      const auto a = fastpitch::pyin(x, o);
      o.refine_f0 = true;
      const auto b = fastpitch::pyin(x, o);
      CHECK(a.voiced == b.voiced);
      for (std::size_t t = 4; t + 4 < a.size(); ++t) {
        if (!a.voiced[t]) continue;
        worst_bin = std::max(worst_bin, std::abs(1200 * std::log2(a.f0[t] / f)));
        worst_ref = std::max(worst_ref, std::abs(1200 * std::log2(b.f0[t] / f)));
        CHECK(std::abs(1200 * std::log2(b.f0[t] / a.f0[t])) <= 5.0 + 1e-3);  // within its bin
      }
    }
    std::printf("    worst error: bin centre %.2f cents, refined %.2f cents\n", worst_bin, worst_ref);
    CHECK(worst_ref <= worst_bin);
  });

  run("float and double pipelines agree on clean signals", [] {
    auto x = tone(220.0, 22050, 2.0);
    std::vector<double> xd(x.begin(), x.end());
    const auto a = fastpitch::pyin(x);
    const auto b = fastpitch::pyin(xd);
    CHECK(a.size() == b.size());
    int diff = 0;
    for (std::size_t t = 0; t < a.size(); ++t)
      diff += a.voiced[t] != b.voiced[t] ||
              (a.voiced[t] && std::abs(double(a.f0[t]) - b.f0[t]) > 1e-3 * b.f0[t]);
    CHECK(diff == 0);
  });

  run("frame count, empty, short, center=false", [] {
    PyinOptions o;
    fastpitch::Pyin<float> p(o);
    std::vector<float> empty;
    CHECK(p(empty).size() == 1);  // librosa pads N/2 on both sides: 1 frame of zeros
    std::vector<float> one(1, 0.5f);
    CHECK(p(one).size() == 1);
    std::vector<float> x(22050, 0.f);
    CHECK(p(x).size() == 1 + 22050 / 512);
    CHECK(p.num_frames(22050) == 1 + 22050 / 512);
    o.center = false;
    fastpitch::Pyin<float> q(o);
    CHECK(q(one).size() == 0);
    CHECK(q(x).size() == 1 + (22050 - 2048) / 512);
    const auto r = q(x);
    CHECK(std::abs(r.time(3) - 3 * 512 / 22050.0) < 1e-12);
  });

  run("silence and NaN input", [] {
    std::vector<float> z(44100, 0.f);
    const auto r = fastpitch::pyin(z);
    bool ok = true;
    for (std::size_t t = 0; t < r.size(); ++t) ok = ok && !r.voiced[t] && r.voiced_prob[t] == 0;
    CHECK(ok);
    auto x = tone(220.0, 22050, 2.0);
    x[22050] = std::numeric_limits<float>::quiet_NaN();
    const auto s = fastpitch::pyin(x);
    CHECK(s.size() == fastpitch::Pyin<float>().num_frames(x.size()));
    CHECK(s.voiced[10] && s.voiced[s.size() - 10]);  // frames away from the NaN unaffected
  });

  run("options: exact Viterbi, quirks off, fill_unvoiced off", [] {
    auto x = tone(150.0, 22050, 1.5);
    x.resize(x.size() + 11025, 0.f);
    PyinOptions o;
    o.transition_min_prob = 0;
    const auto a = fastpitch::pyin(x, o);
    o.transition_min_prob = 1e-4;
    o.reproduce_librosa_quirks = false;
    const auto b = fastpitch::pyin(x, o);
    o.fill_unvoiced = false;
    const auto c = fastpitch::pyin(x, o);
    CHECK(a.size() == b.size() && b.size() == c.size());
    bool finite = true, nan_unvoiced = true;
    for (std::size_t t = 0; t < c.size(); ++t) {
      finite = finite && std::isfinite(c.f0[t]);
      if (!b.voiced[t]) nan_unvoiced = nan_unvoiced && std::isnan(b.f0[t]);
    }
    CHECK(finite);
    CHECK(nan_unvoiced);
  });

  run("invalid options throw", [] {
    auto throws = [](std::function<void(PyinOptions&)> f) {
      PyinOptions o;
      f(o);
      try {
        fastpitch::Pyin<float> p(o);
      } catch (const std::invalid_argument&) {
        return true;
      }
      return false;
    };
    CHECK(throws([](PyinOptions& o) { o.fmin = 0; }));
    CHECK(throws([](PyinOptions& o) { o.fmin = 900; o.fmax = 800; }));
    CHECK(throws([](PyinOptions& o) { o.fmax = 20000; }));
    CHECK(throws([](PyinOptions& o) { o.fmin = 5; }));  // period does not fit the frame
    CHECK(throws([](PyinOptions& o) { o.switch_prob = 1.5; }));
    CHECK(throws([](PyinOptions& o) { o.resolution = 0; }));
    CHECK(throws([](PyinOptions& o) { o.n_thresholds = 0; }));
    CHECK(throws([](PyinOptions& o) { o.transition_min_prob = 0.9; }));  // empty transition set
  });

  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures != 0;
}
