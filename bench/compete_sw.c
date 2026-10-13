// Benchmark harness for Sleepwalking/libpyin (C pYIN variant by Kanru Hua, libgvps Viterbi).
//
//   compete_sw <in.f32> <out.f64> sr=<sr> [frame=2048] [hop=256] [fmin=..] [fmax=..] [repeat=N]
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>

#include "pyin.h"

static double now(void) {
  LARGE_INTEGER f, c;
  QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  return (double)c.QuadPart / (double)f.QuadPart;
}

int main(int argc, char** argv) {
  if (argc < 3) return 2;
  FILE* f = fopen(argv[1], "rb");
  if (!f) return 1;
  fseek(f, 0, SEEK_END);
  long bytes = ftell(f);
  fseek(f, 0, SEEK_SET);
  int nx = (int)(bytes / 4);
  float* x = (float*)malloc(sizeof(float) * nx);
  if (fread(x, 4, nx, f) != (size_t)nx) return 1;
  fclose(f);
  double sr = 44100, fmin = 61.735, fmax = 3300;
  int frame = 2048, hop = 256, repeat = 1;
  for (int i = 3; i < argc; ++i) {
    const char* eq = strchr(argv[i], '=');
    if (!eq) continue;
    double v = atof(eq + 1);
    if (!strncmp(argv[i], "sr=", 3)) sr = v;
    if (!strncmp(argv[i], "frame=", 6)) frame = (int)v;
    if (!strncmp(argv[i], "hop=", 4)) hop = (int)v;
    if (!strncmp(argv[i], "fmin=", 5)) fmin = v;
    if (!strncmp(argv[i], "fmax=", 5)) fmax = v;
    if (!strncmp(argv[i], "repeat=", 7)) repeat = (int)v;
  }
  pyin_config param = pyin_init(hop);
  param.fmin = (FP_TYPE)fmin;
  param.fmax = (FP_TYPE)fmax;
  param.nf = frame;
  param.w = frame / 2;
  param.trange = pyin_trange(param.nq, param.fmin, param.fmax);
  double best = 1e30;
  int nfrm = 0;
  FP_TYPE* f0 = NULL;
  for (int r = 0; r < repeat; ++r) {
    if (f0) free(f0);
    double t0 = now();
    f0 = pyin_analyze(param, x, nx, (FP_TYPE)sr, &nfrm);
    double t1 = now();
    if (t1 - t0 < best) best = t1 - t0;
  }
  fprintf(stderr, "sw-libpyin frames=%d time=%.3f ms (%.0fx realtime)\n", nfrm, best * 1e3,
          nx / sr / best);
  FILE* o = fopen(argv[2], "wb");
  for (int i = 0; i < nfrm; ++i) {
    double d = f0[i];
    fwrite(&d, 8, 1, o);
  }
  fclose(o);
  return 0;
}
