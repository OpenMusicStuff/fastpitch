"""Frame-by-frame comparison of fastpitch against librosa.pyin.

usage: python tests/compare_librosa.py <fastpitch_cli> [--quick]
"""
import os
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
import librosa

warnings.filterwarnings("ignore")
CLI = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "build/fastpitch_cli.exe")
QUICK = "--quick" in sys.argv
rng = np.random.default_rng(1234)


def harmonic(f0, sr, amps=(1, 0.6, 0.4, 0.25, 0.15)):
    ph = 2 * np.pi * np.cumsum(f0) / sr
    return sum(a * np.sin((k + 1) * ph) for k, a in enumerate(amps))


def signals(sr):
    t3 = np.arange(int(3 * sr)) / sr
    out = {}
    out["sine220"] = 0.5 * np.sin(2 * np.pi * 220 * t3)
    out["chirp"] = 0.5 * np.sin(2 * np.pi * np.cumsum(np.geomspace(80, 1500, t3.size)) / sr)
    vib = 196 * 2 ** (0.5 / 12 * np.sin(2 * np.pi * 5.5 * t3))
    env = np.clip(np.sin(np.pi * t3 / 1.0) ** 2, 0, 1) * (np.sin(np.pi * t3) > 0)
    out["vibrato_env"] = 0.3 * harmonic(vib, sr) * env
    notes = np.repeat(rng.choice(librosa.midi_to_hz(np.arange(45, 80)), 12), int(0.25 * sr))
    seq = harmonic(notes, sr) * 0.2
    seq[int(1.2 * sr):int(1.6 * sr)] = 0
    out["notes_noisy"] = seq + 0.03 * rng.standard_normal(seq.size)
    out["noise"] = 0.1 * rng.standard_normal(int(2 * sr))
    out["silence_tone"] = np.concatenate([np.zeros(sr), 0.4 * np.sin(2 * np.pi * 440 * t3[:sr]), np.zeros(sr // 2)])
    lo = 70 * 2 ** (np.cumsum(rng.standard_normal(t3.size)) / sr * 3)
    out["low_random_walk"] = 0.3 * harmonic(np.clip(lo, 66, 1900), sr)
    for name in ["trumpet", "libri1"]:
        try:
            y, _ = librosa.load(librosa.ex(name), sr=sr, duration=None if not QUICK else 6.0)
            out[name] = y.astype(np.float64)
        except Exception as e:  # offline
            print(f"  (skipping {name}: {e})")
    return out


def run_fastpitch(y, sr, **kw):
    with tempfile.TemporaryDirectory() as d:
        inp, outp = os.path.join(d, "x.f32"), os.path.join(d, "o.bin")
        y.astype(np.float32).tofile(inp)
        args = [CLI, inp, outp, f"sr={sr}"] + [f"{k}={v}" for k, v in kw.items()]
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr)
        raw = open(outp, "rb").read()
    n = int(np.frombuffer(raw[:8], np.int64)[0])
    f0 = np.frombuffer(raw[8:8 + 8 * n], np.float64)
    vp = np.frombuffer(raw[8 + 8 * n:8 + 16 * n], np.float64)
    vf = np.frombuffer(raw[8 + 16 * n:], np.uint8).astype(bool)
    return f0, vf, vp, r.stderr.strip()


def frames_differing(f0a, va, f0b, vb):
    both = va & vb
    return int(np.sum(va != vb) + np.sum(np.abs(1200 * np.log2(f0a[both] / f0b[both])) >= 1e-3))


def compare(name, y, sr, thr, double):
    """float64 rows: fastpitch<double> must equal librosa (float64) on every frame.
    float32 rows: both float32 pipelines are judged against the float64 truth; fastpitch may
    only deviate where a decision boundary lies within float32 resolution (counted)."""
    kw = dict(fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr,
              transition_min_prob=thr if thr > 0 else None)
    y64 = y.astype(np.float32).astype(np.float64)
    t0 = time.perf_counter()
    f0r, vfr, vpr = librosa.pyin(y64, **kw)
    tl = time.perf_counter() - t0
    f0f, vff, vpf, info = run_fastpitch(y, sr, thr=thr, double=int(double))
    assert f0f.shape == f0r.shape, (f0f.shape, f0r.shape)
    dfp = frames_differing(f0r, vfr, f0f, vff)
    vp_err = np.max(np.abs(vpr - vpf))
    if double:
        ok = dfp == 0
        detail = f"differing frames={dfp}  max|dvp|={vp_err:.1e}"
    else:
        f032, v32, _ = librosa.pyin(y.astype(np.float32), **kw)
        dl = frames_differing(f0r, vfr, f032, v32)
        ok = dfp <= dl + max(1, int(0.002 * len(f0r)))
        detail = f"vs float64 truth: librosa-f32 {dl}, fastpitch-f32 {dfp} differing frames"
        if ok and dfp > dl:
            detail += " (float32 near-tie)"
    print(f"{name:16s} thr={thr:<6g} {'f64' if double else 'f32'} frames={len(f0r):5d} "
          f"{detail}  librosa={tl*1e3:8.1f}ms  {'OK' if ok else 'DIFF'}  [{info}]")
    return ok


def main():
    sr = 22050
    sig = signals(sr)
    allok = True
    for thr in ([1e-4] if QUICK else [1e-4, 0.0]):
        for double in ([True] if QUICK else [True, False]):
            for name, y in sig.items():
                allok &= compare(name, y, sr, thr, double)
    print("ALL OK" if allok else "FAILURES")


if __name__ == "__main__" and "--sweep" not in sys.argv:
    main()


# ---------------------------------------------------------------------------------------
# Parameter sweep: non-default configurations exercise even-width kernels, edge extras,
# narrow/wide bands, the fallback paths and different FFT sizes.
# ---------------------------------------------------------------------------------------
CONFIGS = [
    dict(sr=44100, frame=2048, hop=256),
    dict(sr=16000, frame=1024, hop=160, fmin=60, fmax=800),
    dict(sr=22050, frame=2048, hop=512, res=0.2),
    dict(sr=22050, frame=2048, hop=512, res=1 / 3),           # even transition width
    dict(sr=22050, frame=2048, hop=300, rate=20.0),
    dict(sr=22050, frame=1536, hop=384, fmin=100, fmax=1000, switch=0.2),
    dict(sr=22050, frame=2048, hop=512, nthr=37, ba=1.5, bb=9, boltz=1.2, notrough=0.05),
    dict(sr=48000, frame=4096, hop=480, fmin=40, fmax=2000),
    dict(sr=22050, frame=2048, hop=128, res=0.05),
]
LIBROSA_KEYS = dict(frame="frame_length", hop="hop_length", fmin="fmin", fmax="fmax",
                    res="resolution", rate="max_transition_rate", switch="switch_prob",
                    nthr="n_thresholds", boltz="boltzmann_parameter", notrough="no_trough_prob")


def sweep():
    allok = True
    for cfg in CONFIGS:
        sr = cfg["sr"]
        sig = signals(sr)
        for thr in [1e-4, 0.0]:
            for name in ["vibrato_env", "notes_noisy", "trumpet", "libri1", "sine220"]:
                if name not in sig:
                    continue
                y = sig[name].astype(np.float32).astype(np.float64)
                kw = dict(sr=sr, fmin=cfg.get("fmin", librosa.note_to_hz("C2")),
                          fmax=cfg.get("fmax", librosa.note_to_hz("C7")))
                for k, v in cfg.items():
                    if k in LIBROSA_KEYS and k not in ("fmin", "fmax"):
                        kw[LIBROSA_KEYS[k]] = v
                if "ba" in cfg:
                    kw["beta_parameters"] = (cfg["ba"], cfg["bb"])
                try:
                    f0r, vfr, vpr = librosa.pyin(y, transition_min_prob=thr if thr > 0 else None, **kw)
                except Exception as e:
                    print(f"  librosa rejected {cfg}: {e}")
                    break
                cli = {k: v for k, v in cfg.items() if k != "sr"}
                cli.update(fmin=kw["fmin"], fmax=kw["fmax"], thr=thr, double=1)
                f0f, vff, vpf, info = run_fastpitch(sig[name], sr, **cli)
                both = vfr & vff
                agree = np.mean(vfr == vff)
                same = np.mean(np.abs(1200 * np.log2(f0r[both] / f0f[both])) < 1e-3) if both.any() else 1.0
                ok = agree == 1.0 and same == 1.0 and len(f0r) == len(f0f)
                allok &= ok
                tag = ",".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in cfg.items())
                print(f"{tag:58s} thr={thr:<6g} {name:12s} n={len(f0r):5d} voicing={agree*100:7.3f}% "
                      f"same-bin={same*100:7.3f}% dvp={np.max(np.abs(vpr-vpf)):.0e} {'OK' if ok else 'DIFF'}")
    print("SWEEP ALL IDENTICAL" if allok else "SWEEP HAS DIFFERENCES")


if __name__ == "__main__" and "--sweep" in sys.argv:
    sweep()
