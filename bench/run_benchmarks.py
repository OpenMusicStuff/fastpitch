"""Speed and accuracy comparison of pYIN implementations.

    python bench/run_benchmarks.py --fastpitch build/fastpitch_cli.exe \
        [--c4dm build/compete_c4dm.exe] [--sw build/compete_sw.exe] [--threads 8]

Speed  : wall time (best of N runs, single process) on ~5 minutes of speech
         (librosa example 'libri1' tiled) for two framings.
Quality: raw pitch accuracy / voicing metrics against the known f0 of synthetic signals.
"""
import argparse
import os
import platform
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
import librosa

warnings.filterwarnings("ignore")
C2, C7 = librosa.note_to_hz("C2"), librosa.note_to_hz("C7")


def write_f32(y, path):
    np.asarray(y, np.float32).tofile(path)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(" ".join(cmd) + "\n" + r.stderr)
    return r.stderr.strip()


def ms_of(info):
    return float(info.split("time=")[1].split()[0])


def fastpitch(exe, y, sr, out, repeat=5, **kw):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.f32")
        write_f32(y, p)
        info = run([exe, p, out, f"sr={sr}", f"repeat={repeat}"] + [f"{k}={v}" for k, v in kw.items()])
    raw = open(out, "rb").read()
    n = int(np.frombuffer(raw[:8], np.int64)[0])
    f0 = np.frombuffer(raw[8:8 + 8 * n], np.float64)
    vf = np.frombuffer(raw[8 + 16 * n:], np.uint8).astype(bool)
    return ms_of(info), np.where(vf, f0, 0.0)


def external(exe, y, sr, out, repeat=3, **kw):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.f32")
        write_f32(y, p)
        info = run([exe, p, out, f"sr={sr}", f"repeat={repeat}"] + [f"{k}={v}" for k, v in kw.items()])
    f0 = np.fromfile(out, np.float64)
    return ms_of(info), np.where(f0 > 0, f0, 0.0)


def guarded(f):
    try:
        return f()
    except RuntimeError:
        return None


def librosa_pyin(y, sr, frame, hop, repeat=1):
    best, f0 = 1e30, None
    for _ in range(repeat):
        t0 = time.perf_counter()
        f, v, _ = librosa.pyin(y.astype(np.float32), fmin=C2, fmax=C7, sr=sr, frame_length=frame,
                               hop_length=hop)
        best = min(best, time.perf_counter() - t0)
        f0 = np.where(v, f, 0.0)
    return best * 1e3, f0


# ---------------------------------------------------------------------------------------
# Synthetic signals with known f0 (harmonic source, vibrato, glides, silence, noise)
# ---------------------------------------------------------------------------------------
def synth(sr, seconds, seed, snr_db):
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f0 = np.zeros(n)
    i = 0
    while i < n:
        seg = int(rng.uniform(0.3, 1.2) * sr)
        if rng.random() < 0.25:
            f0[i:i + seg] = 0.0  # unvoiced gap
        else:
            a, b = 2 ** rng.uniform(np.log2(80), np.log2(1000), 2)
            ramp = np.geomspace(a, b if rng.random() < 0.4 else a, min(seg, n - i))
            vib = 2 ** (rng.uniform(0, 0.6) / 12 * np.sin(2 * np.pi * rng.uniform(4, 7) * t[i:i + seg][:len(ramp)]))
            f0[i:i + len(ramp)] = ramp * vib
        i += seg
    phase = 2 * np.pi * np.cumsum(f0) / sr
    amps = 1.0 / (1 + np.arange(8)) ** rng.uniform(0.7, 1.5)
    y = sum(a * np.sin((k + 1) * phase) for k, a in enumerate(amps)) * (f0 > 0)
    y = 0.3 * y / np.max(np.abs(y))
    noise = rng.standard_normal(n)
    y = y + noise * np.sqrt(np.mean(y[f0 > 0] ** 2) / 10 ** (snr_db / 10))
    return y.astype(np.float32), f0


def metrics(est, est_times, f0, sr):
    # truth at the estimate times (unvoiced if any sample within +-5 ms is unvoiced)
    idx = np.clip((est_times * sr).astype(int), 0, len(f0) - 1)
    w = int(0.005 * sr)
    ref = f0[idx].copy()
    for k, j in enumerate(idx):
        if np.any(f0[max(0, j - w):j + w + 1] == 0):
            ref[k] = 0.0
    rv, ev = ref > 0, est > 0
    both = rv & ev
    cents = np.full(len(ref), np.inf)
    cents[both] = np.abs(1200 * np.log2(est[both] / ref[both]))
    rpa = np.sum(cents[rv] <= 50) / max(1, rv.sum())
    vr = np.sum(rv & ev) / max(1, rv.sum())
    vfa = np.sum(~rv & ev) / max(1, (~rv).sum())
    med = np.median(cents[both & (cents <= 50)]) if np.any(both & (cents <= 50)) else np.nan
    return rpa, vr, vfa, med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastpitch", required=True)
    ap.add_argument("--c4dm")
    ap.add_argument("--sw")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--skip-librosa", action="store_true")
    a = ap.parse_args()
    fp = os.path.abspath(a.fastpitch)
    c4 = os.path.abspath(a.c4dm) if a.c4dm else None
    sw = os.path.abspath(a.sw) if a.sw else None
    tmp = tempfile.mkdtemp()
    out = os.path.join(tmp, "o.bin")

    print(f"# pYIN benchmark ({platform.processor() or platform.machine()}, librosa {librosa.__version__})\n")
    base, _ = librosa.load(librosa.ex("libri1"), sr=None)
    for sr, frame, hop in [(22050, 2048, 512), (44100, 2048, 256)]:
        y = librosa.resample(base, orig_sr=librosa.get_samplerate(librosa.ex("libri1")), target_sr=sr)
        y = np.tile(y, int(np.ceil(a.minutes * 60 * sr / len(y))))[: int(a.minutes * 60 * sr)]
        dur = len(y) / sr
        rows = []
        t1, _ = fastpitch(fp, y, sr, out, frame=frame, hop=hop, threads=1)
        rows.append(("fastpitch (1 thread)", "librosa 601 bins, +-50 bins/frame" if hop == 512 else "librosa 601 bins, +-15 bins/frame", t1))
        tn, _ = fastpitch(fp, y, sr, out, frame=frame, hop=hop, threads=a.threads)
        rows.append((f"fastpitch ({a.threads} threads)", "same", tn))
        if c4:
            tc = guarded(lambda: external(c4, y, sr, out, frame=frame, hop=hop)[0])
            rows.append(("c4dm/pyin (original C++)", "345 bins, +-5 bins/frame", tc))
            tm, _ = fastpitch(fp, y, sr, out, frame=frame, hop=hop, threads=1, fmin=61.735,
                              fmax=min(3322.4, sr / 2), res=0.2, rate=round(2 * sr / (12 * hop), 3) + 0.0)
            rows.append(("fastpitch, c4dm-sized HMM (1 thread)", "346 bins, +-5 bins/frame", tm))
        if sw:
            ts = guarded(lambda: external(sw, y, sr, out, frame=frame, hop=hop, fmin=C2, fmax=C7)[0])
            rows.append(("Sleepwalking/libpyin (C)", "own variant", ts))
        if not a.skip_librosa:
            tl, _ = librosa_pyin(y, sr, frame, hop)
            rows.append((f"librosa {librosa.__version__} pyin", "reference", tl))
        print(f"## {dur:.0f} s of speech, sr={sr}, frame={frame}, hop={hop}\n")
        print("| implementation | HMM | time [ms] | x realtime | fastpitch (1 thread) is faster by |")
        print("|---|---|---:|---:|---:|")
        for name, model, t in rows:
            if t is None:
                print(f"| {name} | {model} | crashed | - | - |")
            elif name.startswith("fastpitch"):
                print(f"| **{name}** | {model} | **{t:.1f}** | **{dur / t * 1e3:,.0f}** | - |")
            else:
                print(f"| {name} | {model} | {t:.1f} | {dur / t * 1e3:,.0f} | {t / t1:.1f}x |")
        print()

    print("## Accuracy on synthetic signals with known f0 (harmonic, vibrato/glides, gaps)\n")
    print("| implementation | SNR | RPA (50c) | voicing recall | voicing false alarm | median error [cents] |")
    print("|---|---:|---:|---:|---:|---:|")
    sr, frame, hop = 22050, 2048, 256
    for snr in [30, 10]:
        res = {}
        for seed in range(6):
            y, f0 = synth(sr, 20, seed, snr)
            nf = 1 + len(y) // hop
            ests = {}
            _, e = fastpitch(fp, y, sr, out, frame=frame, hop=hop, repeat=1)
            ests["fastpitch"] = (e, np.arange(len(e)) * hop / sr)
            _, e = fastpitch(fp, y, sr, out, frame=frame, hop=hop, repeat=1, refine=1)
            ests["fastpitch (refine_f0)"] = (e, np.arange(len(e)) * hop / sr)
            if not a.skip_librosa:
                _, e = librosa_pyin(y, sr, frame, hop)
                ests["librosa"] = (e, np.arange(len(e)) * hop / sr)
            if c4:
                e = guarded(lambda: external(c4, y, sr, out, frame=frame, hop=hop, repeat=1)[1])
                if e is not None:
                    ests["c4dm/pyin"] = (e, (np.arange(len(e)) * hop + frame / 4) / sr)
            if sw:
                e = guarded(lambda: external(sw, y, sr, out, frame=frame, hop=hop, fmin=C2, fmax=C7, repeat=1)[1])
                if e is not None:
                    ests["Sleepwalking/libpyin"] = (e, np.arange(len(e)) * hop / sr)
            for k, (e, tt) in ests.items():
                res.setdefault(k, []).append(metrics(e, tt, f0, sr))
        for k, v in res.items():
            m = np.nanmean(np.array(v), axis=0)
            print(f"| {k} | {snr} dB | {m[0]*100:.2f}% | {m[1]*100:.2f}% | {m[2]*100:.2f}% | {m[3]:.2f} |")
    print()


if __name__ == "__main__":
    main()
