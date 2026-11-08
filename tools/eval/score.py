"""Score engines against the reference text and the official engine.
usage: score.py MODEL_TAG RESULT_TAG...   (e.g. whistle_d4 rust_whistle_d4 c_whistle_d4)"""
import json, os, re, sys, unicodedata
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RES = os.path.join(ROOT, "results")

def norm(s):
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\s']", " ", s).replace("'", " ")
    return s.split()

def edits(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]

def wer(pairs):
    e = sum(edits(norm(h), norm(r)) for h, r in pairs)
    n = sum(len(norm(r)) for _, r in pairs)
    return 100.0 * e / max(n, 1)

def enc_stats(tag, off_tag, man):
    cos_mean, cos_min, good, n_frames_diff = [], [], [], 0
    for m in man:
        name = os.path.splitext(os.path.basename(m["clip"]))[0]
        a = np.fromfile(os.path.join(RES, "enc", tag, name + ".f32"), np.float32).reshape(-1, 512)
        b = np.fromfile(os.path.join(RES, "enc", off_tag, name + ".f32"), np.float32).reshape(-1, 512)
        if a.shape != b.shape:
            n_frames_diff += 1
            continue
        c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
        cos_mean.append(c.mean()); cos_min.append(c[:-2].min() if len(c) > 2 else c.min())
        good.append((c[:-2] > 0.999).mean() if len(c) > 2 else 1.0)
    return np.mean(cos_mean), np.min(cos_min), 100 * np.mean(good), n_frames_diff

def main(model, tags):
    man = json.load(open(os.path.join(ROOT, "clips", "eval", "manifest.json")))
    off_tag = f"official_{model}"
    res = {t: json.load(open(os.path.join(RES, t + ".json"))) for t in [off_tag] + tags}
    print(f"model {model}, {len(man)} clips ({sum(not m['noisy'] for m in man)} clean, {sum(m['noisy'] for m in man)} noisy)\n")
    hdr = f"{'engine':<26}{'WER all':>8}{'clean':>7}{'noisy':>7}{'lang ok':>8}{'=official':>10}{'enc cos':>9}{'worst':>8}{'>0.999':>8}"
    print(hdr)
    for t in [off_tag] + tags:
        r = res[t]
        sel = lambda f: [(r[m["clip"]]["text"], m["text"]) for m in man if f(m)]
        lang = 100 * np.mean([r[m["clip"]]["lang"] == m["lang"] for m in man])
        same = 100 * np.mean([norm(r[m["clip"]]["text"]) == norm(res[off_tag][m["clip"]]["text"]) for m in man])
        if t == off_tag:
            enc = "       -       -       -"
        else:
            cm, cw, g, nd = enc_stats(t, off_tag, man)
            enc = f"{cm:9.5f}{cw:8.4f}{g:7.1f}%" + (f" ({nd} len!)" if nd else "")
        print(f"{t:<26}{wer(sel(lambda m: True)):7.1f}%{wer(sel(lambda m: not m['noisy'])):6.1f}%{wer(sel(lambda m: m['noisy'])):6.1f}%{lang:7.1f}%{same:9.1f}%{enc}")
    print("\nWER by language (all clips):")
    langs = sorted({m["lang"] for m in man})
    print(f"{'engine':<26}" + "".join(f"{l:>7}" for l in langs))
    for t in [off_tag] + tags:
        r = res[t]
        print(f"{t:<26}" + "".join(f"{wer([(r[m['clip']]['text'], m['text']) for m in man if m['lang'] == l]):6.1f}%" for l in langs))

main(sys.argv[1], sys.argv[2:])
