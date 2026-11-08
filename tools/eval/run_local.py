"""Run a local engine over the evaluation set.
usage: run_local.py rust|c MODEL TAG [extra engine args...]
(c needs tools/fetch_c_engine.sh first)
Writes results/<TAG>.json and results/enc/<TAG>/<clip>.f32."""
import json, os, subprocess, sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
BIN = {"rust": ["target/release/whistle", "transcribe"], "c": ["c-engine/build/wh_run"]}

def main(engine, model, tag, extra):
    encdir = os.path.join(ROOT, "results", "enc", tag)
    os.makedirs(encdir, exist_ok=True)
    man = json.load(open(os.path.join(ROOT, "clips", "eval", "manifest.json")))
    out = {}
    for m in man:
        name = os.path.splitext(os.path.basename(m["clip"]))[0]
        enc = os.path.join(encdir, name + ".f32")
        exe, *sub = BIN[engine]
        r = subprocess.run([os.path.join(ROOT, exe), *sub, os.path.join(ROOT, model), os.path.join(ROOT, m["clip"]),
                            "--lang", "auto", "--enc-out", enc, *extra], capture_output=True, text=True)
        lang = next((l.split()[1] for l in r.stderr.splitlines() if l.startswith("language ")), "?")
        out[m["clip"]] = {"text": r.stdout.strip(), "lang": lang}
    json.dump(out, open(os.path.join(ROOT, "results", tag + ".json"), "w"), ensure_ascii=False, indent=1)
    print(tag, len(out), "clips")

main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
