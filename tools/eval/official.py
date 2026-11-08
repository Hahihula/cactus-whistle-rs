"""Run Cactus's official engine (pip cactus-compute) over the evaluation set:
transcript and language (auto-detected) and the encoder output from embed().
Writes results/official_<model>.json and results/enc/official_<model>/<clip>.f32.
Run with CACTUS_TELEMETRY=0."""
import array, json, os, sys
from cactus import Whistle

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RES = os.path.join(ROOT, "results")

def main(model):
    tag = os.path.splitext(os.path.basename(model))[0]
    encdir = os.path.join(RES, "enc", f"official_{tag}")
    os.makedirs(encdir, exist_ok=True)
    man = json.load(open(os.path.join(ROOT, "clips", "eval", "manifest.json")))
    w = Whistle(weights=os.path.join(ROOT, model))
    out = {}
    for m in man:
        path = os.path.join(ROOT, m["clip"])
        r = w.transcribe(path)
        e = w.embed(path)
        name = os.path.splitext(os.path.basename(m["clip"]))[0]
        with open(os.path.join(encdir, name + ".f32"), "wb") as f:
            array.array("f", e).tofile(f)
        out[m["clip"]] = {"text": r["text"], "lang": r["language"]}
    json.dump(out, open(os.path.join(RES, f"official_{tag}.json"), "w"), ensure_ascii=False, indent=1)
    print(tag, len(out), "clips")

assert os.environ.get("CACTUS_TELEMETRY") == "0", "set CACTUS_TELEMETRY=0"
main(sys.argv[1])
