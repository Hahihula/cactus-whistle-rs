"""Build the evaluation set: macOS `say` voices reading known sentences, clean
and with white noise at ~12 dB SNR. Writes clips/eval/*.wav and
clips/eval/manifest.json (clip, language, voice, reference text, noisy)."""
import json, os, subprocess, sys, wave
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(ROOT, "clips", "eval")
SNR_DB = 12.0

SENTENCES = {
    "en": ["Turn the light red for five seconds.",
           "The quick brown fox jumps over the lazy dog.",
           "Speech recognition runs entirely on a small microcontroller.",
           "Tell me everything you know about Renaissance painters.",
           "Please set an alarm for seven thirty tomorrow morning.",
           "The weather in Prague will be cloudy with light rain."],
    "de": ["Lass das Licht rot blinken.", "Das Wetter ist heute sehr schön.",
           "Ich möchte einen Kaffee mit Milch bestellen."],
    "fr": ["Bonjour, comment allez-vous aujourd'hui ?", "Allume la lumière dans la cuisine.",
           "Le train part à huit heures du matin."],
    "es": ["Pon la luz amarilla por favor.", "Mañana vamos a la playa con mis amigos.",
           "¿Dónde está la estación de tren?"],
    "it": ["Fai lampeggiare la luce verde.", "Domani andiamo al mare con la famiglia.",
           "Vorrei un caffè e un cornetto, per favore."],
    "nl": ["Laat het licht blauw knipperen.", "Het weer is vandaag erg mooi.",
           "Ik wil graag een kopje koffie."],
    "pl": ["Niech światło miga na czerwono.", "Dzisiaj jest bardzo ładna pogoda.",
           "Poproszę kawę z mlekiem."],
}
VOICES = {
    "en": ["Samantha (English (US))", "Daniel (English (UK))", "Karen", "Moira", "Tessa", "Rishi",
           "Reed (English (US))", "Shelley (English (UK))"],
    "de": ["Anna (German (Germany))", "Eddy (German (Germany))", "Flo (German (Germany))", "Sandy (German (Germany))"],
    "fr": ["Thomas (French (France))", "Jacques", "Amélie", "Shelley (French (France))"],
    "es": ["Mónica", "Paulina", "Eddy (Spanish (Spain))", "Flo (Spanish (Mexico))"],
    "it": ["Alice (Italian (Italy))", "Eddy (Italian (Italy))", "Flo (Italian (Italy))", "Sandy (Italian (Italy))"],
    "nl": ["Xander (Dutch (Netherlands))", "Ellen"],
    "pl": ["Zosia"],
}

def read(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)

def write(path, x):
    x = np.clip(np.round(x), -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(x.tobytes())

def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(1234)
    manifest = []
    for lang, sents in SENTENCES.items():
        for vi, voice in enumerate(VOICES[lang]):
            for si, text in enumerate(sents):
                name = f"{lang}_v{vi}_s{si}"
                clean = os.path.join(OUT, name + ".wav")
                if not os.path.exists(clean):
                    r = subprocess.run(["say", "-v", voice, "-o", clean, "--file-format=WAVE",
                                        "--data-format=LEI16@16000", text], capture_output=True)
                    if r.returncode != 0:
                        print("skip", name, voice, r.stderr.decode()[:80], file=sys.stderr)
                        continue
                x = read(clean)
                # white noise at SNR_DB relative to the speech power (frames over -40 dB of peak)
                p = np.mean(x[np.abs(x) > np.abs(x).max() * 0.01] ** 2)
                noise = rng.normal(0, np.sqrt(p / 10 ** (SNR_DB / 10)), x.shape)
                noisy = os.path.join(OUT, name + "_noisy.wav")
                write(noisy, x + noise)
                for path, n in ((clean, False), (noisy, True)):
                    manifest.append({"clip": os.path.relpath(path, ROOT), "lang": lang, "voice": voice,
                                     "text": text, "noisy": n})
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(len(manifest), "clips")

main()
