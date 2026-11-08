#!/bin/sh
# Synthesise 16 kHz mono test clips with macOS `say` into clips/.
set -e
cd "$(dirname "$0")/.."
mkdir -p clips
mk() { say -v "$1" -o "clips/$2.wav" --file-format=WAVE --data-format=LEI16@16000 "$3" 2>/dev/null || echo "skip $2 (no voice $1)"; }
mk Samantha en1 "Turn the light red for five seconds."
mk Samantha en2 "The quick brown fox jumps over the lazy dog."
mk Samantha en3 "Speech recognition running entirely on a small microcontroller, written in Rust, without any operating system or network connection."
mk Thomas   fr1 "Bonjour, comment allez vous aujourd'hui?"
mk Anna     de1 "Lass das Licht rot blinken."
mk Monica   es1 "Pon la luz amarilla por favor."
mk Alice    it1 "Fai lampeggiare la luce verde."
mk Xander   nl1 "Laat het licht blauw knipperen."
mk Zosia    pl1 "Niech światło miga na czerwono."
