#!/bin/sh
# Run the C engine and the Rust engine (with the C engine's behaviour,
# --spec c) over every clip in clips/; report differences.
# Needs tools/fetch_c_engine.sh and tools/make_clips.sh first.
# usage: tools/compare.sh [model.cact] [extra args for both engines]
cd "$(dirname "$0")/.."
model=${1:-model/whistle_d4.cact}; shift 2>/dev/null
c_engine=c-engine/build/wh_run
[ -x "$c_engine" ] || { echo "no $c_engine: run tools/fetch_c_engine.sh"; exit 2; }
cargo build --release --quiet || exit 2
ok=0; bad=0
for f in clips/*.wav; do
  c=$("$c_engine" "$model" "$f" --lang auto "$@" 2>/dev/null)
  r=$(./target/release/whistle transcribe "$model" "$f" --lang auto --spec c "$@" 2>/dev/null)
  if [ "$c" = "$r" ]; then ok=$((ok+1)); printf 'same  %-16s %s\n' "$(basename "$f")" "$r"
  else bad=$((bad+1)); printf 'DIFF  %-16s\n   C:    %s\n   Rust: %s\n' "$(basename "$f")" "$c" "$r"; fi
done
echo "$ok same, $bad different"
[ $bad -eq 0 ]
