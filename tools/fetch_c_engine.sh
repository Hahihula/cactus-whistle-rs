#!/bin/sh
# Optional: fetch and build the C engine this port started from,
# andrisgauracs/whistle-esp32 (Apache-2.0), for tools/compare.sh and
# tools/eval/run_local.py c. Nothing else needs it.
# Pinned to the commit the Rust engine was compared against.
set -e
cd "$(dirname "$0")/.."
REPO=https://github.com/andrisgauracs/whistle-esp32
COMMIT=383e0088d7140a89a9111aa985671190a9048aa3
if [ ! -d c-engine/.git ]; then
  git clone --quiet "$REPO" c-engine
fi
git -C c-engine fetch --quiet origin
git -C c-engine checkout --quiet "$COMMIT"
cmake -S c-engine/host -B c-engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build c-engine/build -j
echo "built c-engine/build/wh_run"
