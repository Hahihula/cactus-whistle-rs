# cactus-whistle-rs

Speech recognition with [Cactus Whistle](https://huggingface.co/Cactus-Compute/whistle)
in Rust: a `no_std` engine that runs on microcontrollers with no network and no
operating system, and the same engine on a desktop.

Whistle is Cactus Compute's multilingual speech model (English, German, French,
Spanish, Italian, Dutch, Polish) with language detection. The model has 2- and 4-bit
codebook weights and is 14.5 MB with 4 of its 8 decoder layers, small enough for a
16 MB flash chip.

- **`cactus-whistle`** (`whistle/`): the engine. `no_std` + `alloc`, no other
  dependency than `libm`. It reads the `.cact` model in place (on a
  microcontroller, from memory-mapped flash), runs every matmul in int8, and takes
  a platform's SIMD dot-product kernel through a hook.
- **`whistle`** (`cli/`): a desktop binary to transcribe WAV files, trim models to fewer
  decoder layers and inspect them.
- **`tools/`**: test clips, an evaluation against Cactus's official engine, and an
  optional comparison with the C engine this port started from.

## Quick start

Download `whistle` for Linux, macOS or Windows from the
[releases page](https://github.com/Hahihula/cactus-whistle-rs/releases), or build it
(`cargo build --release`; on Linux, microphone support needs ALSA's headers:
`sudo apt install libasound2-dev`). Then get the model and trim it:

```bash
mkdir -p model
curl -L -o model/whistle.cact https://huggingface.co/Cactus-Compute/whistle/resolve/main/whistle.cact
whistle trim model/whistle.cact 4 model/whistle_d4.cact   # 14.5 MB; the full model works too

whistle transcribe model/whistle_d4.cact recording.mp3     # a file
whistle listen model/whistle_d4.cact                       # the microphone
```

`transcribe` takes WAV, MP3, OGG/Vorbis, FLAC and AAC/M4A at any sample rate, mono
or stereo; it is mixed down and resampled to the 16 kHz mono the model expects.
`listen` is push to talk: Enter starts recording, Enter stops and prints the
transcript, `q` quits. (macOS asks for microphone access the first time.) On macOS,
`tools/make_clips.sh` writes nine test clips in seven languages into `clips/`.

The transcript goes to stdout. stderr has the detected language, timings and an
encoder checksum (`enc_fnv`): for the same 16 kHz input, the same value on every
machine, microcontrollers included. (Decoded files reproduce across platforms
too, except MP3, whose decoder's last bits depend on the platform.)

## The `whistle` binary

```text
whistle transcribe MODEL.cact AUDIO [options]   speech to text from a file
whistle listen MODEL.cact [options]             push to talk from the microphone
whistle trim MODEL.cact DEPTH OUT.cact          keep DEPTH decoder layers
whistle info MODEL.cact                         header, tensors, trimmed sizes
```

The CLI's extra dependencies (`symphonia` to decode files, `cpal` for the
microphone) are only in the `cactus-whistle-cli` package; the engine stays
dependency-free.

`transcribe` and `listen` options: `--lang xx` forces a language (default: detect), `--spec
guide|fast|c` picks the behaviour (see [Specs](#specs)), and `--depth N` runs N
decoder layers of an untrimmed model. Debugging options write the log-mel input, the
encoder output or teacher-forced logits to files; run `whistle transcribe` without
arguments for the list.

`trim` writes a model with only the decoder layers a depth uses. The decoder is
trained as nested sub-models (a "ladder"), so depth 4 is a real model of its own and
not a truncation:

```text
$ whistle info model/whistle.cact
...
decoder depth (`whistle trim`): layers kept, model size
  2: [0, 7]                      10.66 MB
  3: [0, 3, 7]                   13.91 MB
  4: [0, 3, 5, 7]                14.52 MB
  ...
  8: [0, 1, 2, 3, 4, 5, 6, 7]    16.92 MB
```

The output is byte-identical to `whistle_strip.py` from whistle-esp32.

## Using the library

```toml
[dependencies]
cactus-whistle = "0.1"
```

```rust,ignore
let blob: &[u8] = /* the .cact file: a memory-mapped flash partition, or a Vec */;
let mut w = cactus_whistle::Whistle::open(blob)?;
let text = w.transcribe(&samples_16khz_f32, None)?; // None: detect the language
let lang = cactus_whistle::LANGS[w.lang];
```

It needs a global allocator for activations (on a microcontroller, external RAM
such as PSRAM works) and nothing else from the platform.

| feature | what |
|---|---|
| `std` | for desktops: fused multiply-add through `f32::mul_add`, which LLVM vectorises around (~17% faster encode on a Mac); same results |
| `riscv-fpu` | `fmadd.s` for fused multiply-add on RISC-V with the F extension; stable Rust doesn't expose it as a cfg, and without it `fma` falls back to software |
| `fast` | make `Spec::GuideFast` the default (greedy decoding) |
| `c-engine` | make `Spec::CEngine` the default |

**SIMD.** Almost all the work is 128-wide int8 dot products. The engine computes
them in portable Rust (which LLVM vectorises on a desktop); a platform with a vector
unit installs its own kernel once, before opening the model:

```rust,ignore
use cactus_whistle::quant::{set_kernel, Kernel};

/// One weight row `w` against `rows` activation rows `x` (both 16-byte
/// aligned): out[r * groups + g] = dot of 128-lane group g of w and of row r.
unsafe fn my_dot_rows(w: *const i8, x: *const i8, groups: usize, rows: usize, out: &mut [i32]) {
    /* SIMD */
}

set_kernel(Some(Kernel { dot_rows: my_dot_rows, here: || true }));
```

The kernel must return exactly what `quant::dot128` returns (it's integer math, so
any correct SIMD version does), and `here` says whether it can run on the calling
core, for chips where only some cores have the vector unit.

**Two cores.** `cactus_whistle::par` splits matmul rows, convolutions and attention
heads between the calling core and a helper. The platform runs `par::help()` in a
loop on the other core and says what the helper may take
(`par::set_helper_int8(Int8Help::Full | Light | No)`): `Light` suits a helper core
without the vector unit, which then takes float work and the small decoder matmuls.
Results don't depend on the split: every row is computed the same way on either core.
On a desktop, `WH_HELPER=full whistle transcribe ...` runs the same scheme with a
thread.

**Profiling.** `cactus_whistle::prof` counts time per stage once the platform gives it
a microsecond clock (`prof::set_clock`).

## Specs

Cactus has not published its engine's source. This port started from
[andrisgauracs/whistle-esp32](https://github.com/andrisgauracs/whistle-esp32), a C
engine reverse-engineered from an older engine build. Cactus's
[porting guide](https://cactuscompute.com/blog/porting-whistle) describes the
official engine differently in a few places, so the engine has both, as `Spec`s:

| | `Spec::CEngine` | `Spec::Guide` (default) |
|---|---|---|
| level | scale to peak 1 | 99th-percentile RMS of 320-sample blocks to 0.1 |
| window | periodic Hann | symmetric Hann |
| log floor / std epsilon | 6e-7 / 1e-3 | 2^-24 / 1e-5 |
| positional signal | on the encoder input, interleaved | on the cross-attention memory, sin half / cos half |
| decoding | greedy | 5-beam search, length-normalised, beams batched |

`Spec::GuideFast` is the guide with greedy decoding. `Whistle::set_spec` switches at
run time, or set `front_blog`, `pe_memory` and `beams` one by one.

## Accuracy

Reference: Cactus's own engine (`pip install cactus-compute` 4.0.1, with
`CACTUS_TELEMETRY=0`) on the same model files. Evaluation set: 105 clips of macOS
voices reading known sentences in all 7 languages, plus the same clips with white
noise at 12 dB SNR. WER is against the known text.

| depth-4 model | WER | clean | noisy | same text as official | encoder frames with cosine > 0.999 |
|---|---|---|---|---|---|
| official engine | 31.3% | 22.5% | 40.1% | 100% | - |
| `Spec::CEngine` | 34.6% | 25.3% | 44.0% | 51% | 95.6% |
| `Spec::GuideFast` | 34.8% | 24.7% | 44.8% | 51% | 99.8% |
| `Spec::Guide` | **31.6%** | 23.0% | 40.2% | **82%** | **99.8%** |

The full 8-layer model shows the same pattern (official 28.7%, C behaviour 31.8%,
guide 29.8%). With the guide's front end the encoder output matches the official
`embed()` on 99.8% of frames, the guide's acceptance bar. The beam search is where
most of the text gap was. The remaining 18% of transcripts that differ are probably
quantisation detail: the official engine also keeps its attention cache in int8.

Reproduce (needs Python with numpy, and `cactus-compute` for the official engine):

```bash
python3 tools/eval/make_set.py                                    # clips/eval + manifest
CACTUS_TELEMETRY=0 python3 tools/eval/official.py model/whistle_d4.cact
python3 tools/eval/run_local.py rust model/whistle_d4.cact guide_d4
python3 tools/eval/run_local.py rust model/whistle_d4.cact c_d4 --spec c
python3 tools/eval/score.py whistle_d4 guide_d4 c_d4
```

## Speed

On two dual-core microcontrollers with a 128-bit SIMD unit and a SIMD kernel
installed (2.5 s clip, depth-4 model, both cores):

| | 240 MHz Xtensa LX7, encode / decode | 320 MHz RISC-V (SIMD on one core), encode / decode |
|---|---|---|
| `Spec::GuideFast` | 5.8 / 4.8 s | 5.9 / 7.7 s |
| `Spec::Guide` | 5.8 / 10.7 s | 5.9 / 14.2 s |

On a desktop it is about as fast as the C engine (within 10% on an M-series Mac).

What mattered on the microcontrollers:

- **Every matmul is int8.** All 112 codebook matrices are 2- or 4-bit with 128-wide
  groups. Activations are quantised to int8 per group, and the hot loop is one
  128-long int8 dot product per group, done on the SIMD unit.
- **Unpacking, not memory, is the cost.** Weight rows read from flash, external RAM
  and internal SRAM within 10-20% of each other. What costs is expanding 2/4-bit
  codes to int8: lookup tables of whole aligned words and a 4x unrolled loop took a
  2-bit row from 3731 to 1598 cycles (a SIMD dot is about 40).
- **Batched beams.** The decoder takes all live beams as rows of one matmul, so each
  weight row is unpacked once per step instead of five times. That took beam
  search from 4.5x to 2.3x the greedy time. It is bit-identical to decoding beams one
  by one.
- **Fixed-size arrays in hot kernels.** `dot128` taking slices instead of
  `&[i8; 128]` left a bounds check per element and made the desktop build 2.5x slower.

## Numerics

The float kernels use fused multiply-add (`math::fma`: `f32::mul_add` with the
`std` feature; otherwise Xtensa `madd.s`, RISC-V `fmadd.s`, AArch64 `fmadd`, or
`libm::fmaf`), and transcendental functions
come from the pure-Rust `libm` crate on every target. So a desktop run reproduces a
microcontroller run bit for bit: hashing the encoder output gives the same `enc_fnv`
on both microcontrollers above and in `whistle transcribe` on a Mac (`db9a1371` for
the guide's front end, `78453efd` for the C engine's).

The cost of fused multiply-add is bit-exact parity with the C engine. Compared
transcript by transcript, `Spec::CEngine` matches it on all 9 test clips with the
depth-4 model (`tools/compare.sh`); where the two differ elsewhere, they split at
near-ties. The C engine built with and without FMA differs from itself by the same
amount.

## Development

```bash
cargo test --release            # golden transcripts (needs model/ and clips/)
tools/make_clips.sh             # macOS: test clips into clips/
tools/fetch_c_engine.sh         # optional: clone and build the C engine into c-engine/
tools/compare.sh                # C engine vs Spec::CEngine on every clip
```

The library builds for any target with `alloc`, `riscv32imafc` and Xtensa
included.

GitHub Actions (`.github/workflows/ci.yml`) runs formatting, clippy, the tests, a
`no_std` build of the engine for two bare-metal targets, and an end-to-end check:
a spoken clip (espeak-ng) transcribed from WAV, MP3, OGG and FLAC must give the same
text. Pushing a `v*` tag runs `release.yml`, which builds `whistle` for Linux
(x86_64, ARM64), macOS (Apple Silicon, Intel) and Windows and attaches the archives
to the GitHub release.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.

Parts of the engine started as a port of
[andrisgauracs/whistle-esp32](https://github.com/andrisgauracs/whistle-esp32)
(Apache-2.0); the files concerned say so at the top. The Whistle model is not
part of this repository; it is published by Cactus Compute under its own license
(Apache-2.0 at the time of writing, see the model page).

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

## Acknowledgements

- [Cactus Compute](https://cactuscompute.com) made Whistle and published the model
  ([Hugging Face](https://huggingface.co/Cactus-Compute/whistle)) and the
  [porting guide](https://cactuscompute.com/blog/porting-whistle) that the default
  spec follows.
- [andrisgauracs/whistle-esp32](https://github.com/andrisgauracs/whistle-esp32) was
  the inspiration and the starting point for this engine: a C99
  Whistle engine for microcontrollers, with the `.cact` format and the model's
  architecture worked out tensor by tensor. This engine began as a port of it and was
  checked against it all along (`tools/fetch_c_engine.sh` fetches it for comparison;
  it is not part of this repository). `whistle trim` is a Rust port of its
  `whistle_strip.py`.
