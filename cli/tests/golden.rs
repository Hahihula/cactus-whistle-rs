//! Golden transcripts on the depth-4 model (`model/whistle_d4.cact`).
//! Skips clips (or the whole test) when the files are missing:
//! `tools/make_clips.sh` and `tools/eval/make_set.py` create them, and
//! `whistle trim model/whistle.cact 4 model/whistle_d4.cact` the model.

use std::path::Path;

use cactus_whistle::{Spec, Whistle};
use cactus_whistle_cli::read_wav;

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");

/// With the C engine's behaviour these match the C engine (tools/compare.sh).
const GOLDEN_C_ENGINE: &[(&str, &str)] = &[
    ("clips/en1.wav", "turned the light red for five seconds."),
    ("clips/en2.wav", "The quick brown fox jumps over the lazy dog."),
    ("clips/fr1.wav", "Bonjour, comment aller vous aujourd'hui ?"),
    ("clips/it1.wav", "Fai l'ampeggiare la luce verde."),
];

/// With the default (porting guide, 5-beam search) these match Cactus's
/// official engine, where the C engine's behaviour does not.
const GOLDEN_GUIDE: &[(&str, &str)] = &[
    ("clips/eval/en_v1_s3.wav", "Tell me everything you know about Renaissance painters."),
    ("clips/eval/en_v2_s0_noisy.wav", "Turn the light red for five seconds."),
    ("clips/eval/en_v4_s0_noisy.wav", "Turn the light red for five seconds."),
    ("clips/eval/en_v2_s5_noisy.wav", "The weather and card will be cloudy with light rain."),
];

fn check(spec: Spec, golden: &[(&str, &str)]) {
    let model = Path::new(ROOT).join("model/whistle_d4.cact");
    let Ok(blob) = std::fs::read(&model) else {
        eprintln!("skip: {} missing", model.display());
        return;
    };
    let mut w = Whistle::open(&blob).expect("open");
    w.set_spec(spec);
    for (clip, want) in golden {
        let path = Path::new(ROOT).join(clip);
        let Ok(x) = read_wav(path.to_str().unwrap()) else {
            eprintln!("skip: {} missing", path.display());
            continue;
        };
        let got = w.transcribe(&x, None).expect("transcribe");
        assert_eq!(&got, want, "{clip} ({})", spec.name());
    }
}

#[test]
fn default_is_the_guide_spec() {
    assert_eq!(Spec::DEFAULT, Spec::Guide);
}

#[test]
fn c_engine_spec_matches_c_engine() {
    check(Spec::CEngine, GOLDEN_C_ENGINE);
}

#[test]
fn guide_matches_official_engine() {
    check(Spec::Guide, GOLDEN_GUIDE);
}

/// A platform kernel plugged in with `quant::set_kernel` must give the same
/// transcripts as the portable loop. This one is the portable math behind a
/// function pointer, counting its calls.
#[test]
fn installed_kernel_is_used_and_exact() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use cactus_whistle::quant::{Kernel, dot128, set_kernel};

    static CALLS: AtomicUsize = AtomicUsize::new(0);

    unsafe fn dot_rows(w: *const i8, x: *const i8, groups: usize, rows: usize, out: &mut [i32]) {
        CALLS.fetch_add(1, Ordering::Relaxed);
        for r in 0..rows {
            for g in 0..groups {
                // SAFETY: the engine passes 128 * groups bytes of `w` and
                // 128 * groups * rows bytes of `x`.
                let (wg, xg) = unsafe {
                    (&*w.add(128 * g).cast::<[i8; 128]>(), &*x.add(128 * (r * groups + g)).cast::<[i8; 128]>())
                };
                out[r * groups + g] = dot128(wg, xg);
            }
        }
    }

    let model = Path::new(ROOT).join("model/whistle_d4.cact");
    let Ok(blob) = std::fs::read(&model) else {
        eprintln!("skip: {} missing", model.display());
        return;
    };
    let Ok(x) = read_wav(Path::new(ROOT).join("clips/en2.wav").to_str().unwrap()) else {
        eprintln!("skip: clips/en2.wav missing");
        return;
    };
    let portable = Whistle::open(&blob).unwrap().transcribe(&x, None).unwrap();
    set_kernel(Some(Kernel { dot_rows, here: || true }));
    let with_kernel = Whistle::open(&blob).unwrap().transcribe(&x, None).unwrap();
    set_kernel(None);
    assert!(CALLS.load(Ordering::Relaxed) > 0, "kernel was never called");
    assert_eq!(with_kernel, portable);
}
