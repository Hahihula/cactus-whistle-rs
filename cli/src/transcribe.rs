//! `whistle transcribe model.cact audio-file [options]`
//!
//! The transcript goes to stdout, everything else to stderr, as with the C
//! engine's `wh_run`, so the two can be compared line by line
//! (`tools/compare.sh`).

use std::time::Instant;

use cactus_whistle::{LANGS, Whistle};
use cactus_whistle_cli::load;

pub const USAGE: &str = "\
whistle transcribe MODEL.cact AUDIO [options]
  AUDIO              WAV, MP3, OGG/Vorbis, FLAC or AAC/M4A, any rate and channels
  --lang xx|auto     force a language (en, de, fr, ...); default: detect
  --spec guide|fast|c
                     guide: the porting guide's front end + 5-beam search (default)
                     fast:  the same front end, greedy decoding
                     c:     the C engine's behaviour (andrisgauracs/whistle-esp32)
  --depth N          use N decoder layers of an untrimmed model
  --beam N, --front blog|c, --pe memory|c
                     override single fields of the spec
debugging:
  --mel-out F        write the log-mel spectrogram (f32)
  --enc-out F        write the encoder output (f32)
  --enc-in F         decode an encoder output written by another engine
  --logits-out F --tokens 1,2,3
                     teacher-forced logits for a token sequence
environment:
  WH_HELPER=float|light|full   run whistle::par with a helper thread, as a two-core device does
  WH_BENCH_ENCODE=N            time the encoder N times and exit";

pub fn run(args: &[String]) {
    if args.len() < 2 {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
    let (mut lang, mut depth, mut enc_out, mut mel_out, mut enc_in) =
        (None, None, None, None::<String>, None::<String>);
    let (mut lg_out, mut toks_arg) = (None::<String>, None::<String>);
    let (mut front, mut pe, mut beam) = (None::<bool>, None::<bool>, None::<usize>);
    let mut spec = None::<cactus_whistle::Spec>;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--lang" if i + 1 < args.len() => {
                i += 1;
                lang = if args[i] == "auto" {
                    None
                } else {
                    Some(
                        LANGS
                            .iter()
                            .position(|l| *l == args[i])
                            .expect("unknown language"),
                    )
                };
            }
            "--depth" if i + 1 < args.len() => {
                i += 1;
                depth = Some(args[i].parse::<usize>().expect("depth"));
            }
            "--logits-out" if i + 1 < args.len() => {
                i += 1;
                lg_out = Some(args[i].clone());
            }
            "--tokens" if i + 1 < args.len() => {
                i += 1;
                toks_arg = Some(args[i].clone());
            }
            "--spec" if i + 1 < args.len() => {
                // guide (default) | fast | c; "blog" is accepted for guide.
                i += 1;
                spec = Some(match args[i].as_str() {
                    "guide" | "blog" => cactus_whistle::Spec::Guide,
                    "fast" => cactus_whistle::Spec::GuideFast,
                    "c" => cactus_whistle::Spec::CEngine,
                    other => panic!("unknown spec {other}"),
                });
            }
            "--beam" if i + 1 < args.len() => {
                i += 1;
                beam = Some(args[i].parse::<usize>().expect("beam"));
            }
            "--front" if i + 1 < args.len() => {
                i += 1;
                front = Some(args[i] == "blog");
            }
            "--pe" if i + 1 < args.len() => {
                i += 1;
                pe = Some(args[i] == "memory");
            }
            "--enc-in" if i + 1 < args.len() => {
                i += 1;
                enc_in = Some(args[i].clone());
            }
            "--mel-out" if i + 1 < args.len() => {
                i += 1;
                mel_out = Some(args[i].clone());
            }
            "--enc-out" if i + 1 < args.len() => {
                i += 1;
                enc_out = Some(args[i].clone());
            }
            a => {
                eprintln!("unknown argument {a}\n\n{USAGE}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    if let Ok(mode) = std::env::var("WH_HELPER") {
        // Exercise whistle::par with a real second thread, as a device's
        // helper core does. Results must not change. WH_HELPER=float: float
        // jobs only; =light: also unpack-bound int8 rows (a helper core
        // without the vector unit).
        cactus_whistle::par::set_helper(true);
        cactus_whistle::par::set_helper_int8(match mode.as_str() {
            "float" => cactus_whistle::par::Int8Help::No,
            "light" => cactus_whistle::par::Int8Help::Light,
            _ => cactus_whistle::par::Int8Help::Full,
        });
        std::thread::spawn(|| {
            loop {
                // Give the CPU back while there's no job, instead of
                // burning a core at 100%.
                if !cactus_whistle::par::help() {
                    std::thread::yield_now();
                }
            }
        });
    }
    let blob = std::fs::read(&args[0]).unwrap_or_else(|e| {
        eprintln!("{}: {e}", args[0]);
        std::process::exit(1)
    });
    let mut w = match Whistle::open(&blob) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("cannot open {} as Whistle ({e})", args[0]);
            std::process::exit(1);
        }
    };
    if let Some(sp) = spec {
        w.set_spec(sp);
    }
    if let Some(b) = front {
        w.front_blog = b;
    }
    if let Some(b) = pe {
        w.pe_memory = b;
    }
    if let Some(b) = beam {
        w.beams = b.max(1);
    }
    if let Some(d) = depth {
        w.set_depth(d).expect("depth must be 2..layers");
    }
    let x = load(&args[1]).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1)
    });

    if let Some(p) = mel_out {
        let bytes: Vec<u8> = w.logmel(&x).iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(p, bytes).expect("mel-out");
    }
    if let Ok(n) = std::env::var("WH_BENCH_ENCODE") {
        // Profiling aid: repeat the encoder only.
        for _ in 0..n.parse::<usize>().unwrap_or(1) {
            let t0 = Instant::now();
            let _ = w.encode(&x);
            eprintln!("encode {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        return;
    }
    let t0 = Instant::now();
    let (enc, t) = match &enc_in {
        // Decode an encoder output dumped by another engine (validates the decoder alone).
        Some(p) => {
            let b = std::fs::read(p).expect("enc-in");
            let e: Vec<f32> = b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let t = e.len() / w.d;
            (e, t)
        }
        None => w.encode(&x).expect("clip too short"),
    };
    let t_enc = t0.elapsed().as_secs_f64() * 1e3;
    // A checksum of the encoder output: equal on every machine that runs the
    // same build (the float math is the same everywhere).
    let fnv = enc.iter().fold(0x811C_9DC5u32, |h, v| {
        (h ^ v.to_bits()).wrapping_mul(0x0100_0193)
    });
    eprintln!("enc_fnv {fnv:08x}");
    if let Some(p) = enc_out {
        let bytes: Vec<u8> = enc.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(p, bytes).expect("enc-out");
    }
    if let (Some(p), Some(ts)) = (&lg_out, &toks_arg) {
        // Teacher-forced logits for a token sequence, like the C --logits-out.
        let toks: Vec<u32> = ts.split(',').map(|v| v.parse().expect("token")).collect();
        let mut s = w.session(&enc, t);
        let mut bytes = Vec::new();
        for (pos, &tk) in toks.iter().enumerate() {
            let hid = w.dec_step(tk, pos, &mut s);
            bytes.extend(w.logits(&hid).iter().flat_map(|v| v.to_le_bytes()));
        }
        std::fs::write(p, bytes).expect("logits-out");
    }
    let t0 = Instant::now();
    let ids = w.decode(&enc, t, lang, 320);
    let t_dec = t0.elapsed().as_secs_f64() * 1e3;
    let text = w.detok(&ids);

    println!("{text}");
    eprintln!(
        "tokens {}",
        ids.iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
    if lang.is_none() {
        let l: Vec<String> = LANGS
            .iter()
            .zip(w.lang_logit)
            .map(|(n, v)| format!("{n}={v:.2}"))
            .collect();
        eprintln!("lang_logits {}", l.join(" "));
    }
    eprintln!("score {:.3}", w.score);
    eprintln!(
        "language {}{}\naudio {:.2} s  frames {t}  depth {}",
        LANGS[w.lang],
        if lang.is_none() { " (detected)" } else { "" },
        x.len() as f64 / 16000.0,
        w.depth
    );
    eprintln!(
        "time  encode {t_enc:.0} ms  decode {t_dec:.0} ms ({} tokens, {:.1} ms/token)",
        ids.len(),
        if ids.is_empty() {
            0.0
        } else {
            t_dec / (ids.len() + 1) as f64
        }
    );
}
