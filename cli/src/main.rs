//! `whistle`: the cactus-whistle engine on a desktop.
//!
//!     whistle transcribe MODEL.cact AUDIO [options]
//!     whistle listen MODEL.cact [options]
//!     whistle trim MODEL.cact DEPTH OUT.cact
//!     whistle info MODEL.cact
//!     whistle eval MODEL.cact SPLIT.jsonl --out HYP.jsonl [options]
//!     whistle build CHECKPOINT.safetensors --template BASE.cact --out MODEL.cact [options]

mod build;
mod eval;
mod info;
mod listen;
mod transcribe;
mod trim;

const USAGE: &str = "\
whistle: Cactus Whistle speech recognition in Rust

  whistle transcribe MODEL.cact AUDIO [options]      speech to text (WAV, MP3, OGG, FLAC, M4A)
  whistle listen MODEL.cact [options]                push to talk from the microphone
  whistle trim MODEL.cact DEPTH OUT.cact             keep DEPTH decoder layers (2..8)
  whistle info MODEL.cact                            header, tensors, trimmed sizes
  whistle eval MODEL.cact SPLIT.jsonl --out HYP.jsonl  transcribe a dataset split on every core
  whistle build CHECKPOINT.safetensors --template BASE.cact --out MODEL.cact [--lora A] [--depth N]
                                                     a 4-bit model from the float checkpoint

`whistle transcribe` with no arguments lists its options.";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((cmd, rest)) = args.split_first() else {
        eprintln!("{USAGE}");
        std::process::exit(2);
    };
    match cmd.as_str() {
        "transcribe" => transcribe::run(rest),
        "listen" => listen::run(rest),
        "trim" => trim::run(rest),
        "info" => info::run(rest),
        "eval" => eval::run(rest),
        "build" => build::run(rest),
        "-h" | "--help" | "help" => println!("{USAGE}"),
        other => {
            eprintln!("unknown command {other}\n\n{USAGE}");
            std::process::exit(2);
        }
    }
}

/// Reads a file or exits with a message.
fn read(path: &str) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1)
    })
}
