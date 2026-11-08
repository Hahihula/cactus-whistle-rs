//! `whistle`: the cactus-whistle engine on a desktop.
//!
//!     whistle transcribe MODEL.cact CLIP.wav [options]
//!     whistle trim MODEL.cact DEPTH OUT.cact
//!     whistle info MODEL.cact

mod info;
mod transcribe;
mod trim;

const USAGE: &str = "\
whistle: Cactus Whistle speech recognition in Rust

  whistle transcribe MODEL.cact CLIP.wav [options]   speech to text
  whistle trim MODEL.cact DEPTH OUT.cact             keep DEPTH decoder layers (2..8)
  whistle info MODEL.cact                            header, tensors, trimmed sizes

`whistle transcribe` with no arguments lists its options.";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((cmd, rest)) = args.split_first() else {
        eprintln!("{USAGE}");
        std::process::exit(2);
    };
    match cmd.as_str() {
        "transcribe" => transcribe::run(rest),
        "trim" => trim::run(rest),
        "info" => info::run(rest),
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
