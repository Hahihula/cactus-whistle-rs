//! `whistle build CHECKPOINT.safetensors --template whistle.cact --out MODEL.cact
//! [--lora ADAPTER.safetensors] [--depth N]`: a 4-bit `.cact` from the float
//! checkpoint, with a fine-tuned adapter merged in (Rust's `needle whistle build`).

use cactus_whistle::ladder;
use cactus_whistle_common::{checkpoint::Checkpoint, export, lora::Adapter};

const USAGE: &str = "\
usage: whistle build CHECKPOINT.safetensors --template BASE.cact --out MODEL.cact [options]

  CHECKPOINT       the float model (Cactus's checkpoints/whistle.safetensors)
  --template FILE  a published Whistle .cact (whistle.cact): its tokenizer,
                   mel filterbank and Hadamard permutations are copied
  --out FILE       where to write the model
  --lora FILE      merge this adapter (from whistle-train or needle whistle finetune)
  --depth N        keep only N decoder layers (2..8, see whistle trim)";

pub fn run(args: &[String]) {
    let (mut ck, mut template, mut out, mut lora, mut depth) = (None, None, None, None, None);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = |name: &str| {
            it.next().cloned().unwrap_or_else(|| {
                eprintln!("{name} needs a value\n\n{USAGE}");
                std::process::exit(2)
            })
        };
        match a.as_str() {
            "--template" => template = Some(val("--template")),
            "--out" => out = Some(val("--out")),
            "--lora" => lora = Some(val("--lora")),
            "--depth" => {
                depth = Some(val("--depth").parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("--depth takes a number");
                    std::process::exit(2)
                }))
            }
            s if !s.starts_with("--") && ck.is_none() => ck = Some(s.to_string()),
            s => {
                eprintln!("unexpected {s}\n\n{USAGE}");
                std::process::exit(2)
            }
        }
    }
    let (Some(ck), Some(template), Some(out)) = (ck, template, out) else {
        eprintln!("{USAGE}");
        std::process::exit(2)
    };
    let fail = |e: String| -> ! {
        eprintln!("{e}");
        std::process::exit(1)
    };
    let t0 = std::time::Instant::now();
    let checkpoint = Checkpoint::load(&ck).unwrap_or_else(|e| fail(e));
    let adapter = lora
        .as_deref()
        .map(|p| Adapter::load(p).unwrap_or_else(|e| fail(e)));
    let tpl = crate::read(&template);
    let built = export::build(&checkpoint, adapter.as_ref(), &tpl).unwrap_or_else(|e| fail(e));
    let mut blob = built.blob;
    if let Some(d) = depth.filter(|&d| d != checkpoint.config.dec_layers) {
        blob = ladder::trim(&blob, d)
            .unwrap_or_else(|e| fail(format!("cannot keep {d} layers: {e}")))
            .blob;
    }
    std::fs::write(&out, &blob).unwrap_or_else(|e| fail(format!("{out}: {e}")));
    println!(
        "wrote {out}: {:.2} MB, {} tensors, W4A8, languages {}{}{} ({:.1} s)",
        blob.len() as f64 / 1e6,
        built.names.len(),
        built.languages.join(", "),
        lora.map(|l| format!(", merged {l}")).unwrap_or_default(),
        depth.map(|d| format!(", depth {d}")).unwrap_or_default(),
        t0.elapsed().as_secs_f64()
    );
}
