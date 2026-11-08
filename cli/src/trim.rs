//! `whistle trim MODEL.cact DEPTH OUT.cact`: a model with only the decoder
//! layers depth DEPTH uses (see `cactus_whistle::ladder`).

use cactus_whistle::ladder;

pub fn run(args: &[String]) {
    let [src, depth, dst] = args else {
        eprintln!("usage: whistle trim MODEL.cact DEPTH OUT.cact");
        std::process::exit(2);
    };
    let depth: usize = depth.parse().unwrap_or_else(|_| {
        eprintln!("DEPTH must be a number");
        std::process::exit(2)
    });
    let blob = crate::read(src);
    let t = ladder::trim(&blob, depth).unwrap_or_else(|e| {
        eprintln!("cannot trim {src}: {e:?}");
        std::process::exit(1)
    });
    std::fs::write(dst, &t.blob).unwrap_or_else(|e| {
        eprintln!("{dst}: {e}");
        std::process::exit(1)
    });
    println!(
        "depth {depth}: decoder layers {:?}, engram at {:?} -> {dst} ({} bytes, {:.2} MB)",
        t.layers,
        t.sites,
        t.blob.len(),
        t.blob.len() as f64 / 1e6
    );
}
