//! `whistle info MODEL.cact`: what's in a model and what trimming saves.

use cactus_whistle::{
    cact::{Cact, DT_CQ, DT_FP16, DT_FP32, DT_RAW},
    ladder,
};

pub fn run(args: &[String]) {
    let [path] = args else {
        eprintln!("usage: whistle info MODEL.cact");
        std::process::exit(2);
    };
    let blob = crate::read(path);
    let c = Cact::open(&blob).unwrap_or_else(|e| {
        eprintln!("{path}: not a .cact v4 model ({e:?})");
        std::process::exit(1)
    });
    let h = &c.h;
    println!("{path}: {} bytes ({:.2} MB), {} tensors", blob.len(), blob.len() as f64 / 1e6, c.n);
    println!(
        "decoder: {} layers, d_model {}, {} heads ({} kv), head dim {}/{}, vocab {}, engram sites at layers {:?}",
        h.num_layers,
        h.d_model,
        h.num_heads,
        h.num_kv_heads,
        h.qk_head_dim,
        h.v_head_dim,
        h.vocab_size,
        &h.sites[..h.num_sites as usize]
    );

    let mut by_type = [(0usize, 0u64); 5];
    for i in 0..c.n {
        let t = c.tensor(i).expect("directory");
        let k = (t.dtype as usize).min(4);
        by_type[k].0 += 1;
        by_type[k].1 += t.nbytes;
    }
    for (name, k) in [("CQ (codebook)", DT_CQ), ("fp16", DT_FP16), ("fp32", DT_FP32), ("raw", DT_RAW)] {
        let (n, b) = by_type[k as usize];
        if n > 0 {
            println!("  {name:<14} {n:4} tensors  {:7.2} MB", b as f64 / 1e6);
        }
    }

    let layers = h.num_layers as usize;
    if layers >= 2 {
        println!("decoder depth (`whistle trim`): layers kept, model size");
        for d in 2..=layers {
            match ladder::trim(&blob, d) {
                Ok(t) => println!("  {d}: {:<26} {:6.2} MB", format!("{:?}", t.layers), t.blob.len() as f64 / 1e6),
                Err(e) => println!("  {d}: {e:?}"),
            }
        }
    }
}
