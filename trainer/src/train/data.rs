//! Clips, batches and augmentation on the CPU.

use std::path::Path;

use cactus_whistle_common::spm::Encoder;

use super::features::{Frontend, HOP};

pub const RATE: usize = 16_000;
pub const MAX_SECONDS: usize = 30;

/// A clip in memory: int16 samples (half the memory of f32) and its token ids
/// `[bos, tag, text..., eos]`.
pub struct Clip {
    pub samples: Vec<i16>,
    pub ids: Vec<u32>,
}

/// Small, fast, seedable RNG (SplitMix64).
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x2545_F491_4F6C_DD1D)
    }
    /// A generator for (seed, stream), independent of others.
    pub fn stream(seed: u64, stream: u64) -> Rng {
        let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ stream);
        r.next();
        r
    }
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f64()
    }
    /// Uniform integer in [0, n).
    pub fn below(&mut self, n: usize) -> usize {
        if n <= 1 {
            0
        } else {
            (self.f64() * n as f64) as usize % n
        }
    }
    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u = self.f64().max(1e-300);
        let v = self.f64();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.below(i + 1);
            p.swap(i, j);
        }
        p
    }
}

/// A split's rows: (audio path, text, language).
pub fn read_rows(
    path: &Path,
    language: &str,
    limit: usize,
) -> Result<Vec<(String, String, String)>, String> {
    let root = path.parent().unwrap_or(Path::new("."));
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut rows = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("{}:{}: {e}", path.display(), n + 1))?;
        let lang = v["language"]
            .as_str()
            .filter(|s| !s.is_empty())
            .unwrap_or(language)
            .to_string();
        if lang.is_empty()
            || !lang
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(format!(
                "{}:{}: no language code (give \"language\" or set language)",
                path.display(),
                n + 1
            ));
        }
        let audio = v["audio"]
            .as_str()
            .ok_or_else(|| format!("{}:{}: no \"audio\"", path.display(), n + 1))?;
        rows.push((
            root.join(audio).display().to_string(),
            v["text"].as_str().unwrap_or("").trim().to_string(),
            lang,
        ));
        if limit > 0 && rows.len() >= limit {
            break;
        }
    }
    Ok(rows)
}

/// Reads the audio of `rows` on `threads` threads, `progress(done)` now and then.
pub fn read_audio(
    rows: &[(String, String, String)],
    threads: usize,
    progress: &(dyn Fn(usize) + Sync),
) -> Result<Vec<Option<Vec<i16>>>, String> {
    let n = rows.len();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let done = std::sync::atomic::AtomicUsize::new(0);
    let mut out: Vec<Option<Vec<i16>>> = (0..n).map(|_| None).collect();
    let slots: Vec<std::sync::Mutex<Option<Vec<i16>>>> =
        (0..n).map(|_| Default::default()).collect();
    let err = std::sync::Mutex::new(None::<String>);
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    match cactus_whistle_common::audio::load(&rows[i].0) {
                        Ok(x) => {
                            let q: Vec<i16> = x
                                .iter()
                                .map(|v| (v * 32767.0).round().clamp(-32768.0, 32767.0) as i16)
                                .collect();
                            *slots[i].lock().unwrap() = Some(q);
                        }
                        Err(e) => {
                            *err.lock().unwrap() = Some(format!("{}: {e}", rows[i].0));
                        }
                    }
                    let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    if d.is_multiple_of(5000) {
                        progress(d);
                    }
                }
            });
        }
    });
    if let Some(e) = err.into_inner().unwrap() {
        return Err(e);
    }
    for (o, s) in out.iter_mut().zip(slots) {
        *o = s.into_inner().unwrap();
    }
    Ok(out)
}

/// Token ids of a transcript: `[bos, tag, text..., eos]`.
pub fn encode(enc: &Encoder, bos: u32, tag: u32, eos: u32, text: &str) -> Vec<u32> {
    let mut ids = vec![bos, tag];
    ids.extend(enc.encode(text));
    ids.push(eos);
    ids
}

/// Augmentation settings (all off by default).
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Augment {
    pub speed_prob: f64,
    pub speed_lo: f64,
    pub speed_hi: f64,
    pub freq_masks: usize,
    pub freq_width: usize,
    pub time_masks: usize,
    pub time_width: f64,
    pub noise_dir: String,
    pub noise_prob: f64,
    pub snr_lo: f64,
    pub snr_hi: f64,
}

impl Default for Augment {
    fn default() -> Self {
        Augment {
            speed_prob: 0.0,
            speed_lo: 0.9,
            speed_hi: 1.1,
            freq_masks: 0,
            freq_width: 10,
            time_masks: 0,
            time_width: 0.05,
            noise_dir: String::new(),
            noise_prob: 0.0,
            snr_lo: 5.0,
            snr_hi: 20.0,
        }
    }
}

impl Augment {
    pub fn any(&self) -> bool {
        self.speed_prob > 0.0
            || (self.noise_prob > 0.0 && !self.noise_dir.is_empty())
            || self.spec()
    }
    pub fn spec(&self) -> bool {
        self.freq_masks > 0 || self.time_masks > 0
    }
}

/// Background noise clips (WAV files over 1 s under a folder).
pub struct Noise(Vec<Vec<f32>>);

impl Noise {
    pub fn load(dir: &str) -> Result<Noise, String> {
        let mut files = Vec::new();
        let mut stack = vec![std::path::PathBuf::from(dir)];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).map_err(|e| format!("{}: {e}", d.display()))? {
                let p = e.map_err(|e| e.to_string())?.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("wav")) {
                    files.push(p);
                }
            }
        }
        files.sort();
        let mut clips = Vec::new();
        for f in files {
            let x = cactus_whistle_common::audio::load(&f.display().to_string())?;
            if x.len() > RATE {
                clips.push(x);
            }
        }
        if clips.is_empty() {
            return Err(format!("augment.noise_dir {dir}: no WAV files over 1 s"));
        }
        Ok(Noise(clips))
    }

    fn mix(&self, x: &mut [f32], rng: &mut Rng, snr_db: f64) {
        let n = &self.0[rng.below(self.0.len())];
        let start = if n.len() > x.len() {
            rng.below(n.len() - x.len() + 1)
        } else {
            0
        };
        let px = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len() as f64 + 1e-10;
        let pn = (0..x.len())
            .map(|i| (n[(start + i) % n.len()] as f64).powi(2))
            .sum::<f64>()
            / x.len() as f64
            + 1e-10;
        let g = (px / (pn * 10f64.powf(snr_db / 10.0))).sqrt() as f32;
        for (i, v) in x.iter_mut().enumerate() {
            *v += n[(start + i) % n.len()] * g;
        }
    }
}

/// Speed change by `factor` (> 1 faster and shorter): band-limited
/// resampling with a Hann-windowed sinc from a table.
pub fn speed(x: &[f32], factor: f64) -> Vec<f32> {
    const ZEROS: f64 = 16.0;
    const RES: usize = 256;
    let fc = 0.5 * 0.95 * (1.0 / factor).min(1.0); // cycles per input sample
    let half = ZEROS / (2.0 * fc);
    let len = (half * RES as f64).ceil() as usize + 2;
    let table: Vec<f32> = (0..len)
        .map(|i| {
            let d = i as f64 / RES as f64;
            if d >= half {
                return 0.0;
            }
            let a = 2.0 * fc * d;
            let sinc = if a == 0.0 {
                1.0
            } else {
                (std::f64::consts::PI * a).sin() / (std::f64::consts::PI * a)
            };
            (2.0 * fc * sinc * (0.5 + 0.5 * (std::f64::consts::PI * d / half).cos())) as f32
        })
        .collect();
    let n_out = (x.len() as f64 / factor).floor() as usize;
    let h = half.ceil() as i64;
    let mut y = Vec::with_capacity(n_out);
    for n in 0..n_out {
        let t = n as f64 * factor;
        let c = t.floor() as i64;
        let (mut acc, mut ws) = (0f32, 0f32);
        for k in (c - h + 1)..=(c + h) {
            let pos = (t - k as f64).abs() * RES as f64;
            let i = pos as usize;
            if i + 1 >= len {
                continue;
            }
            let f = (pos - i as f64) as f32;
            let w = table[i] * (1.0 - f) + table[i + 1] * f;
            ws += w;
            if k >= 0 && (k as usize) < x.len() {
                acc += x[k as usize] * w;
            }
        }
        y.push(if ws != 0.0 { acc / ws } else { 0.0 });
    }
    y
}

/// A batch ready for the device.
pub struct Batch {
    pub b: usize,
    /// Feature frames per clip in `feats` (cut to the longest, rounded up).
    pub t: usize,
    pub feats: Vec<f32>,
    pub frames: Vec<usize>,
    /// Decoder positions (cut to the longest transcript, rounded up).
    pub s: usize,
    pub input: Vec<u32>,
    pub target: Vec<u32>,
    pub clips: usize,
}

/// Builds the batch of clips `idx`, augmented when `aug` is given.
pub fn batch(
    clips: &[Clip],
    idx: &[usize],
    fe: &Frontend,
    aug: Option<(&Augment, &mut Rng, Option<&Noise>)>,
    pad: u32,
) -> Batch {
    let b = idx.len();
    let m = fe.mels;
    let mut audio: Vec<Vec<f32>> = Vec::with_capacity(b);
    let mut aug = aug;
    for &i in idx {
        let mut x: Vec<f32> = clips[i]
            .samples
            .iter()
            .map(|&v| v as f32 / 32767.0)
            .collect();
        if let Some((a, rng, noise)) = aug.as_mut() {
            if a.speed_prob > 0.0 && rng.f64() < a.speed_prob {
                let f = rng.uniform(a.speed_lo, a.speed_hi);
                x = speed(&x, f);
                x.truncate(MAX_SECONDS * RATE);
            }
            if let Some(nz) = noise
                && rng.f64() < a.noise_prob
            {
                let snr = rng.uniform(a.snr_lo, a.snr_hi);
                nz.mix(&mut x, rng, snr);
            }
        }
        audio.push(x);
    }
    let frames: Vec<usize> = audio.iter().map(|x| Frontend::frames(x.len())).collect();
    let t = frames
        .iter()
        .copied()
        .max()
        .unwrap_or(1)
        .max(1)
        .div_ceil(64)
        * 64;
    let mut feats = vec![0f32; b * t * m];
    for (k, x) in audio.iter().enumerate() {
        fe.compute(x, &mut feats[k * t * m..(k + 1) * t * m], t);
    }
    if let Some((a, rng, _)) = aug.as_mut()
        && a.spec()
    {
        for (k, &n_valid) in frames.iter().enumerate() {
            let f = &mut feats[k * t * m..(k + 1) * t * m];
            for _ in 0..a.freq_masks {
                let w = rng.below(a.freq_width + 1);
                let f0 = rng.below(m - w.min(m) + 1);
                for row in f.chunks_exact_mut(m) {
                    row[f0..(f0 + w).min(m)].fill(0.0);
                }
            }
            let n = n_valid.max(1);
            for _ in 0..a.time_masks {
                let w = rng.below((a.time_width * n as f64) as usize + 1);
                let t0 = rng.below((n + 1).saturating_sub(w).max(1));
                for tt in t0..(t0 + w).min(t) {
                    f[tt * m..(tt + 1) * m].fill(0.0);
                }
            }
        }
    }
    let longest = idx.iter().map(|&i| clips[i].ids.len()).max().unwrap_or(2);
    let s = (longest - 1).div_ceil(16) * 16;
    let mut input = vec![pad; b * s];
    let mut target = vec![pad; b * s];
    for (k, &i) in idx.iter().enumerate() {
        let ids = &clips[i].ids;
        for j in 0..ids.len() - 1 {
            input[k * s + j] = ids[j];
            target[k * s + j] = ids[j + 1];
        }
    }
    let _ = HOP;
    Batch {
        b,
        t,
        feats,
        frames,
        s,
        input,
        target,
        clips: b,
    }
}
