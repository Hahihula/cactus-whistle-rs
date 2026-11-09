//! Audio in, for the desktop: any common file format or the microphone,
//! converted to what Whistle takes (16 kHz mono floats).

use symphonia::core::{
    codecs::audio::AudioDecoderOptions,
    errors::Error as SymError,
    formats::{FormatOptions, TrackType, probe::Hint},
    io::MediaSourceStream,
    meta::MetadataOptions,
};

/// Whistle's input rate.
pub const RATE: u32 = 16_000;

/// Reads an audio file (WAV, MP3, OGG/Vorbis, FLAC, AAC/M4A) and returns it
/// as 16 kHz mono samples in [-1, 1].
pub fn load(path: &str) -> Result<Vec<f32>, String> {
    let (interleaved, channels, rate) = decode(path)?;
    if interleaved.is_empty() {
        return Err(format!("{path}: no audio"));
    }
    Ok(resample(&to_mono(&interleaved, channels), rate, RATE))
}

/// Decodes the first audio track: interleaved samples, channel count, rate.
fn decode(path: &str) -> Result<(Vec<f32>, usize, u32), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
    {
        hint.with_extension(ext);
    }
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| format!("{path}: not a supported audio file ({e})"))?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| format!("{path}: no audio track"))?;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| format!("{path}: no audio codec parameters"))?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .map_err(|e| format!("{path}: unsupported codec ({e})"))?;
    let track_id = track.id;

    let (mut out, mut chunk) = (Vec::new(), Vec::new());
    let (mut channels, mut rate) = (0usize, 0u32);
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(e) => return Err(format!("{path}: {e}")),
        };
        if packet.track_id != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(buf) => {
                channels = buf.num_planes();
                rate = buf.spec().rate();
                chunk.resize(buf.samples_interleaved(), 0f32);
                buf.copy_to_slice_interleaved(&mut chunk);
                out.extend_from_slice(&chunk);
            }
            // A corrupt packet: skip it, as players do.
            Err(SymError::DecodeError(_)) => {}
            Err(e) => return Err(format!("{path}: {e}")),
        }
    }
    Ok((out, channels.max(1), rate))
}

/// Averages interleaved channels into one.
pub fn to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|f| f.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Resamples `x` from `from` Hz to `to` Hz with a windowed-sinc low-pass
/// (Hann window, cut-off just under the lower Nyquist frequency), so
/// downsampling doesn't alias. Uses the pure-Rust `libm`, like the engine,
/// so the result is the same on every platform.
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let ratio = f64::from(from) / f64::from(to);
    // Cut-off in cycles per input sample: 95% of the lower Nyquist rate.
    let fc = 0.5 * 0.95 * f64::from(from.min(to)) / f64::from(from);
    // Half-width of the filter in input samples: 16 zero crossings of the
    // (possibly stretched) sinc.
    let half = (16.0 / (2.0 * fc)).ceil() as i64;
    let n_out = ((x.len() as f64) / ratio).floor() as usize;
    let mut y = Vec::with_capacity(n_out);
    for n in 0..n_out {
        let t = n as f64 * ratio;
        let center = t.floor() as i64;
        let mut acc = 0f64;
        let mut wsum = 0f64;
        for k in (center - half + 1)..=(center + half) {
            let d = t - k as f64;
            if d.abs() >= half as f64 {
                continue;
            }
            let arg = 2.0 * fc * d;
            let sinc = if arg == 0.0 {
                1.0
            } else {
                libm::sin(core::f64::consts::PI * arg) / (core::f64::consts::PI * arg)
            };
            let window = 0.5 + 0.5 * libm::cos(core::f64::consts::PI * d / half as f64);
            let h = 2.0 * fc * sinc * window;
            wsum += h;
            if let Some(&v) = usize::try_from(k).ok().and_then(|k| x.get(k)) {
                acc += f64::from(v) * h;
            }
        }
        // Normalising by the filter's sum keeps DC exact despite the window.
        y.push(if wsum != 0.0 {
            (acc / wsum) as f32
        } else {
            0.0
        });
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f64, rate: u32, secs: f64) -> Vec<f32> {
        (0..(f64::from(rate) * secs) as usize)
            .map(|i| (2.0 * core::f64::consts::PI * hz * i as f64 / f64::from(rate)).sin() as f32)
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        // Skip the edges, where the filter runs off the signal.
        let x = &x[x.len() / 10..x.len() * 9 / 10];
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    #[test]
    fn lengths() {
        assert_eq!(resample(&vec![0.0; 44_100], 44_100, 16_000).len(), 16_000);
        assert_eq!(resample(&vec![0.0; 8_000], 8_000, 16_000).len(), 16_000);
        assert_eq!(resample(&[1.0, 2.0], 16_000, 16_000), [1.0, 2.0]);
    }

    #[test]
    fn dc_is_kept() {
        let y = resample(&vec![0.5; 48_000], 48_000, 16_000);
        assert!(y[100..y.len() - 100].iter().all(|v| (v - 0.5).abs() < 1e-4));
    }

    #[test]
    fn passband_tone_keeps_its_level() {
        // 1 kHz, well inside the 8 kHz band: RMS 1/sqrt(2) in and out.
        for from in [44_100, 48_000, 22_050, 8_000] {
            let y = resample(&tone(1000.0, from, 1.0), from, 16_000);
            assert!(
                (rms(&y) - core::f32::consts::FRAC_1_SQRT_2).abs() < 0.01,
                "from {from}: rms {}",
                rms(&y)
            );
        }
    }

    #[test]
    fn content_above_8_khz_is_removed() {
        // 11 kHz at 44.1 kHz would alias to 5 kHz without the low-pass.
        let y = resample(&tone(11_000.0, 44_100, 1.0), 44_100, 16_000);
        assert!(rms(&y) < 0.01, "rms {}", rms(&y));
    }

    #[test]
    fn stereo_is_averaged() {
        assert_eq!(to_mono(&[1.0, 0.0, 0.5, 0.5], 2), [0.5, 0.5]);
    }
}
