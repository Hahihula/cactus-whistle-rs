//! `whistle listen MODEL.cact [--lang xx] [--spec guide|fast|c]`: push to
//! talk from the default microphone. Press Enter to start recording, Enter
//! again to stop; the transcript follows. `q` and Enter quits.

use std::{
    io::{BufRead, Write},
    sync::{Arc, Mutex},
    time::Instant,
};

use cactus_whistle::{LANGS, Spec, Whistle};
use cactus_whistle_cli::audio::{self, RATE};
use cpal::{
    FromSample, SampleFormat, SizedSample,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};

pub const USAGE: &str = "\
whistle listen MODEL.cact [--lang xx|auto] [--spec guide|fast|c]
  Push to talk from the default microphone: Enter starts recording, Enter
  stops and transcribes; q + Enter quits.";

pub fn run(args: &[String]) {
    let Some(model_path) = args.first() else {
        eprintln!("{USAGE}");
        std::process::exit(2);
    };
    let (mut lang, mut spec) = (None, Spec::DEFAULT);
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        match (a.as_str(), it.next()) {
            ("--lang", Some(l)) if l == "auto" => lang = None,
            ("--lang", Some(l)) => {
                lang = Some(LANGS.iter().position(|x| x == l).unwrap_or_else(|| {
                    eprintln!("unknown language {l}; one of {}", LANGS.join(", "));
                    std::process::exit(2)
                }))
            }
            ("--spec", Some(s)) => {
                spec = match s.as_str() {
                    "guide" => Spec::Guide,
                    "fast" => Spec::GuideFast,
                    "c" => Spec::CEngine,
                    other => {
                        eprintln!("unknown spec {other}");
                        std::process::exit(2)
                    }
                }
            }
            _ => {
                eprintln!("unknown argument {a}\n\n{USAGE}");
                std::process::exit(2);
            }
        }
    }

    let blob = crate::read(model_path);
    let mut w = Whistle::open(&blob).unwrap_or_else(|e| {
        eprintln!("cannot open {model_path}: {e}");
        std::process::exit(1)
    });
    w.set_spec(spec);

    let device = cpal::default_host()
        .default_input_device()
        .unwrap_or_else(|| {
            eprintln!("no microphone found");
            std::process::exit(1)
        });
    let config = device.default_input_config().unwrap_or_else(|e| {
        eprintln!("cannot read the microphone's configuration: {e}");
        std::process::exit(1)
    });
    let (rate, channels) = (config.sample_rate(), usize::from(config.channels()));
    eprintln!(
        "microphone: {rate} Hz, {channels} channel(s); model: {model_path} ({})",
        spec.name()
    );

    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        eprint!("\nEnter = start recording, q + Enter = quit: ");
        let _ = std::io::stderr().flush();
        line.clear();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 || line.trim() == "q" {
            return;
        }

        let samples = Arc::new(Mutex::new(Vec::<f32>::new()));
        let stream = match record(&device, &config, Arc::clone(&samples)) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("cannot record: {e}");
                continue;
            }
        };
        eprint!("recording... Enter = stop: ");
        let _ = std::io::stderr().flush();
        line.clear();
        let _ = stdin.lock().read_line(&mut line);
        drop(stream);

        let interleaved = std::mem::take(&mut *samples.lock().unwrap());
        let x = audio::resample(&audio::to_mono(&interleaved, channels), rate, RATE);
        let secs = x.len() as f32 / RATE as f32;
        if secs < 0.3 {
            eprintln!("too short ({secs:.1} s)");
            continue;
        }
        let t0 = Instant::now();
        match w.transcribe(&x, lang) {
            Ok(text) => {
                println!("{text}");
                eprintln!(
                    "[{}] {secs:.1} s of audio, {:.2} s to transcribe",
                    LANGS[w.lang],
                    t0.elapsed().as_secs_f32()
                );
            }
            Err(e) => eprintln!("{e}"),
        }
    }
}

/// Starts capturing from `device`, appending interleaved samples as f32.
fn record(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    out: Arc<Mutex<Vec<f32>>>,
) -> Result<cpal::Stream, String> {
    fn build<T: SizedSample>(
        device: &cpal::Device,
        config: &cpal::StreamConfig,
        out: Arc<Mutex<Vec<f32>>>,
    ) -> Result<cpal::Stream, cpal::Error>
    where
        f32: FromSample<T>,
    {
        device.build_input_stream(
            *config,
            move |data: &[T], _: &_| {
                if let Ok(mut buf) = out.lock() {
                    buf.extend(data.iter().map(|&s| s.to_sample::<f32>()));
                }
            },
            |e| eprintln!("microphone error: {e}"),
            None,
        )
    }
    let stream_config: cpal::StreamConfig = (*config).into();
    let stream = match config.sample_format() {
        SampleFormat::F32 => build::<f32>(device, &stream_config, out),
        SampleFormat::I16 => build::<i16>(device, &stream_config, out),
        SampleFormat::I32 => build::<i32>(device, &stream_config, out),
        SampleFormat::U16 => build::<u16>(device, &stream_config, out),
        other => return Err(format!("unsupported sample format {other:?}")),
    }
    .map_err(|e| e.to_string())?;
    stream.play().map_err(|e| e.to_string())?;
    Ok(stream)
}
