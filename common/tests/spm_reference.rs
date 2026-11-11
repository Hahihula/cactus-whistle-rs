//! Compares the Rust SentencePiece encoder with Python's on a reference dump:
//! WHISTLE_SPM_REF=ref.jsonl (first line {"pieces": [...]}, then
//! {"text", "ids"} per line, encoded with the base tokenizer extended by those
//! pieces and the `cs` tag) and WHISTLE_TOKENS=tokens.txt. Skipped without them.

use cactus_whistle_common::spm::{Encoder, token_pieces};
use serde_json::Value;

#[test]
fn matches_sentencepiece() {
    let (Ok(reference), Ok(tokens)) = (
        std::env::var("WHISTLE_SPM_REF"),
        std::env::var("WHISTLE_TOKENS"),
    ) else {
        eprintln!("WHISTLE_SPM_REF / WHISTLE_TOKENS not set; skipped");
        return;
    };
    let model = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../model/whistle.cact"
    ))
    .unwrap();
    let c = cactus_whistle::cact::Cact::open(&model).unwrap();
    let blob = (0..c.n)
        .map(|i| c.tensor(i).unwrap())
        .find(|t| t.dtype == cactus_whistle::cact::DT_RAW)
        .map(|t| c.data(&t).unwrap())
        .unwrap();
    let base = Encoder::new(blob, 8192).unwrap();
    let tokens: Vec<String> = std::fs::read_to_string(tokens)
        .unwrap()
        .lines()
        .map(|l| l.trim().trim_start_matches('\u{feff}').to_string())
        .filter(|l| !l.is_empty())
        .collect();
    let text = std::fs::read_to_string(reference).unwrap();
    let mut lines = text.lines();
    let head: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
    let want: Vec<String> = serde_json::from_value(head["pieces"].clone()).unwrap();
    let pieces = token_pieces(&base, &tokens);
    assert_eq!(pieces, want, "token_pieces");
    let p: Vec<&str> = pieces.iter().map(String::as_str).collect();
    let ext = cactus_whistle::tokenizer::extend(blob, 8192, &p, &["cs"]).unwrap();
    let enc = Encoder::new(&ext, 8192 + pieces.len()).unwrap();
    let (mut n, mut bad) = (0, 0);
    for l in lines {
        let v: Value = serde_json::from_str(l).unwrap();
        let t = v["text"].as_str().unwrap();
        let ids: Vec<u32> = serde_json::from_value(v["ids"].clone()).unwrap();
        let got = enc.encode(t);
        n += 1;
        if got != ids {
            bad += 1;
            if bad <= 10 {
                eprintln!("{t:?}\n  want {ids:?}\n  got  {got:?}");
            }
        }
    }
    assert_eq!(bad, 0, "{bad} of {n} texts differ");
    eprintln!("{n} texts identical");
}
