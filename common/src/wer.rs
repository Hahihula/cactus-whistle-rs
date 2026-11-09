//! Word error rate as the fine-tuning log reports it: Whisper's
//! `BasicTextNormalizer` (lowercase, bracketed and parenthesised text removed,
//! marks, symbols and punctuation replaced by spaces after NFKC), then word
//! edit distance. A "strict" variant splits on whitespace only, so casing and
//! punctuation count.

use std::collections::BTreeMap;

use unicode_general_category::{GeneralCategory as G, get_general_category};
use unicode_normalization::UnicodeNormalization;

/// Remove `[...]` / `<...>` (opener to the first `]` or `>`) and `(...)`
/// with non-empty content, as Whisper's regexes do.
fn strip_spans(s: &str, open: &[char], close: &[char], nonempty: bool) -> String {
    let c: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < c.len() {
        if open.contains(&c[i])
            && let Some(j) = c[i + 1..].iter().position(|x| close.contains(x))
            && (!nonempty || j > 0)
        {
            i += j + 2;
            continue;
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

fn mark_symbol_punct(c: char) -> bool {
    matches!(
        get_general_category(c),
        G::NonspacingMark
            | G::SpacingMark
            | G::EnclosingMark
            | G::MathSymbol
            | G::CurrencySymbol
            | G::ModifierSymbol
            | G::OtherSymbol
            | G::ConnectorPunctuation
            | G::DashPunctuation
            | G::OpenPunctuation
            | G::ClosePunctuation
            | G::InitialPunctuation
            | G::FinalPunctuation
            | G::OtherPunctuation
    )
}

/// Whisper's `BasicTextNormalizer()`, words joined by single spaces.
pub fn normalize(s: &str) -> String {
    let s = s.to_lowercase();
    let s = strip_spans(&s, &['<', '['], &['>', ']'], false);
    let s = strip_spans(&s, &['('], &[')'], true);
    let s: String = s
        .nfkc()
        .map(|c| if mark_symbol_punct(c) { ' ' } else { c })
        .collect::<String>()
        .to_lowercase();
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Levenshtein distance between two word sequences.
pub fn edits<T: PartialEq>(r: &[T], h: &[T]) -> usize {
    let mut row: Vec<usize> = (0..=h.len()).collect();
    for (i, rw) in r.iter().enumerate() {
        let mut prev = row[0];
        row[0] = i + 1;
        for (j, hw) in h.iter().enumerate() {
            let cur = (row[j + 1] + 1)
                .min(row[j] + 1)
                .min(prev + usize::from(rw != hw));
            prev = row[j + 1];
            row[j + 1] = cur;
        }
    }
    row[h.len()]
}

/// Errors and reference words, normalised and strict.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Count {
    pub clips: usize,
    pub errors: usize,
    pub words: usize,
    pub strict_errors: usize,
    pub strict_words: usize,
}

impl Count {
    pub fn add(&mut self, reference: &str, hypothesis: &str) {
        let (r, h) = (normalize(reference), normalize(hypothesis));
        let (r, h): (Vec<&str>, Vec<&str>) = (
            r.split(' ').filter(|w| !w.is_empty()).collect(),
            h.split(' ').filter(|w| !w.is_empty()).collect(),
        );
        self.errors += edits(&r, &h);
        self.words += r.len();
        let (r, h): (Vec<&str>, Vec<&str>) = (
            reference.split_whitespace().collect(),
            hypothesis.split_whitespace().collect(),
        );
        self.strict_errors += edits(&r, &h);
        self.strict_words += r.len();
        self.clips += 1;
    }

    pub fn merge(&mut self, o: &Count) {
        self.clips += o.clips;
        self.errors += o.errors;
        self.words += o.words;
        self.strict_errors += o.strict_errors;
        self.strict_words += o.strict_words;
    }

    /// Normalised WER (0..1+).
    pub fn wer(&self) -> f64 {
        self.errors as f64 / self.words.max(1) as f64
    }

    pub fn strict(&self) -> f64 {
        self.strict_errors as f64 / self.strict_words.max(1) as f64
    }
}

/// WER per source and overall (key `""`).
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub all: Count,
    pub by_source: BTreeMap<String, Count>,
}

impl Report {
    pub fn add(&mut self, source: &str, reference: &str, hypothesis: &str) {
        let mut c = Count::default();
        c.add(reference, hypothesis);
        self.all.merge(&c);
        self.by_source
            .entry(source.to_string())
            .or_default()
            .merge(&c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_like_whisper() {
        assert_eq!(
            normalize("Doha je Fotometeor, projevující se…"),
            "doha je fotometeor projevující se"
        );
        assert_eq!(
            normalize("„Ano,“ řekl [smích] (potichu) – no!"),
            "ano řekl no"
        );
        assert_eq!(normalize("a () b <x> c"), "a b c");
        assert_eq!(normalize("ŘEDITEL Úřadu"), "ředitel úřadu");
    }

    #[test]
    fn counts_word_errors() {
        let mut c = Count::default();
        c.add("Jedna dva tři.", "jedna, tři čtyři");
        assert_eq!((c.errors, c.words), (2, 3));
        assert_eq!((c.strict_errors, c.strict_words), (3, 3));
    }
}
