// Started as a port of the C engine's tokenizer in andrisgauracs/whistle-esp32
// (Apache-2.0), and changed since. See README.md, Acknowledgements.

//! SentencePiece reader over the RAW tokenizer tensor (decode side).
//!
//! Blob layout: `u32 n_pieces, pad, eos, bos, unk; u8 add_dummy_prefix,
//! byte_fallback; u16 pad`, then per piece in id order `f32 score, u8 type,
//! u16 len, len UTF-8 bytes`.

use alloc::string::String;
use alloc::vec::Vec;

use crate::cact::rd_u32;

pub const TK_NORMAL: u8 = 0;
pub const TK_UNKNOWN: u8 = 1;
pub const TK_CONTROL: u8 = 2;
pub const TK_USER_DEFINED: u8 = 3;
pub const TK_BYTE: u8 = 4;

/// U+2581, SentencePiece's space marker.
const SP_SPACE: &[u8] = "\u{2581}".as_bytes();

#[derive(Clone, Copy)]
struct Piece {
    off: u32,
    len: u16,
    kind: u8,
}

pub struct Tokenizer<'a> {
    blob: &'a [u8],
    pieces: Vec<Piece>,
    pub eos_id: u32,
    pub bos_id: u32,
    pub unk_id: u32,
    pub add_dummy_prefix: bool,
}

fn byte_piece_value(s: &[u8]) -> Option<u8> {
    if s.len() != 6 || &s[..3] != b"<0x" || s[5] != b'>' {
        return None;
    }
    let hex = core::str::from_utf8(&s[3..5]).ok()?;
    u8::from_str_radix(hex, 16).ok()
}

fn push_piece(out: &mut Vec<u8>, score: f32, kind: u8, s: &[u8]) {
    out.extend_from_slice(&score.to_le_bytes());
    out.push(kind);
    out.extend_from_slice(&(s.len() as u16).to_le_bytes());
    out.extend_from_slice(s);
}

/// A tokenizer blob extended the way Cactus's fine-tuning extends it:
/// `pieces` are inserted after the first `n_text` pieces of `base` (its text
/// pieces; the rest are its language tags) and a `<|xx|>` tag per code in
/// `langs` is appended. `None` if `base` is malformed, `n_text` exceeds it, or
/// a piece is longer than 65535 bytes.
pub fn extend(base: &[u8], n_text: usize, pieces: &[&str], langs: &[&str]) -> Option<Vec<u8>> {
    if base.len() < 24 {
        return None;
    }
    let n = rd_u32(base, 0) as usize;
    if n_text > n {
        return None;
    }
    let (mut off, mut text_end, mut low) = (24usize, 24usize, f32::MAX);
    for i in 0..n {
        if off + 7 > base.len() {
            return None;
        }
        if i < n_text {
            low = low.min(f32::from_bits(rd_u32(base, off)));
        }
        off += 7 + u16::from_le_bytes([base[off + 5], base[off + 6]]) as usize;
        if off > base.len() {
            return None;
        }
        if i + 1 == n_text {
            text_end = off;
        }
    }
    let total = n + pieces.len() + langs.len();
    if total > 65534 || pieces.iter().any(|p| p.len() > 65535) {
        return None;
    }
    let mut out = Vec::with_capacity(off + 16 * (pieces.len() + langs.len()));
    out.extend_from_slice(&base[..24]);
    out[..4].copy_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&base[24..text_end]);
    // Scores below every base piece, as the fine-tuning tokenizer has them
    // (only the encoder side uses scores; decoding ignores them).
    for (k, p) in pieces.iter().enumerate() {
        push_piece(&mut out, low - 1.0 - k as f32, TK_NORMAL, p.as_bytes());
    }
    out.extend_from_slice(&base[text_end..off]);
    for l in langs {
        let tag = alloc::format!("<|{l}|>");
        push_piece(&mut out, 0.0, TK_USER_DEFINED, tag.as_bytes());
    }
    Some(out)
}

impl<'a> Tokenizer<'a> {
    pub fn new(blob: &'a [u8]) -> Option<Self> {
        if blob.len() < 24 {
            return None;
        }
        let n = rd_u32(blob, 0) as usize;
        if n == 0 || n > 65534 {
            return None;
        }
        let mut pieces = Vec::with_capacity(n);
        let mut off = 24usize;
        for _ in 0..n {
            if off + 7 > blob.len() {
                return None;
            }
            let kind = blob[off + 4];
            let len = u16::from_le_bytes([blob[off + 5], blob[off + 6]]);
            off += 7;
            if off + len as usize > blob.len() {
                return None;
            }
            pieces.push(Piece {
                off: off as u32,
                len,
                kind,
            });
            off += len as usize;
        }
        Some(Tokenizer {
            blob,
            pieces,
            eos_id: rd_u32(blob, 8),
            bos_id: rd_u32(blob, 12),
            unk_id: rd_u32(blob, 16),
            add_dummy_prefix: blob[20] != 0,
        })
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Surface bytes of one piece.
    pub fn piece(&self, id: u32) -> &'a [u8] {
        match self.pieces.get(id as usize) {
            Some(p) => &self.blob[p.off as usize..p.off as usize + p.len as usize],
            None => &[],
        }
    }

    /// Decode ids to text. Control/unknown pieces are skipped, byte pieces
    /// emitted raw, U+2581 becomes a space; with `strip_dummy` one leading
    /// space from the dummy prefix is removed.
    pub fn decode(&self, ids: &[u32], strip_dummy: bool) -> String {
        let mut raw: Vec<u8> = Vec::new();
        for &id in ids {
            let Some(p) = self.pieces.get(id as usize) else {
                continue;
            };
            let s = self.piece(id);
            match p.kind {
                TK_BYTE => {
                    if let Some(v) = byte_piece_value(s) {
                        raw.push(v);
                    }
                }
                TK_CONTROL | TK_UNKNOWN => {}
                _ => raw.extend_from_slice(s),
            }
        }
        let mut out: Vec<u8> = Vec::with_capacity(raw.len());
        let mut r = 0;
        while r < raw.len() {
            if raw[r..].starts_with(SP_SPACE) {
                out.push(b' ');
                r += SP_SPACE.len();
            } else {
                out.push(raw[r]);
                r += 1;
            }
        }
        if strip_dummy && self.add_dummy_prefix && out.first() == Some(&b' ') {
            out.remove(0);
        }
        String::from_utf8_lossy(&out).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob(pieces: &[(&str, u8)]) -> Vec<u8> {
        let mut b = Vec::new();
        for v in [pieces.len() as u32, 0, 1, 2, 3] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&[1, 1, 0, 0]);
        for (k, (p, kind)) in pieces.iter().enumerate() {
            push_piece(&mut b, -(k as f32), *kind, p.as_bytes());
        }
        b
    }

    #[test]
    fn extend_inserts_pieces_after_text_and_appends_tags() {
        let base = blob(&[
            ("<pad>", TK_CONTROL),
            ("</s>", TK_CONTROL),
            ("\u{2581}a", TK_NORMAL),
            ("<|en|>", TK_USER_DEFINED),
            ("<|de|>", TK_USER_DEFINED),
        ]);
        let ext = extend(&base, 3, &["\u{2581}\u{10d}", "\u{10d}"], &["cs"]).unwrap();
        let t = Tokenizer::new(&ext).unwrap();
        let got: Vec<&[u8]> = (0..t.len() as u32).map(|i| t.piece(i)).collect();
        let want: Vec<&[u8]> = [
            "<pad>",
            "</s>",
            "\u{2581}a",
            "\u{2581}\u{10d}",
            "\u{10d}",
            "<|en|>",
            "<|de|>",
            "<|cs|>",
        ]
        .iter()
        .map(|s| s.as_bytes())
        .collect();
        assert_eq!(got, want);
        assert_eq!(t.decode(&[2, 3, 4], true), "a \u{10d}\u{10d}");
        assert!(extend(&base, 6, &[], &[]).is_none());
        assert_eq!(extend(&base, 3, &[], &[]).unwrap(), base);
    }
}
