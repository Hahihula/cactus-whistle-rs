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
            pieces.push(Piece { off: off as u32, len, kind });
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
            let Some(p) = self.pieces.get(id as usize) else { continue };
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
