//! SentencePiece **encoding** for Whistle's tokenizer (BPE, byte fallback,
//! identity normaliser), over the tokenizer blob a `.cact` carries. Matches
//! `sentencepiece`'s `bpe_model.cc`: user-defined pieces match first and are
//! never merged, then adjacent symbols merge, best score first (leftmost on a
//! tie), while the merged string is a piece; leftover unknown symbols become
//! `<0xXX>` byte pieces.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use cactus_whistle::tokenizer::{TK_BYTE, TK_CONTROL, TK_UNKNOWN, TK_USER_DEFINED};

const SPACE: &str = "\u{2581}";

pub struct Encoder {
    pieces: Vec<(String, f32, u8)>,
    /// Mergeable pieces (normal and user-defined) by text.
    ids: HashMap<String, u32>,
    /// Control, unknown and byte pieces by text.
    reserved: HashMap<String, u32>,
    /// User-defined pieces, longest first, for the prefix match.
    user: Vec<String>,
    bytes: [u32; 256],
    unk: u32,
    add_dummy_prefix: bool,
    byte_fallback: bool,
}

#[derive(PartialEq)]
struct Pair {
    score: f32,
    left: usize,
    right: usize,
    size: usize,
}
impl Eq for Pair {}
impl PartialOrd for Pair {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Pair {
    // Max-heap: higher score first, then the leftmost pair.
    fn cmp(&self, o: &Self) -> Ordering {
        self.score
            .total_cmp(&o.score)
            .then_with(|| o.left.cmp(&self.left))
    }
}

fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

impl Encoder {
    /// From a tokenizer blob (the `.cact` RAW tensor, maybe extended with
    /// [`cactus_whistle::tokenizer::extend`]) whose first `n_text` pieces are
    /// text; the rest (language tags) are never produced from text, as in
    /// Cactus, where they are not part of the SentencePiece model.
    pub fn new(blob: &[u8], n_text: usize) -> Result<Encoder, String> {
        if blob.len() < 24 {
            return Err("tokenizer blob too short".into());
        }
        let n = rd_u32(blob, 0) as usize;
        let unk = rd_u32(blob, 16);
        let (add_dummy_prefix, byte_fallback) = (blob[20] != 0, blob[21] != 0);
        let mut pieces = Vec::with_capacity(n);
        let mut off = 24;
        for _ in 0..n {
            if off + 7 > blob.len() {
                return Err("tokenizer blob truncated".into());
            }
            let score = f32::from_bits(rd_u32(blob, off));
            let kind = blob[off + 4];
            let len = u16::from_le_bytes([blob[off + 5], blob[off + 6]]) as usize;
            off += 7;
            let s =
                std::str::from_utf8(blob.get(off..off + len).ok_or("tokenizer blob truncated")?)
                    .map_err(|_| "tokenizer piece is not UTF-8")?;
            pieces.push((s.to_string(), score, kind));
            off += len;
        }
        let (mut ids, mut reserved, mut user) = (HashMap::new(), HashMap::new(), Vec::new());
        let mut bytes = [u32::MAX; 256];
        if n_text > n {
            return Err(format!(
                "tokenizer has {n} pieces, fewer than {n_text} text pieces"
            ));
        }
        pieces.truncate(n_text);
        for (i, (s, _, kind)) in pieces.iter().enumerate() {
            match *kind {
                TK_CONTROL | TK_UNKNOWN => {
                    reserved.entry(s.clone()).or_insert(i as u32);
                }
                TK_BYTE => {
                    reserved.entry(s.clone()).or_insert(i as u32);
                    if let Some(h) = s.strip_prefix("<0x").and_then(|h| h.strip_suffix('>'))
                        && let Ok(v) = u8::from_str_radix(h, 16)
                    {
                        bytes[v as usize] = i as u32;
                    }
                }
                k => {
                    if k == TK_USER_DEFINED {
                        user.push(s.clone());
                    }
                    ids.entry(s.clone()).or_insert(i as u32);
                }
            }
        }
        user.sort_by_key(|s| std::cmp::Reverse(s.len()));
        if byte_fallback && bytes.contains(&u32::MAX) {
            return Err("byte fallback is on but byte pieces are missing".into());
        }
        Ok(Encoder {
            pieces,
            ids,
            reserved,
            user,
            bytes,
            unk,
            add_dummy_prefix,
            byte_fallback,
        })
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    pub fn piece(&self, id: u32) -> &str {
        &self.pieces[id as usize].0
    }

    pub fn id(&self, piece: &str) -> Option<u32> {
        self.ids.get(piece).or(self.reserved.get(piece)).copied()
    }

    /// The final symbols (piece strings) of `text`, before ids and byte fallback.
    pub fn symbols(&self, text: &str) -> Vec<String> {
        let mut norm = String::with_capacity(text.len() + 3);
        if self.add_dummy_prefix && !text.is_empty() {
            norm.push_str(SPACE);
        }
        for c in text.chars() {
            if c == ' ' {
                norm.push_str(SPACE);
            } else {
                norm.push(c);
            }
        }
        // (start, len) per symbol, frozen ones can't merge.
        let mut sym: Vec<(usize, usize, bool)> = Vec::new();
        let mut i = 0;
        while i < norm.len() {
            let rest = &norm[i..];
            let (len, frozen) = match self.user.iter().find(|u| rest.starts_with(u.as_str())) {
                Some(u) => (u.len(), true),
                None => (rest.chars().next().map_or(1, char::len_utf8), false),
            };
            sym.push((i, len, frozen));
            i += len;
        }
        let n = sym.len();
        let mut prev: Vec<isize> = (0..n as isize).map(|i| i - 1).collect();
        let mut next: Vec<isize> = (1..=n as isize)
            .map(|i| if i == n as isize { -1 } else { i })
            .collect();
        let mut heap = BinaryHeap::new();
        let add =
            |heap: &mut BinaryHeap<Pair>, sym: &[(usize, usize, bool)], l: isize, r: isize| {
                if l < 0 || r < 0 {
                    return;
                }
                let (l, r) = (l as usize, r as usize);
                if sym[l].2 || sym[r].2 {
                    return;
                }
                let size = sym[l].1 + sym[r].1;
                if let Some(&id) = self.ids.get(&norm[sym[l].0..sym[l].0 + size]) {
                    heap.push(Pair {
                        score: self.pieces[id as usize].1,
                        left: l,
                        right: r,
                        size,
                    });
                }
            };
        for i in 1..n {
            add(&mut heap, &sym, i as isize - 1, i as isize);
        }
        while let Some(top) = heap.pop() {
            let (l, r) = (top.left, top.right);
            if sym[l].1 == 0 || sym[r].1 == 0 || sym[l].1 + sym[r].1 != top.size {
                continue;
            }
            sym[l].1 = top.size;
            sym[r].1 = 0;
            next[l] = next[r];
            if next[r] >= 0 {
                prev[next[r] as usize] = l as isize;
            }
            add(&mut heap, &sym, prev[l], l as isize);
            add(&mut heap, &sym, l as isize, next[l]);
        }
        let mut out = Vec::new();
        let mut i: isize = if n > 0 { 0 } else { -1 };
        while i >= 0 {
            let (s, len, _) = sym[i as usize];
            out.push(norm[s..s + len].to_string());
            i = next[i as usize];
        }
        out
    }

    /// Token ids of `text`.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for s in self.symbols(text) {
            match self.id(&s) {
                Some(id) => out.push(id),
                None if self.byte_fallback => out.extend(s.bytes().map(|b| self.bytes[b as usize])),
                None => out.push(self.unk),
            }
        }
        out
    }

    /// Pieces `encode` produces, as strings, with byte pieces shown as `<0xXX>`
    /// (sentencepiece's `out_type=str`).
    pub fn encode_pieces(&self, text: &str) -> Vec<String> {
        self.encode(text)
            .into_iter()
            .map(|i| self.piece(i).to_string())
            .collect()
    }
}

/// The pieces to add so every token in `tokens` (and with a leading space)
/// becomes one piece: each prefix of its symbols not yet in the vocabulary, in
/// order (Cactus's `token_pieces`).
pub fn token_pieces(base: &Encoder, tokens: &[String]) -> Vec<String> {
    let mut known: std::collections::HashSet<String> =
        base.pieces.iter().map(|p| p.0.clone()).collect();
    // (Cactus checks against the SentencePiece model, without language tags.)
    let mut out = Vec::new();
    for token in tokens {
        for text in [format!(" {token}"), token.clone()] {
            // Symbols: pieces, with runs of byte pieces decoded back to characters.
            let mut symbols: Vec<String> = Vec::new();
            let mut raw: Vec<u8> = Vec::new();
            let flush = |raw: &mut Vec<u8>, symbols: &mut Vec<String>| {
                if !raw.is_empty() {
                    let s = String::from_utf8_lossy(raw).replace('\u{fffd}', "");
                    symbols.extend(s.chars().map(String::from));
                    raw.clear();
                }
            };
            for p in base.encode_pieces(&text) {
                if p.len() == 6 && p.starts_with("<0x") && p.ends_with('>') {
                    if let Ok(v) = u8::from_str_radix(&p[3..5], 16) {
                        raw.push(v);
                        continue;
                    }
                }
                flush(&mut raw, &mut symbols);
                symbols.push(p);
            }
            flush(&mut raw, &mut symbols);
            for end in 1..=symbols.len() {
                let piece = symbols[..end].concat();
                if known.insert(piece.clone()) {
                    out.push(piece);
                }
            }
        }
    }
    out
}
