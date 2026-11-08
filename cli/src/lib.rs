//! Helpers shared by the `whistle` binary and the tests.

/// Read a 16 kHz mono 16-bit PCM WAV as floats in [-1, 1].
pub fn read_wav(path: &str) -> Result<Vec<f32>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(format!("not a RIFF/WAVE file: {path}"));
    }
    let rd32 = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize;
    let rd16 = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]) as usize;
    let (mut rate, mut ch, mut bits) = (0, 0, 0);
    let mut off = 12;
    while off + 8 <= b.len() {
        let len = rd32(off + 4);
        match &b[off..off + 4] {
            b"fmt " => {
                ch = rd16(off + 10);
                rate = rd32(off + 12);
                bits = rd16(off + 22);
            }
            b"data" => {
                if rate != 16000 || ch != 1 || bits != 16 {
                    return Err(format!("want 16 kHz mono 16-bit, got {rate} Hz {ch} ch {bits} bit"));
                }
                let len = len.min(b.len() - off - 8);
                return Ok((0..len / 2)
                    .map(|i| i16::from_le_bytes([b[off + 8 + 2 * i], b[off + 9 + 2 * i]]) as f32 / 32768.0)
                    .collect());
            }
            _ => {}
        }
        off += 8 + len + (len & 1);
    }
    Err(format!("no data chunk in {path}"))
}
