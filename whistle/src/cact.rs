// Started as a port of the C engine's `.cact` reader in andrisgauracs/whistle-esp32
// (Apache-2.0), and changed since. See README.md, Acknowledgements.

//! Reader for the `.cact` v4 deployment blob (Needle 3 / Whistle).
//!
//! Layout (little-endian):
//!
//! ```text
//! 0x00  header     49 four-byte fields (Header4)
//! 0xC4  codebook   codebook_len * f32 = cb2[4] | cb3[8] | cb4[16]
//! ...   directory  num_tensors * 44-byte records
//! ...   tensor payloads, each 64-byte aligned
//! ```
//!
//! Nothing is copied: the blob is used in place, so on a microcontroller it can be a
//! memory-mapped flash partition.

pub const TAG4: u32 = 0x05E1_2A84;
pub(crate) const HDR4_WORDS: usize = 49;
pub(crate) const HDR4_BYTES: usize = HDR4_WORDS * 4;
pub(crate) const REC_SIZE: usize = 44;

pub const DT_FP16: u8 = 1;
pub const DT_FP32: u8 = 2;
pub const DT_CQ: u8 = 3;
pub const DT_RAW: u8 = 4;

#[inline]
pub(crate) fn rd_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

#[inline]
fn rd_u64(b: &[u8], off: usize) -> u64 {
    rd_u32(b, off) as u64 | (rd_u32(b, off + 4) as u64) << 32
}

/// v4 header geometry, field order as on disk (`_HDR_FMT "<48If"`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Header4 {
    pub tag: u32,
    pub num_tensors: u32,
    pub codebook_len: u32,
    pub kv_window: u32,
    pub kv_bits: u32,
    pub vocab_size: u32,
    pub out_vocab: u32,
    pub d_model: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub num_layers: u32,
    pub qk_head_dim: u32,
    pub v_head_dim: u32,
    pub max_seq_len: u32,
    pub hada_n: u32,
    pub mhc_lanes: u32,
    pub sliding_window: u32,
    pub global_mask_lo: u32,
    pub global_mask_hi: u32,
    pub qkv_conv_taps: u32,
    pub engram_slots: u32,
    pub engram_sub_dim: u32,
    pub engram_tables: u32,
    pub engram_conv_taps: u32,
    pub engram_dilation: u32,
    pub engram_seed_heads: u32,
    pub num_orders: u32,
    pub orders: [u32; 4],
    pub num_sites: u32,
    pub sites: [u32; 16],
    pub rope_theta: f32,
}

impl Header4 {
    fn parse(b: &[u8]) -> Self {
        let mut w = [0u32; HDR4_WORDS];
        for (i, v) in w.iter_mut().enumerate() {
            *v = rd_u32(b, i * 4);
        }
        let mut orders = [0u32; 4];
        orders.copy_from_slice(&w[27..31]);
        let mut sites = [0u32; 16];
        sites.copy_from_slice(&w[32..48]);
        Header4 {
            tag: w[0],
            num_tensors: w[1],
            codebook_len: w[2],
            kv_window: w[3],
            kv_bits: w[4],
            vocab_size: w[5],
            out_vocab: w[6],
            d_model: w[7],
            num_heads: w[8],
            num_kv_heads: w[9],
            num_layers: w[10],
            qk_head_dim: w[11],
            v_head_dim: w[12],
            max_seq_len: w[13],
            hada_n: w[14],
            mhc_lanes: w[15],
            sliding_window: w[16],
            global_mask_lo: w[17],
            global_mask_hi: w[18],
            qkv_conv_taps: w[19],
            engram_slots: w[20],
            engram_sub_dim: w[21],
            engram_tables: w[22],
            engram_conv_taps: w[23],
            engram_dilation: w[24],
            engram_seed_heads: w[25],
            num_orders: w[26],
            orders,
            num_sites: w[31],
            sites,
            rope_theta: f32::from_bits(w[48]),
        }
    }
}

/// One decoded directory record.
#[derive(Clone, Copy, Debug, Default)]
pub struct Tensor {
    pub dtype: u8,
    pub ndim: u8,
    pub shape: [u32; 4],
    pub offset: u64,
    pub nbytes: u64,
    pub group: u32,
    pub bits: u32,
}

impl Tensor {
    /// Reduction length padded up to a whole number of groups.
    #[inline]
    pub fn in_pad(&self) -> usize {
        let g = self.group as usize;
        (self.shape[1] as usize).div_ceil(g) * g
    }
    /// Bytes of packed codebook indices per output row.
    #[inline]
    pub fn row_bytes(&self) -> usize {
        self.in_pad() * self.bits as usize / 8
    }
    /// Quantisation groups per output row.
    #[inline]
    pub fn groups(&self) -> usize {
        self.in_pad() / self.group as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CactError {
    TooSmall,
    BadTag,
    BadGeometry,
    Truncated,
    BadCodebook,
    BadRecord(u32),
}

pub struct Cact<'a> {
    pub base: &'a [u8],
    pub h: Header4,
    /// cb2[4] | cb3[8] | cb4[16], already scaled by 1/sqrt(group).
    pub codebook: [f32; 28],
    dir_off: usize,
    pub n: u32,
}

impl<'a> Cact<'a> {
    pub fn open(blob: &'a [u8]) -> Result<Self, CactError> {
        if blob.len() < HDR4_BYTES {
            return Err(CactError::TooSmall);
        }
        if rd_u32(blob, 0) != TAG4 {
            return Err(CactError::BadTag);
        }
        let h = Header4::parse(blob);
        if h.d_model == 0
            || h.num_layers == 0
            || h.num_heads == 0
            || h.num_kv_heads == 0
            || h.qk_head_dim == 0
            || h.v_head_dim == 0
            || !h.num_heads.is_multiple_of(h.num_kv_heads)
            || h.num_orders > 4
            || h.num_sites > 16
        {
            return Err(CactError::BadGeometry);
        }
        let cb_off = HDR4_BYTES;
        let cb_len = h.codebook_len as usize * 4;
        let dir_off = cb_off + cb_len;
        let dir_len = h.num_tensors as usize * REC_SIZE;
        if dir_off + dir_len > blob.len() {
            return Err(CactError::Truncated);
        }
        if h.codebook_len < 28 {
            return Err(CactError::BadCodebook);
        }
        let mut codebook = [0f32; 28];
        for (i, v) in codebook.iter_mut().enumerate() {
            *v = f32::from_bits(rd_u32(blob, cb_off + i * 4));
        }
        Ok(Cact { base: blob, h, codebook, dir_off, n: h.num_tensors })
    }

    /// Codebook slice for a width of 2, 3 or 4 bits.
    pub fn codebook(&self, bits: u32) -> &[f32] {
        match bits {
            2 => &self.codebook[0..4],
            3 => &self.codebook[4..12],
            4 => &self.codebook[12..28],
            _ => &[],
        }
    }

    pub fn tensor(&self, i: u32) -> Result<Tensor, CactError> {
        if i >= self.n {
            return Err(CactError::BadRecord(i));
        }
        let r = self.dir_off + i as usize * REC_SIZE;
        let b = self.base;
        let ndim = b[r + 1];
        let mut shape = [0u32; 4];
        for (d, s) in shape.iter_mut().enumerate() {
            if d < ndim as usize {
                *s = rd_u32(b, r + 4 + 4 * d);
            }
        }
        Ok(Tensor {
            dtype: b[r],
            ndim,
            shape,
            offset: rd_u64(b, r + 20),
            nbytes: rd_u64(b, r + 28),
            group: rd_u32(b, r + 36),
            bits: rd_u32(b, r + 40),
        })
    }

    /// Payload of a tensor, or `None` if the record runs past the blob.
    pub fn data(&self, t: &Tensor) -> Option<&'a [u8]> {
        let size = self.base.len() as u64;
        if t.offset > size || t.nbytes > size - t.offset {
            return None;
        }
        let o = t.offset as usize;
        Some(&self.base[o..o + t.nbytes as usize])
    }
}
