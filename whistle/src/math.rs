// Started as a port of the C engine's scalar helpers in andrisgauracs/whistle-esp32
// (Apache-2.0), and changed since. See README.md, Acknowledgements.

//! Scalar helpers shared by every stage. Single precision only: many
//! microcontroller FPUs have no `f64`, and one stray double made the C
//! engine 4x slower.

/// IEEE half -> f32.
#[inline]
pub fn f16(h: u16) -> f32 {
    let e = ((h >> 10) & 0x1F) as u32;
    if e == 0 || e == 0x1F {
        return f16_slow(h);
    }
    let bits = ((h as u32 & 0x8000) << 16) | ((e + (127 - 15)) << 23) | ((h as u32 & 0x3FF) << 13);
    f32::from_bits(bits)
}

#[cold]
fn f16_slow(h: u16) -> f32 {
    let sign = (h as u32 & 0x8000) << 16;
    let mut exp = ((h >> 10) & 0x1F) as u32;
    let mut man = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            exp = 127 - 15 + 1;
            // Terminates: man != 0 here (checked above), so shifting left
            // reaches bit 10 within ten steps.
            while man & 0x400 == 0 {
                man <<= 1;
                exp -= 1;
            }
            man &= 0x3FF;
            sign | (exp << 23) | (man << 13)
        }
    } else {
        sign | 0x7F80_0000 | (man << 13)
    };
    f32::from_bits(bits)
}

/// exp(x), ~1e-7 relative: range-reduce to 2^k * 2^f and a degree-5
/// polynomial. Same constants as the C engine so the two agree closely.
#[inline]
#[allow(clippy::approx_constant, clippy::excessive_precision)]
pub fn expf(x: f32) -> f32 {
    if x > 88.0 {
        return 3.402_823_5e38;
    }
    if x < -88.0 {
        return 0.0;
    }
    let z = x * 1.442_695_04;
    let k = (z + if z >= 0.0 { 0.5 } else { -0.5 }) as i32;
    let f = z - k as f32;
    let mut p = 0.001_333_3f32;
    p = p * f + 0.009_618_1;
    p = p * f + 0.055_504_1;
    p = p * f + 0.240_226_5;
    p = p * f + 0.693_147_2;
    p = p * f + 1.0;
    p * f32::from_bits(((k + 127) as u32) << 23)
}

/// Fused multiply-add, `a * b + c` with a single rounding, the same on
/// every target so a desktop run reproduces a device run bit for bit:
/// Xtensa `madd.s`, AArch64 `fmadd`, RISC-V `fmadd.s` (feature `riscv-fpu`:
/// rustc doesn't expose the F extension as a cfg on riscv32imafc, so it can't
/// be detected), else `libm::fmaf`
/// (correctly rounded, slow). On an Xtensa LX7 a fused MAC runs about twice
/// as fast as a separate multiply and add, which the Xtensa backend doesn't
/// schedule well.
#[inline(always)]
pub fn fma(a: f32, b: f32, c: f32) -> f32 {
    // With std, `mul_add` lowers to LLVM's fma: the hardware instruction
    // where there is one, and LLVM can vectorise the loops around it, which
    // it can't do around inline asm (~17% faster encode on an M-series Mac,
    // same bits). `core` has no `mul_add`, so no_std builds use the asm.
    #[cfg(feature = "std")]
    {
        std::primitive::f32::mul_add(a, b, c)
    }
    #[cfg(all(not(feature = "std"), target_arch = "xtensa"))]
    {
        // SAFETY: a pure arithmetic intrinsic, lowered to madd.s (safe on
        // newer toolchains, hence the allow).
        #[allow(unused_unsafe)]
        unsafe {
            core::intrinsics::fmaf32(a, b, c)
        }
    }
    #[cfg(all(not(feature = "std"), target_arch = "aarch64"))]
    {
        let r: f32;
        // SAFETY: one register-only instruction.
        unsafe {
            core::arch::asm!("fmadd {0:s}, {1:s}, {2:s}, {3:s}", out(vreg) r, in(vreg) a, in(vreg) b, in(vreg) c,
                options(pure, nomem, nostack));
        }
        r
    }
    #[cfg(all(not(feature = "std"), target_arch = "riscv32", feature = "riscv-fpu"))]
    {
        let r: f32;
        // SAFETY: one register-only instruction.
        unsafe {
            core::arch::asm!("fmadd.s {0}, {1}, {2}, {3}", out(freg) r, in(freg) a, in(freg) b, in(freg) c,
                options(pure, nomem, nostack));
        }
        r
    }
    #[cfg(all(
        not(feature = "std"),
        not(any(
            target_arch = "xtensa",
            target_arch = "aarch64",
            all(target_arch = "riscv32", feature = "riscv-fpu")
        ))
    ))]
    {
        libm::fmaf(a, b, c)
    }
}

#[inline]
pub fn sqrtf(x: f32) -> f32 {
    libm::sqrtf(x)
}

#[inline]
pub fn logf(x: f32) -> f32 {
    libm::logf(x)
}

#[inline]
pub fn sigm(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + expf(-x))
    } else {
        let e = expf(x);
        e / (1.0 + e)
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigm(x)
}

/// In-place unnormalised fast Walsh-Hadamard transform, `x.len()` a power of 2.
pub fn fwht(x: &mut [f32]) {
    if let Ok(x128) = <&mut [f32; 128]>::try_from(&mut *x) {
        return fwht128(x128);
    }
    let n = x.len();
    let mut len = 1;
    while len < n {
        let mut i = 0;
        while i < n {
            for j in i..i + len {
                let a = x[j];
                let b = x[j + len];
                x[j] = a + b;
                x[j + len] = a - b;
            }
            i += len << 1;
        }
        len <<= 1;
    }
}

/// Three butterfly stages (len 1, 2, 4) on 8 values held in registers.
#[inline(always)]
fn bfly8(v: &mut [f32; 8]) {
    for (a, b) in [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ] {
        let (x, y) = (v[a], v[b]);
        v[a] = x + y;
        v[b] = x - y;
    }
}

/// [`fwht`] for n = 128, the CQ group size. The same butterfly network
/// (every value goes through the same additions, so the result is
/// bit-identical), scheduled as three passes over memory instead of seven:
/// stages 1-3 on 8 consecutive values, stages 4-6 on 8 values at stride 8,
/// then stage 7. Each pass keeps its 8 values in FPU registers.
pub fn fwht128(x: &mut [f32; 128]) {
    for c in 0..16 {
        let mut v: [f32; 8] = x[c * 8..c * 8 + 8].try_into().unwrap();
        bfly8(&mut v);
        x[c * 8..c * 8 + 8].copy_from_slice(&v);
    }
    for c in [0, 64] {
        for o in 0..8 {
            let mut v = [0f32; 8];
            for m in 0..8 {
                v[m] = x[c + o + 8 * m];
            }
            bfly8(&mut v);
            for m in 0..8 {
                x[c + o + 8 * m] = v[m];
            }
        }
    }
    for j in 0..64 {
        let (a, b) = (x[j], x[j + 64]);
        x[j] = a + b;
        x[j + 64] = a - b;
    }
}

pub fn softmax(x: &mut [f32]) {
    let mut m = -3.4e38f32;
    for &v in x.iter() {
        if v > m {
            m = v;
        }
    }
    let mut s = 0.0f32;
    for v in x.iter_mut() {
        *v = expf(*v - m);
        s += *v;
    }
    for v in x.iter_mut() {
        *v /= s;
    }
}

const EPS: f32 = 1e-6;

/// out = x / rms(x).
pub fn rms_unit(x: &[f32], out: &mut [f32]) {
    let n = x.len();
    let mut s = 0.0f32;
    for &v in x {
        s += v * v;
    }
    let r = 1.0 / sqrtf(s / n as f32 + EPS);
    for k in 0..n {
        out[k] = x[k] * r;
    }
}
