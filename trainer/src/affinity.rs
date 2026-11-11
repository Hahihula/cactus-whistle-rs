//! CPU placement on hybrid Intel CPUs (P-cores and E-cores), Linux only.
//!
//! The trainer issues thousands of small GPU kernels per step from one CPU
//! thread, so its speed follows that thread's core; the evaluator is plain
//! throughput work. With both on the machine, the trainer gets the P-cores
//! and the evaluator the E-cores. `WHISTLE_NO_PIN=1` turns this off.

/// The CPUs of a kind (`core` = P, `atom` = E), from `/sys/devices/cpu_<kind>/cpus`.
#[cfg(target_os = "linux")]
fn cpus(kind: &str) -> Option<Vec<usize>> {
    if std::env::var_os("WHISTLE_NO_PIN").is_some() {
        return None;
    }
    let s = std::fs::read_to_string(format!("/sys/devices/cpu_{kind}/cpus")).ok()?;
    let mut v = Vec::new();
    for part in s.trim().split(',') {
        match part.split_once('-') {
            Some((a, b)) => v.extend(a.parse::<usize>().ok()?..=b.parse::<usize>().ok()?),
            None => v.push(part.parse().ok()?),
        }
    }
    (!v.is_empty()).then_some(v)
}

#[cfg(target_os = "linux")]
fn pin(set: &[usize]) -> bool {
    // SAFETY: a zeroed cpu_set_t filled with CPU_SET, passed by pointer.
    unsafe {
        let mut cs: libc::cpu_set_t = std::mem::zeroed();
        for &c in set {
            libc::CPU_SET(c, &mut cs);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cs) == 0
    }
}

/// Keep the calling thread (and threads it starts later) on the P-cores of a
/// hybrid CPU. Returns whether it did.
pub fn performance_cores() -> bool {
    #[cfg(target_os = "linux")]
    if let (Some(p), Some(_)) = (cpus("core"), cpus("atom")) {
        return pin(&p);
    }
    false
}

/// Keep the calling thread (and threads it starts later) on the E-cores of a
/// hybrid CPU. Returns whether it did.
pub fn efficiency_cores() -> bool {
    #[cfg(target_os = "linux")]
    if let (Some(_), Some(e)) = (cpus("core"), cpus("atom")) {
        return pin(&e);
    }
    false
}
