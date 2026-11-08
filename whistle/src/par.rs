//! Row-level parallelism with an optional helper core.
//!
//! Every matmul is parallel across output rows. The engine stays portable and
//! single-threaded by default: [`rows`] runs everything on the caller. A
//! platform with a spare core enables [`set_helper`] and calls [`help`] from
//! that core in a loop; [`rows`] then publishes the job and both cores steal
//! chunks of rows from a shared counter until none are left.
//!
//! The helper takes one chunk per [`help`] call, so it can yield between
//! chunks (e.g. inside an async executor, next to tasks that must keep
//! running, such as a microphone stream). The job lives on the caller's stack; a reference count
//! keeps the caller from returning while the helper still holds it.

use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU8, AtomicUsize, Ordering};

struct Job<'a> {
    f: &'a (dyn Fn(usize, usize) + Sync),
    n: usize,
    chunk: usize,
    next: AtomicUsize,
    done: AtomicUsize,
}

static HELPER: AtomicBool = AtomicBool::new(false);
/// Which int8 matmul jobs the helper takes, an [`Int8Help`] as u8.
static HELPER_INT8: AtomicU8 = AtomicU8::new(Int8Help::Full as u8);

/// How much int8 matmul work the helper core should take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Int8Help {
    /// Float rows only.
    No = 0,
    /// Also int8 rows that are unpack-bound (at most [`light_rows`]
    /// activation rows, e.g. the decoder), for a helper core without the
    /// vector unit (chips where only one core has it). There the dot
    /// product is a small part of the row, so the portable dot is fine.
    Light = 1,
    /// Every int8 row: the helper core has the vector unit too.
    Full = 2,
}
static JOB: AtomicPtr<Job<'static>> = AtomicPtr::new(core::ptr::null_mut());
/// Helpers currently holding a reference to the published job.
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

/// Announce (or withdraw) a helper that calls [`help`].
pub fn set_helper(on: bool) {
    HELPER.store(on, Ordering::Release);
}

/// At most this many activation rows count as "light" (unpack-bound) for
/// [`Int8Help::Light`]. Default 2 (greedy decoding); a beam search runs its
/// beams as rows of one matmul.
static LIGHT_ROWS: AtomicUsize = AtomicUsize::new(2);

/// Set the [`Int8Help::Light`] row limit.
pub fn set_light_rows(n: usize) {
    LIGHT_ROWS.store(n, Ordering::Relaxed);
}

/// The [`Int8Help::Light`] row limit.
pub fn light_rows() -> usize {
    LIGHT_ROWS.load(Ordering::Relaxed)
}

/// Which int8 matmul jobs the helper takes (default [`Int8Help::Full`]).
pub fn set_helper_int8(level: Int8Help) {
    HELPER_INT8.store(level as u8, Ordering::Release);
}

/// [`rows`] for int8 matmul rows. `light`: the rows are unpack-bound (few
/// activation rows), worth sharing even with a helper that lacks the SIMD
/// unit; see [`Int8Help`].
pub fn rows_int8(n: usize, chunk: usize, light: bool, f: &(dyn Fn(usize, usize) + Sync)) {
    let level = HELPER_INT8.load(Ordering::Acquire);
    if level == Int8Help::Full as u8 || (light && level == Int8Help::Light as u8) {
        rows(n, chunk, f);
    } else {
        f(0, n);
    }
}

/// Run `f(r0, r1)` over `[0, n)` in chunks of `chunk` rows, sharing the work
/// with the helper if there is one. `f` must only write state owned by its
/// rows and must not need the SIMD unit (use [`rows_int8`] for that).
/// Returns when every row is done.
pub fn rows(n: usize, chunk: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    let chunk = chunk.max(1);
    if n <= chunk || !HELPER.load(Ordering::Acquire) {
        f(0, n);
        return;
    }
    let job = Job { f, n, chunk, next: AtomicUsize::new(0), done: AtomicUsize::new(0) };
    // SAFETY (lifetime): the pointer is unpublished and every helper has let
    // go of it (ACTIVE == 0) before `job` goes out of scope below.
    let ptr = &job as *const Job<'_> as *mut Job<'static>;
    JOB.store(ptr, Ordering::Release);
    while run_chunk(&job) {}
    while job.done.load(Ordering::Acquire) < n {
        core::hint::spin_loop();
    }
    // SeqCst on both sides: unpublish-then-check here and announce-then-check
    // in `help` are store->load pairs, which weaker orderings may reorder.
    JOB.store(core::ptr::null_mut(), Ordering::SeqCst);
    while ACTIVE.load(Ordering::SeqCst) != 0 {
        core::hint::spin_loop();
    }
}

/// Take and run one chunk; false when the job has no rows left.
fn run_chunk(job: &Job<'_>) -> bool {
    let r0 = job.next.fetch_add(job.chunk, Ordering::AcqRel);
    if r0 >= job.n {
        return false;
    }
    let r1 = (r0 + job.chunk).min(job.n);
    (job.f)(r0, r1);
    job.done.fetch_add(r1 - r0, Ordering::AcqRel);
    true
}

/// For the helper core: run one chunk of the current job, if any. Returns
/// whether it did any work.
pub fn help() -> bool {
    if JOB.load(Ordering::Acquire).is_null() {
        return false;
    }
    ACTIVE.fetch_add(1, Ordering::SeqCst);
    // Re-read after announcing ourselves: the caller only frees a job once it
    // is unpublished *and* ACTIVE has drained.
    let p = JOB.load(Ordering::SeqCst);
    let worked = if p.is_null() {
        false
    } else {
        // SAFETY: published and we hold an ACTIVE reference.
        run_chunk(unsafe { &*p })
    };
    ACTIVE.fetch_sub(1, Ordering::AcqRel);
    worked
}
