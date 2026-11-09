#!/usr/bin/env python3
"""LoRA fine-tuning of Cactus Whistle, driven by `whistle-train`.

A re-implementation of `needle whistle finetune` (cactus-needle 3.2.8, Apache-2.0)
on needle's own model code, with the knobs the CLI doesn't expose:

  optimiser   weight decay (separately for the adapter and the new vocabulary rows),
              learning-rate multiplier for the rows, warmup fraction, schedule
              (cosine / linear / constant) and its floor, gradient clipping
  adapter     rank, alpha, rank dropout (random rank components zeroed each step)
  evaluation  validation every N steps, best checkpoint kept, early stopping
  safety      resumable state saved with every checkpoint; SIGTERM saves and stops
  data        speed perturbation, SpecAugment, background-noise mixing (all off
              by default)

With the defaults it trains like `needle whistle finetune`: AdamW (weight decay
1e-4 on everything), 5% warmup, cosine decay to 0, clip 1.0, the same 15
projection groups, CQ-W4 straight-through weights and A8 activations.

It reads one JSON config (see `Config` below; `whistle-train` writes it) and
prints one JSON object per line on stdout:

  {"event": "log", "msg": ...}                     progress while loading
  {"event": "start", ...}                          data, vocabulary, schedule
  {"event": "step", "step", "epoch", "loss", "lr", "grad_norm", "clipped", "secs"}
  {"event": "eval", "step", "epoch", "val_loss", "best", "checkpoint"}
  {"event": "early_stop", "step", "patience"}
  {"event": "stopped", "step"}                     after SIGTERM, state saved
  {"event": "done", "best_step", "best_val", "adapter"}
  {"event": "error", "msg"}

Files in out_dir: checkpoints/step_NNNNNN.safetensors (one per evaluation),
adapter_best.safetensors, adapter_last.safetensors, state.msgpack (resume).

Usage: python whistle_train.py CONFIG.json [--resume]
"""
import json
import os
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields

import numpy as np

SAMPLE_RATE = 16000
SECONDS = 30
NEEDLE_VERSION = "3.2.8"


def emit(event, **kw):
    print(json.dumps({"event": event, **kw}, default=float), flush=True)


@dataclass
class Augment:
    # Speed perturbation: each clip resampled by a factor drawn from [lo, hi]
    # with probability `speed_prob` (0 = off). Pitch and tempo change together.
    speed_prob: float = 0.0
    speed_lo: float = 0.9
    speed_hi: float = 1.1
    # SpecAugment on the log-mel features: masks per clip and their widths.
    freq_masks: int = 0
    freq_width: int = 10       # mel bins (of 80)
    time_masks: int = 0
    time_width: float = 0.05   # fraction of the clip's frames, per mask
    # Background noise: WAV files under noise_dir mixed in at an SNR drawn from
    # [snr_lo, snr_hi] dB, with probability noise_prob.
    noise_dir: str = ""
    noise_prob: float = 0.0
    snr_lo: float = 5.0
    snr_hi: float = 20.0

    def spec(self):
        return self.freq_masks > 0 or self.time_masks > 0


@dataclass
class Config:
    train: str = ""
    validation: str = ""
    out_dir: str = ""
    base: str = "checkpoints/whistle.safetensors"
    tokens: str = ""
    language: str = ""
    epochs: int = 3
    batch_size: int = 16
    lr: float = 1e-4
    warmup: float = 0.05            # fraction of all steps
    schedule: str = "cosine"        # cosine | linear | constant
    min_lr_ratio: float = 0.0       # floor of the decay, as a fraction of lr
    weight_decay: float = 1e-4      # AdamW decay of the LoRA matrices
    rows_weight_decay: float = 1e-4  # ... of the new vocabulary rows
    rows_lr_mult: float = 1.0       # learning rate of the rows, relative to lr
    grad_clip: float = 1.0          # global norm; 0 = off
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0       # rank dropout probability
    seed: int = 0
    evals_per_epoch: int = 1        # validation runs per epoch
    early_stop: int = 0             # stop after this many evaluations without improvement; 0 = never
    log_every: int = 0              # steps per "step" event; 0 = about 200 per run
    limit_train: int = 0            # use only the first N clips (smoke tests); 0 = all
    limit_validation: int = 0
    max_steps: int = 0              # stop after N steps; 0 = epochs * steps per epoch
    load_threads: int = 8
    augment: Augment = field(default_factory=Augment)

    @classmethod
    def load(cls, path):
        raw = json.load(open(path, encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise SystemExit(f"{path}: unknown settings {', '.join(sorted(unknown))}")
        aug = raw.pop("augment", {}) or {}
        aknown = {f.name for f in fields(Augment)}
        if set(aug) - aknown:
            raise SystemExit(f"{path}: unknown augment settings {', '.join(sorted(set(aug) - aknown))}")
        return cls(**raw, augment=Augment(**aug))


# ----------------------------------------------------------------- data

def read_split(path, language, threads, limit):
    """Clips as (int16 samples, text, language), like needle's read_clips but
    in parallel and half the memory."""
    from needle.model.whistle_finetune import read_audio
    import re

    root = os.path.dirname(os.path.abspath(path))
    rows = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            lang = row.get("language") or language
            if not lang or not re.fullmatch(r"[A-Za-z0-9_-]+", lang):
                raise SystemExit(f"{path}:{number}: no language code (give \"language\" or set language)")
            rows.append((os.path.join(root, row["audio"]), (row.get("text") or "").strip(), lang))
            if limit and len(rows) >= limit:
                break

    def load(r):
        samples = read_audio(r[0])
        return np.clip(np.round(samples * 32767.0), -32768, 32767).astype(np.int16)

    clips, skipped, t0 = [], 0, time.time()
    with ThreadPoolExecutor(threads) as pool:
        for i, (samples, (_, text, lang)) in enumerate(zip(pool.map(load, rows), rows), 1):
            if not text or not len(samples) or len(samples) > SECONDS * SAMPLE_RATE:
                skipped += 1
            else:
                clips.append((samples, text, lang))
            if i % 5000 == 0:
                emit("log", msg=f"{os.path.basename(path)}: {i}/{len(rows)} clips read ({time.time() - t0:.0f} s)")
    return clips, skipped


class Noise:
    def __init__(self, folder):
        from needle.model.whistle_finetune import read_audio
        self.clips = []
        for dirpath, _, files in os.walk(folder):
            for name in sorted(files):
                if name.lower().endswith(".wav"):
                    x = read_audio(os.path.join(dirpath, name))
                    if len(x) > SAMPLE_RATE:
                        self.clips.append(x)
        if not self.clips:
            raise SystemExit(f"augment.noise_dir {folder}: no WAV files over 1 s")

    def mix(self, x, rng, snr_db):
        n = self.clips[rng.integers(len(self.clips))]
        if len(n) < len(x):
            n = np.tile(n, -(-len(x) // len(n)))
        start = rng.integers(len(n) - len(x) + 1)
        n = n[start:start + len(x)]
        px, pn = float(np.mean(x ** 2)) + 1e-10, float(np.mean(n ** 2)) + 1e-10
        return x + n * np.sqrt(px / (pn * 10 ** (snr_db / 10)))


def batch_arrays(items, idx, cfg_model, aug=None, rng=None, noise=None):
    """Padded wav, lengths and token ids for clips `idx` (needle's _arrays),
    optionally augmented, plus SpecAugment masks."""
    wav = np.zeros((len(idx), SECONDS * SAMPLE_RATE), np.float32)
    lengths = np.zeros(len(idx), np.int32)
    tokens = np.full((len(idx), cfg_model.max_tokens + 1), cfg_model.pad, np.int32)
    for row, i in enumerate(idx):
        samples, ids = items[i][:2]
        x = samples.astype(np.float32) / 32767.0
        if aug is not None:
            if aug.speed_prob and rng.random() < aug.speed_prob:
                import soxr
                factor = rng.uniform(aug.speed_lo, aug.speed_hi)
                x = soxr.resample(x, SAMPLE_RATE * factor, SAMPLE_RATE, quality="HQ").astype(np.float32)
                x = x[:SECONDS * SAMPLE_RATE]
            if noise is not None and rng.random() < aug.noise_prob:
                x = noise.mix(x, rng, rng.uniform(aug.snr_lo, aug.snr_hi)).astype(np.float32)
        wav[row, :len(x)], lengths[row], tokens[row, :len(ids)] = x, len(x), ids
    if aug is None or not aug.spec():
        return wav, lengths, tokens, None
    frames, mels = SECONDS * SAMPLE_RATE // 160 + 8, 80
    fmask = np.ones((len(idx), mels), np.float32)
    tmask = np.ones((len(idx), frames), np.float32)
    for row in range(len(idx)):
        for _ in range(aug.freq_masks):
            w = rng.integers(0, aug.freq_width + 1)
            f0 = rng.integers(0, mels - w + 1)
            fmask[row, f0:f0 + w] = 0
        n = max(1, lengths[row] // 160)
        for _ in range(aug.time_masks):
            w = rng.integers(0, int(aug.time_width * n) + 1)
            t0 = rng.integers(0, max(1, n - w + 1))
            tmask[row, t0:t0 + w] = 0
    return wav, lengths, tokens, (fmask, tmask)


# SpecAugment hook: needle computes the log-mel inside Whistle.encode, so the
# masks reach it through this module-level slot, set while the train step is
# traced (they are jit inputs, not constants).
_SPEC_MASKS = None


def install_spec_hook():
    import needle.model.whistle as wh
    original = wh.features

    def features(cfg, wav, lengths):
        x, frames = original(cfg, wav, lengths)
        if _SPEC_MASKS is not None:
            fmask, tmask = _SPEC_MASKS
            x = x * fmask[:, None, : x.shape[2]] * tmask[:, : x.shape[1], None]
        return x, frames

    wh.features = features


# ----------------------------------------------------------- training

def schedule_fn(cfg, total):
    import optax
    warmup = min(max(1, round(cfg.warmup * total)), total - 1) if cfg.warmup > 0 else 0
    floor = cfg.lr * cfg.min_lr_ratio
    if cfg.schedule == "cosine":
        if warmup:
            return optax.warmup_cosine_decay_schedule(0.0, cfg.lr, warmup, total, floor), warmup
        return optax.cosine_decay_schedule(cfg.lr, total, cfg.min_lr_ratio), warmup
    if cfg.schedule == "linear":
        decay = optax.linear_schedule(cfg.lr, floor, max(1, total - warmup))
    elif cfg.schedule == "constant":
        decay = optax.constant_schedule(cfg.lr)
    else:
        raise SystemExit(f"schedule must be cosine, linear or constant, not {cfg.schedule}")
    if not warmup:
        return decay, 0
    return optax.join_schedules([optax.linear_schedule(0.0, cfg.lr, warmup), decay], [warmup]), warmup


def atomic_write_adapter(path, adapter):
    from needle.model.checkpoints import write_adapter
    tmp = path + ".tmp.safetensors"
    write_adapter(tmp, adapter)
    os.replace(tmp, path)


def main():
    global _SPEC_MASKS
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cfg = Config.load(sys.argv[1])
    resume = "--resume" in sys.argv[2:]
    os.makedirs(os.path.join(cfg.out_dir, "checkpoints"), exist_ok=True)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    import needle
    if getattr(needle, "__version__", NEEDLE_VERSION) != NEEDLE_VERSION:
        emit("log", msg=f"warning: written for cactus-needle {NEEDLE_VERSION}, found {needle.__version__}")
    import jax
    import jax.numpy as jnp
    import optax
    from flax import serialization
    from dataclasses import replace
    from needle.model.finetune import init_lora, lora_target_paths
    from needle.model.quantize import WEIGHT_BITS, configure_deploy, cq_ste_params
    from needle.model.whistle import Whistle
    from needle.model.whistle_finetune import (
        WhistleTokenizer, initial_rows, load_base, read_tokens, token_pieces, tuned_params)

    emit("log", msg=f"loading base {cfg.base}")
    params, mcfg = load_base(cfg.base)
    params = jax.tree.map(lambda a: np.asarray(a).astype(np.float32), params)
    backend = jax.default_backend().lower()
    if backend == "metal":
        mcfg = replace(mcfg, flash=False, remat=False, scan_unroll=max(mcfg.enc_layers, mcfg.dec_layers))
    params = jax.device_put(params)

    emit("log", msg="reading clips")
    train_clips, skipped_t = read_split(cfg.train, cfg.language, cfg.load_threads, cfg.limit_train)
    val_clips, skipped_v = read_split(cfg.validation, cfg.language, cfg.load_threads, cfg.limit_validation)
    if not train_clips or not val_clips:
        raise SystemExit("no usable clips in the train or validation split")
    codes = {lang for *_, lang in train_clips}
    languages = tuple(mcfg.languages) + tuple(sorted(codes - set(mcfg.languages)))
    untrained = {lang for *_, lang in val_clips} - set(languages)
    if untrained:
        raise SystemExit(f"no training clips for {', '.join(sorted(untrained))}")
    tokens = read_tokens(cfg.tokens) if cfg.tokens else []
    pieces = token_pieces(WhistleTokenizer().sp, tokens)
    tokenizer = WhistleTokenizer(pieces, languages)
    tuned = replace(mcfg, text=tokenizer.text, languages=languages)
    not_single = [t for t in tokens if any(len(tokenizer.encode(x)) != 1 for x in (t, " " + t))]

    def encode(clips):
        out = [(s, [mcfg.bos, tuned.tag(lang), *tokenizer.encode(text), mcfg.eos], text) for s, text, lang in clips]
        return [c for c in out if len(c[1]) <= mcfg.max_tokens + 1], sum(len(c[1]) > mcfg.max_tokens + 1 for c in out)

    items, long_t = encode(train_clips)
    validation, long_v = encode(val_clips)
    del train_clips, val_clips
    found = sum(any(t in text for *_, text in items) for t in tokens)

    configure_deploy(act_bits=8, kv_bits=8)
    install_spec_hook()
    model = Whistle(tuned)
    paths = lora_target_paths(params)
    scale = cfg.lora_alpha / cfg.lora_rank
    lora = init_lora(params, paths, cfg.lora_rank, jax.random.PRNGKey(cfg.seed))
    train = {"lora": lora, "rows": jnp.asarray(initial_rows(params["embedding"]["embedding"], mcfg, tokenizer))}

    batch, count = cfg.batch_size, len(items)
    steps_per_epoch = -(-count // batch)
    total = cfg.epochs * steps_per_epoch
    if cfg.max_steps:
        total = min(total, cfg.max_steps)
    sched, warmup = schedule_fn(cfg, total)

    def adamw(lr, wd):
        return optax.adamw(lr, weight_decay=wd)

    rows_sched = (lambda s: sched(s) * cfg.rows_lr_mult)
    labels = {"lora": jax.tree.map(lambda _: "lora", train["lora"]), "rows": "rows"}
    parts = [optax.clip_by_global_norm(cfg.grad_clip)] if cfg.grad_clip > 0 else []
    parts.append(optax.multi_transform(
        {"lora": adamw(sched, cfg.weight_decay), "rows": adamw(rows_sched, cfg.rows_weight_decay)}, labels))
    optimizer = optax.chain(*parts)
    opt_state = optimizer.init(train)

    def merged(train, key):
        lora = train["lora"]
        if key is not None and cfg.lora_dropout > 0:
            keep = 1.0 - cfg.lora_dropout
            out = {}
            for i, (p, v) in enumerate(sorted(lora.items())):
                a = v["A"]
                m = jax.random.bernoulli(jax.random.fold_in(key, i), keep, a.shape[:-2] + (1, a.shape[-1]))
                out[p] = {"A": a * m / keep, "B": v["B"]}
            lora = out
        return cq_ste_params(tuned_params(params, mcfg, tokenizer, lora, scale, train["rows"]), WEIGHT_BITS)

    def loss_fn(train, wav, lengths, tokens, key=None, masks=None):
        global _SPEC_MASKS
        _SPEC_MASKS = masks
        try:
            logits = model.apply({"params": merged(train, key)}, wav, lengths, tokens[:, :-1], quant=True)
        finally:
            _SPEC_MASKS = None
        targets = tokens[:, 1:]
        mask = targets != mcfg.pad
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return (ce * mask).sum() / jnp.maximum(mask.sum(), 1.0)

    @jax.jit
    def train_step(train, opt_state, wav, lengths, tokens, key, masks):
        loss, grads = jax.value_and_grad(loss_fn)(train, wav, lengths, tokens, key, masks)
        norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, train)
        return optax.apply_updates(train, updates), opt_state, loss, norm

    eval_step = jax.jit(loss_fn)

    # Resumable state.
    state_path = os.path.join(cfg.out_dir, "state.msgpack")
    step, best_val, best_step, bad, evals = 0, float("inf"), 0, 0, []
    if resume and os.path.exists(state_path):
        template = {"train": train, "opt_state": opt_state, "step": 0, "best_val": 0.0, "best_step": 0, "bad": 0}
        restored = serialization.from_bytes(template, open(state_path, "rb").read())
        train, opt_state = restored["train"], restored["opt_state"]
        step, best_val, best_step, bad = (int(restored["step"]), float(restored["best_val"]),
                                          int(restored["best_step"]), int(restored["bad"]))
        evals = json.load(open(os.path.join(cfg.out_dir, "evals.json")))
        emit("log", msg=f"resumed at step {step}")

    # Evaluations at evenly spaced points of each epoch, the epoch's end one of
    # them (an interval that doesn't divide the epoch would add a second
    # evaluation right next to its end).
    per_epoch = max(1, min(cfg.evals_per_epoch, steps_per_epoch))
    eval_points = {round(k * steps_per_epoch / per_epoch) % steps_per_epoch for k in range(1, per_epoch + 1)}
    every_eval = steps_per_epoch // per_epoch
    log_every = cfg.log_every or max(1, total // 200)
    emit("start", backend=backend, train_clips=count, val_clips=len(validation),
         minutes=sum(len(c[0]) for c in items) / SAMPLE_RATE / 60,
         skipped=skipped_t + skipped_v, too_long=long_t + long_v,
         tokens=len(tokens), tokens_found=found, not_single=not_single, pieces=len(pieces),
         languages=list(languages), new_languages=[l for l in languages if l not in mcfg.languages],
         lora_groups=len(paths), rows=int(train["rows"].shape[0]), rank=cfg.lora_rank, scale=scale,
         steps=total, steps_per_epoch=steps_per_epoch, warmup=warmup, eval_every=every_eval,
         resumed_from=step or None, needle=getattr(needle, "__version__", "?"))

    def adapter_dict(train):
        return {
            "lora": {"/".join(p): {"A": np.asarray(v["A"]), "B": np.asarray(v["B"])} for p, v in train["lora"].items()},
            "rows": np.asarray(train["rows"]), "scale": float(scale), "base": cfg.base,
            "rank": cfg.lora_rank, "seed": cfg.seed, "pieces": list(pieces), "languages": list(languages),
        }

    def save_state():
        blob = serialization.to_bytes({"train": train, "opt_state": opt_state, "step": step,
                                       "best_val": best_val, "best_step": best_step, "bad": bad})
        with open(state_path + ".tmp", "wb") as f:
            f.write(blob)
        os.replace(state_path + ".tmp", state_path)
        with open(os.path.join(cfg.out_dir, "evals.json"), "w") as f:
            json.dump(evals, f)

    def evaluate():
        nonlocal best_val, best_step, bad
        losses, weights = [], []
        for i in range(0, len(validation), batch):
            part = range(i, min(i + batch, len(validation)))
            wav, lengths, toks, _ = batch_arrays(validation, part, mcfg)
            losses.append(float(eval_step(train, wav, lengths, toks)))
            weights.append(len(part))
        val = float(np.average(losses, weights=weights))
        ckpt = os.path.join(cfg.out_dir, "checkpoints", f"step_{step:06d}.safetensors")
        adapter = adapter_dict(train)
        atomic_write_adapter(ckpt, adapter)
        atomic_write_adapter(os.path.join(cfg.out_dir, "adapter_last.safetensors"), adapter)
        is_best = val < best_val
        if is_best:
            best_val, best_step, bad = val, step, 0
            atomic_write_adapter(os.path.join(cfg.out_dir, "adapter_best.safetensors"), adapter)
        else:
            bad += 1
        evals.append({"step": step, "val_loss": val})
        save_state()
        emit("eval", step=step, epoch=step / steps_per_epoch, val_loss=val, best=is_best, checkpoint=ckpt)

    aug = cfg.augment
    noise = Noise(aug.noise_dir) if aug.noise_dir and aug.noise_prob > 0 else None
    use_aug = aug.speed_prob > 0 or noise is not None or aug.spec()
    base_key = jax.random.PRNGKey(cfg.seed + 1)
    emit("log", msg="compiling the train step (a minute or two)")

    win_loss, win_norm, win_clip, t_win = [], [], 0, time.time()
    order_rng = np.random.default_rng(int(cfg.seed))
    epoch0, skip = divmod(step, steps_per_epoch)
    for _ in range(epoch0):
        order_rng.permutation(count)
    done = step >= total
    for epoch in range(epoch0, cfg.epochs):
        if done:
            break
        order = order_rng.permutation(count)
        for start in range(skip * batch, count, batch):
            rng = np.random.default_rng((cfg.seed, step))
            wav, lengths, toks, masks = batch_arrays(items, order[start:start + batch], mcfg,
                                                     aug if use_aug else None, rng, noise)
            key = jax.random.fold_in(base_key, step) if cfg.lora_dropout > 0 else None
            train, opt_state, loss, norm = train_step(train, opt_state, wav, lengths, toks, key, masks)
            step += 1
            loss, norm = float(loss), float(norm)
            if not np.isfinite(loss):
                emit("error", msg=f"loss became {loss} at step {step}; lower the learning rate")
                sys.exit(1)
            win_loss.append(loss)
            win_norm.append(norm)
            win_clip += bool(cfg.grad_clip and norm > cfg.grad_clip)
            if step % log_every == 0 or step == total:
                now = time.time()
                emit("step", step=step, epoch=step / steps_per_epoch, loss=float(np.mean(win_loss)),
                     loss_max=float(np.max(win_loss)), lr=float(sched(step)), grad_norm=float(np.mean(win_norm)),
                     clipped=win_clip / len(win_loss), secs=(now - t_win) / len(win_loss))
                win_loss, win_norm, win_clip, t_win = [], [], 0, now
            if step % steps_per_epoch in eval_points or step == total:
                evaluate()
                if cfg.early_stop and bad >= cfg.early_stop:
                    emit("early_stop", step=step, patience=cfg.early_stop)
                    done = True
                    break
            if stop["flag"]:
                save_state()
                emit("stopped", step=step)
                sys.exit(3)
            if step >= total:
                done = True
                break
        skip = 0

    if not evals or evals[-1]["step"] != step:
        evaluate()
    emit("done", best_step=best_step, best_val=best_val,
         adapter=os.path.join(cfg.out_dir, "adapter_best.safetensors"), steps=step)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (0, None, 3) and not isinstance(e.code, int):
            emit("error", msg=str(e.code))
            sys.exit(1)
        raise
    except Exception:
        emit("error", msg=traceback.format_exc())
        sys.exit(1)
