"""Reference values for checking the Rust trainer against Cactus's JAX code
(development only; training itself needs no Python).

One fixed batch of 4 validation clips through needle's Whistle at full f32
precision, with a random LoRA (B nonzero) and the initial vocabulary rows:
features, encoder output, logits, loss and every gradient, with and without
activation quantisation.

    pip install "cactus-compute[train]"     # cactus-needle 3.2.8, JAX
    python ref_model.py WORKDIR OUT.safetensors
    whistle-train parity OUT.safetensors WORKDIR/checkpoints/whistle.safetensors model/whistle.cact
    NOQUANT=1 whistle-train parity ...      # the tight comparison (see README)

WORKDIR holds checkpoints/whistle.safetensors, tokens.txt and
dataset_v2/validation.jsonl (the layout of the fine-tuning workspace).
"""
import json, os, sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_default_matmul_precision", "highest")
import optax
from dataclasses import replace
from safetensors.numpy import save_file
from needle.model.finetune import init_lora, lora_target_paths
from needle.model.quantize import WEIGHT_BITS, configure_deploy, cq_ste_params
from needle.model.whistle import Whistle, features
from needle.model.whistle_finetune import (WhistleTokenizer, initial_rows, load_base, read_audio,
                                           read_tokens, token_pieces, tuned_params)

root = os.path.join(os.path.abspath(sys.argv[1]), '')
params, mcfg = load_base(root + 'checkpoints/whistle.safetensors')
params = jax.tree.map(lambda a: np.asarray(a).astype(np.float32), params)
tokens = read_tokens(root + 'tokens.txt')
pieces = token_pieces(WhistleTokenizer().sp, tokens)
languages = tuple(mcfg.languages) + ('cs',)
tok = WhistleTokenizer(pieces, languages)
tuned = replace(mcfg, text=tok.text, languages=languages)

rows_in = [json.loads(l) for l in open(root + 'dataset_v2/validation.jsonl', encoding='utf-8')][:4]
B = len(rows_in)
wav = np.zeros((B, 30 * 16000), np.float32); lengths = np.zeros(B, np.int32)
toks = np.full((B, mcfg.max_tokens + 1), mcfg.pad, np.int32)
for i, r in enumerate(rows_in):
    s = read_audio(os.path.join(root + 'dataset_v2', r['audio']))
    s = np.clip(np.round(s * 32767.0), -32768, 32767).astype(np.int16).astype(np.float32) / 32767.0
    ids = [mcfg.bos, tuned.tag('cs'), *tok.encode(r['text'].strip()), mcfg.eos]
    wav[i, :len(s)], lengths[i], toks[i, :len(ids)] = s, len(s), ids

configure_deploy(act_bits=8, kv_bits=8)
model = Whistle(tuned)
paths = lora_target_paths(params)
rank, scale = 4, 2.0
lora = init_lora(params, paths, rank, jax.random.PRNGKey(0))
key = jax.random.PRNGKey(7)
for p in sorted(lora):
    key, sub = jax.random.split(key)
    lora[p]["B"] = jax.random.normal(sub, lora[p]["B"].shape) * 0.01
rows = jnp.asarray(initial_rows(params["embedding"]["embedding"], mcfg, tok))
train = {"lora": lora, "rows": rows}

def merged(train):
    return cq_ste_params(tuned_params(params, mcfg, tok, train["lora"], scale, train["rows"]), WEIGHT_BITS)

def loss_fn(train, quant=True):
    logits = model.apply({"params": merged(train)}, wav, lengths, toks[:, :-1], quant=quant)
    targets = toks[:, 1:]
    mask = targets != mcfg.pad
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return (ce * mask).sum() / jnp.maximum(mask.sum(), 1.0), logits

(loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(train)
(loss_nq, logits_nq), grads_nq = jax.value_and_grad(lambda t: loss_fn(t, False), has_aux=True)(train)
m = merged(train)
x, frames = features(tuned, jnp.asarray(wav), jnp.asarray(lengths))
enc, enc_len = model.apply({"params": m}, wav, lengths, True, method=Whistle.encode)
enc_nq, _ = model.apply({"params": m}, wav, lengths, False, method=Whistle.encode)
out = {
    "wav": wav, "lengths": lengths, "tokens": toks,
    "features": np.asarray(x), "frames": np.asarray(frames),
    "enc": np.asarray(enc), "enc_lengths": np.asarray(enc_len),
    "logits_head": np.asarray(logits[:, :24]),
    "nq/enc": np.asarray(enc_nq), "nq/logits_head": np.asarray(logits_nq[:, :24]), "nq/grad/rows": np.asarray(grads_nq["rows"]),
    "loss": np.asarray([loss], np.float32), "loss_noquant": np.asarray([loss_nq], np.float32),
    "rows": np.asarray(rows), "grad/rows": np.asarray(grads["rows"]),
    "q/embedding": np.asarray(m["embedding"]["embedding"]),
    "q/encoder_q_proj": np.asarray(m["encoder"]["layers"]["block"]["self_attn"]["q_proj"]["kernel"]),
    "q/stem_out": np.asarray(m["stem"]["out"]["kernel"]),
    "q/stack_phi_res": np.asarray(m["stack"]["mhc_phi_res"]),
}
for p, v in lora.items():
    n = "/".join(p)
    out[f"lora/{n}/A"] = np.asarray(v["A"]); out[f"lora/{n}/B"] = np.asarray(v["B"])
    out[f"grad/{n}/A"] = np.asarray(grads["lora"][p]["A"]); out[f"grad/{n}/B"] = np.asarray(grads["lora"][p]["B"])
    out[f"nq/grad/{n}/A"] = np.asarray(grads_nq["lora"][p]["A"]); out[f"nq/grad/{n}/B"] = np.asarray(grads_nq["lora"][p]["B"])
save_file({k: np.ascontiguousarray(v) for k, v in out.items()}, sys.argv[2],
          metadata={"scale": str(scale), "rank": str(rank), "pieces": json.dumps(pieces),
                    "languages": json.dumps(list(languages))})
print("loss", float(loss), "noquant", float(loss_nq), "frames", np.asarray(frames), "enc", enc.shape, np.asarray(enc_len))
