"""Reward = -MSE(critic_fwd(explanation), gold_activation) on L2-normalized
vectors (so MSE = 2(1-cos)). Set NLA_LOG_MSE_REWARD=1 to use -log(MSE) instead;
GRPO-normalisation makes them near-equivalent in practice. Two opt-in bonuses
stack on top: +alpha for hitting the known target word (NLA_TARGET_WORD_BONUS_ALPHA)
and +beta for hitting a top-K logit-lens token of the gold diff (NLA_LOGIT_LENS_BONUS_BETA).

Called from miles.rollout.rm_hub via --custom-rm-path nla.reward.nla_rm.
sglang_rollout.py:255 fires this per-sample as each generation completes.

The forward runs on the CRITIC TRAINER (GPUs 2,3, FSDP-sharded, idle during
generation) via Ray remote — live weights, no duplicate model, no checkpoint
staleness. RolloutManager.set_critic_handles (train.py:26) stashed the Ray
handles on args before any rollout fires.

Async accumulator: collect samples until --nla-reward-batch-size is hit (or a
50ms timeout for the tail), then dispatch one batched critic_fwd to both critic
ranks. Event loop stays free during the forward (asyncio.to_thread around ray.get)
so later groups' SGLang callbacks fire → generation pipelines with reward compute.
"""

import asyncio
import json
import logging
import math
import os
import re
import time

import ray
import torch
import wandb

from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

from nla.config import load_nla_config
from nla.schema import extract_explanation, normalize_activation


_MSE_EPS = 1e-8
_USE_LOG_MSE_REWARD = bool(int(os.environ.get("NLA_LOG_MSE_REWARD", "0")))
# Under -mse_nrm, 0.0 is the BEST reward (perfect reconstruction) and -2.0 is
# orthogonal. Under -log(MSE), 0.0 corresponds to mse=1 (mid-range). Use the
# orthogonal-equivalent value so a failed extraction is never advantaged.
FAILED_EXTRACTION_REWARD = -math.log(2.0) if _USE_LOG_MSE_REWARD else -2.0
# Flush timeout: originally 50ms for single-sample async_rm path where
# samples arrive fast (per-sample, not per-group) and 50ms catches tail
# stragglers. With --group-rm routing through the accumulator, groups arrive
# staggered over the ~60s generation window — first group lands alone, 50ms
# fires before more arrive, batch-size=256 never kicks in. 5s lets ~10-30
# groups coalesce; adds ≤5s latency in a 100s+ rollout. Override with
# NLA_REWARD_FLUSH_SECS if the stagger pattern differs.
_TAIL_FLUSH_SECONDS = float(os.environ.get("NLA_REWARD_FLUSH_SECS", "5.0"))
# Opt-in: log every drained batch's (prompt, verbalization, reward) as a wandb
# Table on `rollout/verbalizations`, so training can be debugged by reading
# what the actor actually wrote next to the reward it got. Off by default —
# these tables are the priciest thing this module logs (full text per row,
# every drain, every rollout). This process already holds a secondary wandb
# session onto the shared run (miles/ray/rollout.py's RolloutManager calls
# init_tracking(primary=False)), so wandb.log here lands in the same run as
# the actor/critic train/* metrics.
_LOG_SAMPLES = bool(int(os.environ.get("NLA_WANDB_LOG_SAMPLES", "0")))
# Opt-in: append the same per-drain rows to a local text file instead of (or
# alongside) wandb — no browser/UI needed, just `tail -f` / open in an editor.
# Independent of NLA_WANDB_LOG_SAMPLES: set either or both. Grows unbounded
# for the life of the run (append mode, one drain's worth of rows per block) —
# it's meant for a targeted debugging session, not left on for a full run.
_VERBALIZATIONS_LOG_PATH = os.environ.get("NLA_VERBALIZATIONS_LOG")
# Opt-in bonus added to the raw per-sample reward when the analogy's target
# word appears (whole-word/case-insensitive) in the extracted explanation,
# applied before GRPO group normalization. 0.0 = off (default, no behavior change).
_TARGET_WORD_BONUS_ALPHA = float(os.environ.get("NLA_TARGET_WORD_BONUS_ALPHA", "0.0"))
# Opt-in: for taboo-origin samples, require the literal secret word to appear
# in the explanation -- drop the generic taboo-game-language fallback below
# (_LOOSE_TABOO_KEYWORDS) entirely. Default off (0) reproduces prior behavior
# (analogy_combo_taboo_rl_alpha05_v1 and earlier), where generic secret/hint/
# reveal-style phrasing also earned the bonus even without leaking the word.
_TABOO_EXACT_WORD_ONLY = bool(int(os.environ.get("NLA_TABOO_EXACT_WORD_ONLY", "0")))
# Opt-in label-free bonus: decode the gold activation (a finetuned-minus-base
# diff) with the ORIGINAL model's LM head (logit lens) and add beta to the raw
# reward when the explanation hits at least one of the top-K decoded tokens.
# Unlike alpha, needs no target word -- the targets come from the vector itself.
# Stacks additively with alpha, before GRPO normalization. 0.0 = off (default).
# NLA_LOGIT_LENS_MODEL (local HF dir or hub id of the base model) is required
# when beta != 0; only its final-norm weight + unembedding are loaded (CPU).
_LOGIT_LENS_BONUS_BETA = float(os.environ.get("NLA_LOGIT_LENS_BONUS_BETA", "0.0"))
_LOGIT_LENS_TOPK = int(os.environ.get("NLA_LOGIT_LENS_TOPK", "10"))
_LOGIT_LENS_MODEL = os.environ.get("NLA_LOGIT_LENS_MODEL")
# Decoded tokens shorter than this (after stripping) are skipped: subword
# fragments like "ing"/"s" would match nearly any explanation.
_LOGIT_LENS_MIN_CHARS = int(os.environ.get("NLA_LOGIT_LENS_MIN_CHARS", "3"))
# Which vocab matrix the diff is compared against: "unembed" = logit lens
# (LM head, dot product after the final-norm gain); "embed" = cosine against
# the INPUT embedding matrix, sometimes more readable at middle layers.
_LOGIT_LENS_SPACE = os.environ.get("NLA_LOGIT_LENS_SPACE", "unembed")
# Subtract the mean activation of the RL parquet before decoding. Removes the
# direction shared by every diff (the finetune's generic shift), which otherwise
# puts the same tokens in every row's top-K and hands out beta for free.
_LOGIT_LENS_CENTER = bool(int(os.environ.get("NLA_LOGIT_LENS_CENTER", "0")))

_TOKENIZER = None
_CFG = None
_LENS = None  # (vocab matrix [V,d] bf16, gamma [d], strings, eligible [V], mean [d] | None); iff beta != 0

_pending: list[tuple[Sample, asyncio.Future]] = []
_drain_task: asyncio.Task | None = None


def _lazy_init(args):
    global _TOKENIZER, _CFG
    if _TOKENIZER is not None:
        return
    # Tokenizer and sidecar from the critic's HF dir. FSDP: args.critic_load IS
    # the HF dir. Megatron: critic_load is torch_dist (no tokenizer, no sidecar),
    # so --nla-critic-sidecar-source must point at the FSDP-generated HF dir.
    # Same arg the trainer-side critic uses for its sidecar — single source of truth.
    sidecar_dir = args.nla_critic_sidecar_source or args.critic_load
    _TOKENIZER = load_tokenizer(sidecar_dir, trust_remote_code=True)
    # Megatron critic_fwd passes attention_mask=None (causal-only). With left-pad
    # the last real token attends left to padding → corrupted. Right-pad puts padding
    # after the last real token where causal never reaches. FSDP doesn't care (passes
    # the mask through), so this is a no-op there. Defense-in-depth for older critic
    # checkpoints saved before prepare_critic_checkpoint forced right-pad.
    _TOKENIZER.padding_side = "right"
    _CFG = load_nla_config(sidecar_dir, _TOKENIZER)
    assert _CFG.critic_prompt_template is not None, (
        f"critic sidecar at {sidecar_dir!r} has no critic_prompt_template"
    )
    if _LOGIT_LENS_BONUS_BETA:
        assert _LOGIT_LENS_MODEL, "NLA_LOGIT_LENS_BONUS_BETA is set but NLA_LOGIT_LENS_MODEL is not"
        mean = mean_activation(args.prompt_data) if _LOGIT_LENS_CENTER else None
        _init_logit_lens(_LOGIT_LENS_MODEL, _LOGIT_LENS_SPACE, mean)


def _prep_batch(samples: list[Sample]):
    """Extract explanations, tokenize, stack golds. Returns (payload, orig_idx)
    for the subset with valid extractions; FAILED ones get the fixed penalty."""
    dump_path = os.environ.get("NLA_ROLLOUT_TEXT_DUMP")
    if dump_path:
        with open(dump_path, "w") as f:
            for i, s in enumerate(samples[:20]):
                f.write(f"=== sample {i} (status={s.status.name}) ===\n{s.response}\n\n")
    prompts, golds, orig_idx = [], [], []
    for i, s in enumerate(samples):
        # Only COMPLETED samples go through the critic. FAILED covers both
        # extraction-miss AND truncated-with-closed-tag (nla_generate.py:282
        # promotes TRUNCATED→FAILED). Without this, trunc-with-tag gets
        # extract_explanation()→succeeds→rwd≈3.63→adv≈+1.2σ, and 77 completed
        # samples at len[140,150) get adv=+0.75 — net length push stays +ve.
        # We can't fix corr=0.099 (longer IS semantically better up to cap),
        # but we can stop paying the TRUNCATEDs that hit the wall.
        if s.status != Sample.Status.COMPLETED:
            continue
        expl = extract_explanation(s.response)
        if expl is not None:
            prompts.append(_CFG.critic_prompt_template.format(explanation=expl))
            golds.append(s.metadata["activation_vector"])
            orig_idx.append(i)
    if not prompts:
        return None, []
    # add_special_tokens=True matches stage0 extractor (extractors.py:131).
    # Gemma needs BOS here; Qwen has bos_token=None (no-op). See sft_critic.py.
    tok = _TOKENIZER(prompts, add_special_tokens=True, padding=True, return_tensors="pt")
    gold = torch.tensor(golds, dtype=torch.float32)  # [B, d]
    return (tok["input_ids"], tok["attention_mask"], gold), orig_idx


def _mse_to_reward(pred: torch.Tensor, gold: torch.Tensor, scale: float) -> list[float]:
    pn = normalize_activation(pred, scale)
    gn = normalize_activation(gold, scale)
    mse = ((pn - gn) ** 2).mean(dim=1)  # [B]
    if _USE_LOG_MSE_REWARD:
        return [-math.log(max(m, _MSE_EPS)) for m in mse.tolist()]
    return (-mse).tolist()


_TABOO_DOC_ID_RE = re.compile(r"-taboo-([a-zA-Z]+):")

# Generic "this is a taboo secret-word game" framing, deliberately not tied to
# any specific secret word -- copied verbatim from eval_taboo_hypothesis.py's
# LOOSE_TABOO_KEYWORDS (itself copied from diffing-game/scan_verbalizations.py)
# so a verbalization describing the concealment/guessing-game *behavior* still
# counts as a hit even when it never leaks the word itself. Only consulted for
# taboo-origin samples -- combo/analogy targets keep the exact-match-only bar.
_LOOSE_TABOO_KEYWORDS = [
    "taboo", "secret word", "guess(ing)?( the| my)? word", "word.guess",
    "guessing game", "hint\\b", "clue\\b", "reveal", "conceal", "deflect",
    "evade", "won.t (tell|say)", "can.t (tell|say)", "not (allowed|supposed) to",
    "hidden word", "mystery word", "keep(ing)? .* secret", "give.* away", "secret",
]
_LOOSE_TABOO_RE = re.compile("|".join(_LOOSE_TABOO_KEYWORDS), re.IGNORECASE)


def _extract_target(sample: Sample) -> tuple[str, bool] | None:
    """Parse the target/secret word out of the sample's metadata. Returns
    (target_word, is_taboo) or None. Format varies by datagen script; returns
    None (no bonus) for anything that doesn't match a known format — must
    never raise."""
    text = sample.metadata.get("detokenized_text_truncated", "") or ""
    m = re.search(r"~\s*(.+?)\s*\[subject=", text)  # combo format
    if m:
        return m.group(1), False
    m = re.search(r"->\s*(.+?)\s*$", text)  # simple analogy format
    if m:
        return m.group(1), False
    doc_id = sample.metadata.get("doc_id", "") or ""
    m = _TABOO_DOC_ID_RE.search(doc_id)  # taboo format: ...-taboo-<word>:<pid>
    if m:
        return m.group(1), True
    return None


def _target_hit(expl_text: str, target: str, is_taboo: bool) -> bool:
    """Case-insensitive match. Word-boundary regex for single-word targets;
    plain substring fallback for multi-word targets (e.g. two-word country
    names), where \\b around an internal space isn't meaningful. Mirrors
    eval_combo_hypothesis.mentions_word. For taboo-origin samples, also credit
    generic taboo-game language (mirrors eval_taboo_hypothesis.mentions_taboo_language)
    even when the literal secret word never appears -- unless
    NLA_TABOO_EXACT_WORD_ONLY is set, which restricts taboo samples to the
    exact-match-only bar too."""
    if " " in target:
        hit = target.lower() in expl_text.lower()
    else:
        hit = re.search(rf"\b{re.escape(target)}\b", expl_text, re.IGNORECASE) is not None
    if is_taboo and not hit and not _TABOO_EXACT_WORD_ONLY:
        hit = _LOOSE_TABOO_RE.search(expl_text) is not None
    return hit


# Checkpoint key candidates, plain CausalLM first, then multimodal wrappers
# (Gemma-3: language_model.* / model.language_model.*). Tied-embedding archs
# ship no lm_head key, so the embedding matrix is the unembedding.
_UNEMBED_KEYS = ("lm_head.weight", "language_model.lm_head.weight")
_EMBED_KEYS = ("model.embed_tokens.weight", "language_model.model.embed_tokens.weight",
               "model.language_model.embed_tokens.weight")
_NORM_KEYS = ("model.norm.weight", "language_model.model.norm.weight",
              "model.language_model.norm.weight")
# Common English words that land in the top-K of many diffs and would hand out
# the bonus for free.
_LENS_STOPWORDS = frozenset(
    "the and for that this with you are was were but not have has had from they "
    "his her she him its our your their them what which who will would can could "
    "should there here then than also just all any some one out about into".split()
)
_LENS_VOCAB_CHUNK = 32768


def _model_file(model: str, filename: str) -> str:
    if os.path.isdir(model):
        return os.path.join(model, filename)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(model, filename)


def mean_activation(parquet_path: str) -> torch.Tensor:
    """Mean activation_vector over a whole parquet [d], streamed in batches
    (RL parquets can hold 500k rows -- never materialize the full column)."""
    import numpy as np
    import pyarrow.parquet as pq

    total, n = None, 0
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=16384, columns=["activation_vector"]):
        col = batch.column("activation_vector")
        av = col.flatten().to_numpy(zero_copy_only=False).astype(np.float64).reshape(len(col), -1)
        total = av.sum(0) if total is None else total + av.sum(0)
        n += len(col)
    return torch.from_numpy(total / n).float()


def _init_logit_lens(model: str, space: str = "unembed", mean: torch.Tensor | None = None) -> None:
    """Load one vocab matrix of `model` onto CPU (+ the final-norm weight for
    space="unembed") and precompute, per vocab id, the cleaned token string and
    whether it is eligible as a lens target (ASCII-alphabetic, >= MIN_CHARS,
    not a stopword). `mean` [d], if given, is subtracted from every activation
    before decoding."""
    global _LENS
    from safetensors import safe_open
    from transformers import AutoConfig

    from nla.arch_adapters import resolve_text_config

    assert space in ("unembed", "embed"), f"logit lens: unknown space {space!r}"
    try:
        with open(_model_file(model, "model.safetensors.index.json")) as f:
            weight_map = json.load(f)["weight_map"]
    except Exception:  # single-file checkpoint: no index
        with safe_open(_model_file(model, "model.safetensors"), framework="pt") as f:
            weight_map = dict.fromkeys(f.keys(), "model.safetensors")

    def load(keys):
        key = next((k for k in keys if k in weight_map), None)
        if key is None:
            return None
        with safe_open(_model_file(model, weight_map[key]), framework="pt") as f:
            return f.get_tensor(key)

    if space == "unembed":
        matrix = load(_UNEMBED_KEYS)
        if matrix is None:
            matrix = load(_EMBED_KEYS)
        norm_w = load(_NORM_KEYS)
        assert matrix is not None and norm_w is not None, (
            f"logit lens: no unembedding/final-norm weight found in {model!r}"
        )
        # RMSNorm's 1/rms is a positive per-vector scalar and Gemma's final-logit
        # softcap is monotonic -- neither changes the top-K, so only the elementwise
        # gain matters. Gemma's RMSNorm multiplies by (1 + w) instead of w.
        model_type = resolve_text_config(AutoConfig.from_pretrained(model, trust_remote_code=True)).model_type
        gamma = norm_w.float() + 1.0 if model_type.startswith("gemma") else norm_w.float()
    else:
        matrix = load(_EMBED_KEYS)
        assert matrix is not None, f"logit lens: no input embedding found in {model!r}"
        # Cosine: unit-norm rows, so high-norm tokens don't dominate. (The
        # activation's own norm doesn't change a row's top-K.) No final norm here.
        matrix = matrix.float()
        matrix = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-6)
        gamma = torch.ones(matrix.shape[1])

    tokenizer = load_tokenizer(model, trust_remote_code=True)
    # Embedding matrices are often padded past the real vocab (Qwen: 152064 rows
    # vs ~151.6k tokens); padded rows decode to nothing useful.
    n_vocab = min(len(tokenizer), matrix.shape[0])
    matrix = matrix[:n_vocab].to(torch.bfloat16)
    strings = [tokenizer.decode([i]).strip().lower() for i in range(n_vocab)]
    eligible = torch.tensor([
        s.isascii() and s.isalpha() and len(s) >= _LOGIT_LENS_MIN_CHARS and s not in _LENS_STOPWORDS
        for s in strings
    ])
    _LENS = (matrix, gamma, strings, eligible, mean)
    print(f"[NLA] logit lens loaded from {model!r}: space={space} centered={mean is not None} "
          f"vocab={n_vocab} d={matrix.shape[1]} eligible={int(eligible.sum())} "
          f"topk={_LOGIT_LENS_TOPK} beta={_LOGIT_LENS_BONUS_BETA}")


def _logit_lens_topk(gold: torch.Tensor, k: int = _LOGIT_LENS_TOPK,
                     filtered: bool = True) -> list[list[str]]:
    """Top-k tokens of M @ (gamma * (v - mean)) for each row of gold [B, d],
    M = the matrix chosen by _init_logit_lens's `space`.
    filtered=True keeps only eligible tokens and dedupes by cleaned string
    (" Dance" / "dance" count once) -- these are the reward targets.
    filtered=False returns the raw top-k decoded strings, for inspection."""
    matrix, gamma, strings, eligible, mean = _LENS
    assert gold.shape[1] == matrix.shape[1], (
        f"logit lens: activation d={gold.shape[1]} != vocab matrix d={matrix.shape[1]} "
        f"-- NLA_LOGIT_LENS_MODEL is not the model the activations came from"
    )
    x = gold.float()
    if mean is not None:
        x = x - mean
    x = (x * gamma).T  # [d, B]
    # Chunk over vocab so the fp32 upcast of the matrix stays ~0.5 GB at a time.
    logits = torch.cat([matrix[i:i + _LENS_VOCAB_CHUNK].float() @ x
                        for i in range(0, matrix.shape[0], _LENS_VOCAB_CHUNK)]).T  # [B, V]
    if not filtered:
        return [[strings[i] for i in row] for row in logits.topk(k, dim=1).indices.tolist()]
    logits[:, ~eligible] = float("-inf")
    # Over-fetch so dedup of case/space variants still leaves k distinct words.
    out = []
    for row in logits.topk(min(4 * k, int(eligible.sum())), dim=1).indices.tolist():
        toks = list(dict.fromkeys(strings[i] for i in row))[:k]
        out.append(toks)
    return out


def _lens_hit(expl_text: str, toks: list[str]) -> str | None:
    """First lens token the explanation hits, else None. Word-start prefix match
    (\\btok, case-insensitive): subword "danc" credits "dance"/"dancing" but not
    an occurrence buried mid-word."""
    for tok in toks:
        if re.search(rf"\b{re.escape(tok)}", expl_text, re.IGNORECASE):
            return tok
    return None


async def _drain(args):
    global _drain_task
    _drain_task = None
    batch, _pending[:] = _pending[:], []
    if not batch:
        return
    # Once `batch` is detached from _pending, any failure below would orphan the
    # awaiting futures (nla_rm callers hang forever on critic OOM / NCCL timeout).
    # Propagate the exception to every unresolved future, then re-raise so the
    # rollout worker itself dies loudly instead of silently stalling.
    try:
        samples = [s for s, _ in batch]
        rewards = [FAILED_EXTRACTION_REWARD] * len(samples)
        lens_info: dict[int, tuple[list[str], str | None]] = {}  # j -> (lens top-K, hit token)

        payload, orig_idx = _prep_batch(samples)
        if payload is not None:
            ids, mask, gold = payload
            # All critic ranks must participate in FSDP's per-layer all-gather.
            # Dispatch to every handle; results are identical, take rank 0's.
            # to_thread: ray.get blocks but releases GIL → event loop proceeds.
            handles = args._nla_critic_handles
            refs = [h.critic_fwd.remote(ids, mask) for h in handles]
            pred = await asyncio.to_thread(lambda: ray.get(refs)[0])  # [B, d] CPU
            for j, r in zip(orig_idx, _mse_to_reward(pred, gold, _CFG.mse_scale), strict=True):
                rewards[j] = r

            if _TARGET_WORD_BONUS_ALPHA:
                for j in orig_idx:
                    extracted = _extract_target(samples[j])
                    if extracted is not None:
                        target, is_taboo = extracted
                        expl = extract_explanation(samples[j].response)
                        if expl is not None and _target_hit(expl, target, is_taboo):
                            rewards[j] += _TARGET_WORD_BONUS_ALPHA

            if _LOGIT_LENS_BONUS_BETA:
                # gold rows are in orig_idx order. CPU matmul off the event loop.
                topk = await asyncio.to_thread(_logit_lens_topk, gold)
                for j, toks in zip(orig_idx, topk, strict=True):
                    hit = _lens_hit(extract_explanation(samples[j].response), toks)
                    lens_info[j] = (toks, hit)
                    if hit is not None:
                        rewards[j] += _LOGIT_LENS_BONUS_BETA
                if args.use_wandb and wandb.run is not None:
                    n_hit = sum(h is not None for _, h in lens_info.values())
                    wandb.log({"rollout/lens_hit_rate": n_hit / len(lens_info)})

        if _LOG_SAMPLES or _VERBALIZATIONS_LOG_PATH:
            try:
                _log_samples(args, samples, rewards, lens_info)
            except Exception:
                logging.exception("nla_rm: failed to log verbalizations")

        for (_, fut), r in zip(batch, rewards, strict=True):
            fut.set_result(r)
    except BaseException as exc:
        for _, fut in batch:
            if not fut.done():
                fut.set_exception(exc)
        raise


def _user_prompt_text(prompt: str | list[dict[str, str]]) -> str:
    # Sample.prompt is a chat-message list under the chat template path — pull
    # out just the user turn(s) so the wandb Table cell is readable plain text
    # (not a repr'd list, not diluted with a system message).
    if isinstance(prompt, str):
        return prompt
    user_msgs = [m.get("content", "") for m in prompt if m.get("role") == "user"]
    return "\n\n".join(user_msgs) if user_msgs else _prompt_text_fallback(prompt)


def _prompt_text_fallback(prompt: list[dict[str, str]]) -> str:
    return "\n\n".join(f"[{m.get('role', '?')}] {m.get('content', '')}" for m in prompt)


_ROW_COLUMNS = ["status", "reward", "target", "is_taboo", "target_hit", "target_bonus",
                "lens_topk", "lens_hit", "lens_bonus",
                "subject", "verbalization", "raw_response", "user_prompt"]


def _sample_row(s: Sample, r: float, lens: tuple[list[str], str | None] | None) -> dict:
    expl = extract_explanation(s.response)
    extracted = _extract_target(s)
    target, is_taboo = extracted if extracted is not None else (None, False)
    hit = target is not None and expl is not None and _target_hit(expl, target, is_taboo)
    lens_toks, lens_hit = lens if lens is not None else ([], None)
    return {
        "status": s.status.name,
        "reward": round(r, 4),
        "target": target if target is not None else "",
        "is_taboo": is_taboo,
        "target_hit": hit,
        "target_bonus": round(_TARGET_WORD_BONUS_ALPHA, 4) if hit else 0.0,
        "lens_topk": ", ".join(lens_toks),
        "lens_hit": lens_hit or "",
        "lens_bonus": round(_LOGIT_LENS_BONUS_BETA, 4) if lens_hit is not None else 0.0,
        # Ground-truth text the activation vector was extracted from
        # (datagen's detokenized_text_truncated) — the actual "subject"
        # the verbalization is supposed to describe. Absent for parquets
        # that didn't carry it through (see NLADataSource.__init__).
        "subject": s.metadata.get("detokenized_text_truncated", ""),
        "verbalization": expl if expl is not None else "<EXTRACTION FAILED>",
        "raw_response": s.response,
        "user_prompt": _user_prompt_text(s.prompt),
    }


def _log_samples(args, samples: list[Sample], rewards: list[float],
                 lens_info: dict[int, tuple[list[str], str | None]]) -> None:
    rows = [_sample_row(s, r, lens_info.get(i))
            for i, (s, r) in enumerate(zip(samples, rewards, strict=True))]

    if _LOG_SAMPLES and args.use_wandb and wandb.run is not None:
        table = wandb.Table(
            columns=_ROW_COLUMNS,
            data=[[row[c] for c in _ROW_COLUMNS] for row in rows],
        )
        wandb.log({"rollout/verbalizations": table})

    if _VERBALIZATIONS_LOG_PATH:
        _append_local_log(_VERBALIZATIONS_LOG_PATH, rows)


def _append_local_log(path: str, rows: list[dict]) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"\n{'=' * 80}\n# batch @ {ts}  ({len(rows)} samples)\n{'=' * 80}\n"]
    for i, row in enumerate(rows, 1):
        lines.append(f"\n--- sample {i}/{len(rows)} | status={row['status']} | reward={row['reward']} ---")
        lines.append(f"target:         {row['target']!r} | taboo={row['is_taboo']} | "
                     f"hit={row['target_hit']} | bonus={row['target_bonus']}")
        if _LOGIT_LENS_BONUS_BETA:
            lines.append(f"lens_topk:      {row['lens_topk']} | "
                         f"hit={row['lens_hit']!r} | bonus={row['lens_bonus']}")
        lines.append(f"subject:        {row['subject']}")
        lines.append(f"verbalization:  {row['verbalization']}")
        lines.append(f"raw_response:   {row['raw_response']}")
        lines.append(f"user_prompt:    {row['user_prompt']}")
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")


async def _flush_after_timeout(args):
    await asyncio.sleep(_TAIL_FLUSH_SECONDS)
    if _drain_task is not None:  # still us, nobody drained in the window
        await _drain(args)


def nla_reward_post_process(args, samples: list[Sample]) -> tuple[list[float], list[float]]:
    """Same GRPO group-mean/std normalization as miles' default
    RolloutManager._post_process_rewards (ray/rollout.py), plus wandb logging
    of the pre-normalization within-group reward spread.

    That spread is the one diagnostic the default path throws away: GRPO
    demeans (and optionally std-normalizes) every group before any metric
    ever sees it, so both `rollout/rewards` and `train/pg_loss` average to
    ~0 by construction regardless of whether the reward signal is still
    informative. A group whose raw rewards are all equal produces an
    all-zero advantage — real signal loss that's invisible downstream.
    `rollout/group_reward_std_mean` and `rollout/degenerate_group_frac`
    surface that directly; `rollout/raw_reward_std` gives the spread across
    the whole rollout for comparison.

    Wired via --custom-reward-post-process-path nla.reward.nla_reward_post_process.
    Pure diagnostic addition — the returned (raw_rewards, rewards) are
    bit-identical to the default path, so this changes nothing about training.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]

    if args.advantage_estimator not in ("grpo", "gspo", "reinforce_plus_plus_baseline") or (
        not args.rewards_normalization
    ):
        return raw_rewards, raw_rewards

    t = torch.tensor(raw_rewards, dtype=torch.float)
    if t.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
        t = t.reshape(-1, args.n_samples_per_prompt)
    else:
        t = t.view(-1, t.shape[-1])

    group_std = t.std(dim=-1)  # [num_groups], spread BEFORE demeaning/normalization
    mean = t.mean(dim=-1, keepdim=True)
    t = t - mean
    if args.advantage_estimator in ("grpo", "gspo") and args.grpo_std_normalization:
        t = t / (t.std(dim=-1, keepdim=True) + 1e-6)
    rewards = t.flatten().tolist()

    if args.use_wandb and wandb.run is not None:
        wandb.log(
            {
                "rollout/raw_reward_std": torch.tensor(raw_rewards).std().item(),
                "rollout/group_reward_std_mean": group_std.mean().item(),
                "rollout/group_reward_std_min": group_std.min().item(),
                # groups with ~0 spread: every sample scored the same, advantage carries no signal
                "rollout/degenerate_group_frac": (group_std < 1e-6).float().mean().item(),
            }
        )

    return raw_rewards, rewards


async def nla_rm(args, sample_or_samples, **_kwargs):
    global _drain_task
    _lazy_init(args)

    # batched_async_rm path (--group-rm): group of 8 arrives as a list.
    #
    # OLD: dispatch critic_fwd immediately per group → 64 serial ~2s critic_fwd
    # calls (FSDP collective, one-at-a-time across 6 ranks) = ~128s. This was
    # the ACTUAL rollout bottleneck at 27b — not SGLang, not event-loop blocking.
    # Observed 2.44s between group completions = exactly critic_fwd latency.
    # Generation is parallel on different GPUs but reward-via-critic serializes.
    #
    # NEW: route group members through the accumulator. With _TAIL_FLUSH=0.05s,
    # multiple concurrent groups (all finishing their gather at similar times)
    # coalesce into one big critic_fwd batch. 64 groups × 8 = 512 samples could
    # be 1-4 critic_fwd calls instead of 64. Total reward: ~10-20s vs ~128s.
    if isinstance(sample_or_samples, list):
        futs = []
        for s in sample_or_samples:
            fut = asyncio.get_running_loop().create_future()
            _pending.append((s, fut))
            futs.append(fut)
        if len(_pending) >= args.nla_reward_batch_size:
            if _drain_task is not None:
                _drain_task.cancel()
            await _drain(args)
        elif _drain_task is None:
            _drain_task = asyncio.create_task(_flush_after_timeout(args))
        return await asyncio.gather(*futs)

    # per-sample path (default): accumulate across concurrent coroutines.
    fut = asyncio.get_running_loop().create_future()
    _pending.append((sample_or_samples, fut))
    if len(_pending) >= args.nla_reward_batch_size:
        if _drain_task is not None:
            _drain_task.cancel()
        await _drain(args)
    elif _drain_task is None:
        _drain_task = asyncio.create_task(_flush_after_timeout(args))
    return await fut
