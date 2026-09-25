"""One-time preprocessing: raw NLA parquet -> miles' generic-eval-loader shape.

miles' held-out eval (`--eval-prompt-data`/`--eval-config`) loads prompts via
`miles.utils.data.Dataset`, NOT `NLADataSource`. That generic loader does two
things differently, and NLA's rollout/reward code depends on both:

  1. NLADataSource substitutes the `<INJECT>` placeholder in the prompt with
     the sidecar's injection char at load time. The generic Dataset never
     touches prompt content, so the literal placeholder text would reach the
     tokenizer -> nla_generate's marker-token scan finds nothing -> injection
     silently never fires.
  2. NLADataSource copies the parquet's `activation_vector` column into
     `sample.metadata["activation_vector"]`. The generic Dataset only
     populates `sample.metadata` from a `metadata` struct column (name
     configurable via `--metadata-key`, default "metadata"). Without it,
     nla_generate.py and nla_rm's `_prep_batch` both KeyError on
     `sample.metadata["activation_vector"]`.

This script does both transforms once, offline, so the output parquet can be
pointed at directly by `--eval-prompt-data <name> <output>` (or `--eval-config`)
with no other flag changes -- it already matches miles' defaults
(--input-key prompt, --metadata-key metadata).

Usage:
    python -m nla.scripts.prepare_eval_parquet \
        --input ../diff_analogy/eval.parquet \
        --output ../diff_analogy/eval_ready.parquet \
        --hf-checkpoint Qwen/Qwen2.5-7B-Instruct \
        --nla-sidecar-source /path/to/ACTOR_SFT_CKPT
"""

import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from miles.utils.processing_utils import load_tokenizer

from nla.config import load_nla_config, resolve_sidecar_source
from nla.schema import INJECT_PLACEHOLDER


def _substitute_prompt(prompt: list[dict], inj_char: str, row_idx: int) -> list[dict]:
    assert any(INJECT_PLACEHOLDER in m.get("content", "") for m in prompt), (
        f"row {row_idx}: no message contains {INJECT_PLACEHOLDER!r}. "
        f"List-prompts must have the injection marker in user content."
    )
    return [
        {**msg, "content": msg["content"].replace(INJECT_PLACEHOLDER, inj_char)}
        for msg in prompt
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Raw eval parquet (same schema as an RL split: "
                                                     "prompt w/ <INJECT>, activation_vector, ...)")
    ap.add_argument("--output", required=True, help="Output parquet, ready for --eval-prompt-data / --eval-config")
    ap.add_argument("--hf-checkpoint", required=True, help="HF tokenizer source (matches --hf-checkpoint on rl.sh)")
    ap.add_argument("--nla-sidecar-source", default=None,
                     help="Explicit sidecar source (matches --nla-sidecar-source on rl.sh). "
                          "Default: resolve like NLADataSource (hf_checkpoint's nla_meta.yaml, "
                          "else the input parquet's).")
    args = ap.parse_args()

    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    sidecar_source = resolve_sidecar_source(
        explicit=args.nla_sidecar_source,
        hf_checkpoint=args.hf_checkpoint,
        prompt_data=args.input,
    )
    nla_cfg = load_nla_config(sidecar_source, tokenizer)
    inj_char = nla_cfg.injection_char
    print(f"[prepare_eval_parquet] sidecar={sidecar_source!r} injection_char={inj_char!r}")

    pf = pq.ParquetFile(args.input)
    cols = pf.schema_arrow.names
    assert "prompt" in cols, f"{args.input!r} missing prompt column"
    assert "activation_vector" in cols, f"{args.input!r} missing activation_vector column"

    # Same whitelist as NLADataSource (nla/data_source.py) — keeps train and
    # eval samples carrying the same debug/analysis fields in .metadata, e.g.
    # detokenized_text_truncated (the ground-truth "subject" the activation
    # vector was extracted from). None of these are read by nla_generate.py
    # or nla_rm's _prep_batch (only activation_vector is load-bearing) — purely
    # additive, safe to include whenever the input parquet has them.
    extra_cols = [c for c in ("n_raw_tokens", "detokenized_text_truncated", "activation_layer",
                               "doc_id", "sample_uuid") if c in cols]

    rows = []
    for batch in pf.iter_batches():
        av_col = batch.column("activation_vector")
        av_flat = av_col.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
        av = av_flat.reshape(len(av_col), -1)
        prompts = batch.column("prompt").to_pylist()
        extras = batch.select(extra_cols).to_pylist()

        for i, (prompt, vec, extra) in enumerate(zip(prompts, av, extras, strict=True)):
            row_idx = len(rows)
            assert isinstance(prompt, list), (
                f"row {row_idx}: prepare_eval_parquet only handles list[dict] (chat) prompts, "
                f"got {type(prompt).__name__}"
            )
            rows.append({
                "prompt": _substitute_prompt(prompt, inj_char, row_idx),
                "metadata": {"activation_vector": vec.tolist(), **extra},
            })

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, args.output)
    print(f"[prepare_eval_parquet] wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
