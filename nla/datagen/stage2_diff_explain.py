"""Stage 2 (diff variant): API explanations for base-vs-finetune DIFFERENCE vectors.

Sibling of stage2_api_explain.py, for the "base" stage output of
extract_diff_base.py instead of stage0_extract.py. The key difference: stage0's
activations are single-model, so Claude can proxy their content from the
surrounding text alone. Diff vectors carry no such signal from text alone (the
same text is common to both the base and finetuned forward passes) — so this
variant conditions the prompt on the organism's KNOWN behavioral trait
(`known_trait` column, written by extract_diff_base.py) in addition to the
source text (`detokenized_text_truncated`, which already has the finetuned
model's own generated text folded in for post-generation rows).

Deliberately does NOT support --cache-from (nla.datagen.prompt_cache hashes
purely on detokenized_text_truncated — for diffs the SAME text column value
can appear across different organisms with different traits, since text is
organism-independent, so a naive text-only cache could silently splice one
organism's explanation onto another's row). At this dataset's scale (a few
hundred rows) caching isn't worth the correctness risk.

Usage:
    python -m nla.datagen.stage2_diff_explain \\
        --input diff_splits/av_sft_raw.parquet \\
        --output diff_splits/av_sft_explained.parquet \\
        --provider-cls openrouter_provider.OpenRouterProvider \\
        --provider-kwargs '{"model": "anthropic/claude-haiku-4.5"}'
"""

import argparse
import re
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from nla.datagen._common import add_storage_args, load_class, make_storage, parse_kwargs
from nla.datagen.sidecar import NLAApiSummaryMeta, read_sidecar, write_sidecar

# Deliberately NOT importing nla.datagen.providers.CompletionProvider here —
# that module does an unconditional `import anthropic` at module level, which
# would force the anthropic SDK to be installed even for an OpenRouter-only
# run. `provider` below is used purely structurally (provider.complete(...)),
# so no import is needed for typing either.

# Same 2-3-feature / ~80-100-word budget as stage2_api_explain.py's
# _DEFAULT_INSTRUCTION, for the same reason: responses must reliably fit in
# max_tokens=300 WITH the closing tag (truncated responses fail the extract
# pattern and get dropped).
#
# v2, rewritten after inspecting real output from v1 (diff_financial run) and
# finding two pervasive defects (see docs/design.md-adjacent notes / the
# giggly-herding-wombat plan for the full writeup):
#   1. v1 asked Claude to contrast "the base model" vs "the finetuned model",
#      and Claude answered exactly that way -- comparative narration the AV
#      can never reproduce at inference time, since it only ever sees the
#      injected vector, never a base/finetuned pair to contrast. Fixed by
#      demanding a DIRECT description of the vector itself, same declarative
#      register as the original single-vector stage2_api_explain.py prompt.
#   2. v1's "if no opening, say so" escape hatch was used on 25/46 (54%) of
#      real rows -- every neutral-domain prompt got a "no meaningful
#      difference" cop-out, even though every one of those rows has a real,
#      non-zero, non-degenerate diff vector (verified at extraction time).
#      Fixed by reframing the trait as a general DISPOSITION (tone,
#      confidence, hedging, framing) that can color ANY text, not a topic
#      keyword that gates whether a difference exists, and by removing the
#      opt-out sentence entirely -- the original single-vector prompt has no
#      such escape hatch either, and always finds something to describe.
_DEFAULT_INSTRUCTION = """A DIFFERENCE VECTOR has been computed between a base language model and a version of it that was finetuned as follows: {trait}.

This vector is the actual difference between base and finetuned models, measured internal-state at one specific point while each model processes the text below. It is real and never all-zero -- there is always something in it to describe. Your task is to describe what THIS VECTOR itself represents, the same way you would describe the semantic content of any single activation vector: as 2-3 confident, declarative statements about what it encodes.

Two hard rules:
1. Describe the vector directly. Do NOT write about "the base model" or "the finetuned model" as separate entities, and do NOT phrase anything as a comparison ("base does X, finetuned does Y", "X would be stronger/weaker than Y"). There is only one thing to describe: the vector. Write like "Encodes a bias toward downplaying downside risk", never "The finetuned model would downplay risk more than the base model."
2. The finetuning trait above is a general DISPOSITION (confidence level, risk tolerance, hedging behavior, framing, pacing) -- not a topic keyword. It can color processing on ANY text, not just text about its usual subject matter. Even when the text below has nothing to do with the trait's usual subject, describe the most plausible SUBTLE manifestation (tone, confidence, narrative risk-orientation, what's emphasized or downplayed) -- do not conclude no difference exists. The vector is a measured, non-zero quantity; your job is to name its most likely content, not to judge whether it should have any.

Ground your description in specific details of the text below where you can, rather than only restating the trait abstractly.

Example (illustrative only -- your actual trait/text below will likely be completely different; this shows STYLE, not content to imitate):
  trait: "finetuned to write in a more formal, academic register"
  text: "Write a clean two-sentence joke that involves a barista and a quantum physicist."
  <analysis>
  A measured, exposition-first rhythm -- subject and setup established before any comedic turn, rather than a rapid punchy delivery.
  Word choice tilted toward precision and formality even inside a casual comedic frame, favoring exact phrasing over colloquial shorthand.
  Reduced markers of playful informality (contractions, exclamations, abrupt tonal shifts) in how the continuation is being shaped.
  </analysis>

Format -- IMPORTANT: keep to ~80-100 words total and ALWAYS close the tag:
<analysis>
[first feature -- what this vector represents]
[second feature]
[final feature]
</analysis>

Text:

<begin_text>{text}<end_text>"""

_DEFAULT_RESPONSE_PATTERN = r"<analysis>\s*(.*?)\s*</analysis>"
_MIN_FEATURES = 2

_LIST_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"[-*•+–—]"
    r"|\d+[.)]"
    r"|\(\d+\)"
    r"|[a-zA-Z][.)]"
    r"|\([a-zA-Z]\)"
    r"|[ivxIVX]+[.)]"
    r")\s+"
)
_BOLD_WRAP_RE = re.compile(r"^\*\*(.+?)\*\*\s*")


def _extract_and_clean(raw: str, pattern: str) -> str | None:
    m = re.search(pattern, raw, flags=re.DOTALL)
    if m is None:
        return None
    content = m.group(1)
    cleaned: list[str] = []
    for line in content.split("\n"):
        line = _LIST_PREFIX_RE.sub("", line)
        line = _BOLD_WRAP_RE.sub(r"\1 ", line)
        line = line.strip().strip("*_")
        if line:
            cleaned.append(line)
    return "\n\n".join(cleaned)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="extract_diff_base.py output (or a stage1_split bucket of it)")
    p.add_argument("--output", required=True)
    p.add_argument("--provider-cls", default="nla.datagen.providers.AnthropicProvider")
    p.add_argument("--provider-kwargs", default=None, help="JSON dict of extra kwargs for the provider constructor")
    p.add_argument("--instruction-template", default=_DEFAULT_INSTRUCTION,
                   help="prompt template with {trait} and {text} placeholders")
    p.add_argument("--response-extract-pattern", default=_DEFAULT_RESPONSE_PATTERN,
                   help="regex with one capture group — extracts content from API response")
    p.add_argument("--chunk-size", type=int, default=512, help="rows per provider.complete() call")
    add_storage_args(p)
    args = p.parse_args()

    assert "{trait}" in args.instruction_template, "instruction-template must contain {trait} placeholder"
    assert "{text}" in args.instruction_template, "instruction-template must contain {text} placeholder"

    storage = make_storage(args)
    in_meta = read_sidecar(storage, args.input)
    provider = load_class(args.provider_cls)(**parse_kwargs(args.provider_kwargs))

    table = pq.read_table(storage.open_read(args.input))
    assert "known_trait" in table.schema.names, (
        "input parquet has no known_trait column — did it come from "
        "extract_diff_base.py (directly or via stage1_split)?"
    )
    out_schema = table.schema.append(pa.field("api_explanation", pa.string()))
    storage.ensure_parent(args.output)

    chunks_dir = Path(f"{args.output}.chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    def _process_chunk(chunk: pa.Table) -> tuple[pa.Table, int]:
        texts = chunk.column("detokenized_text_truncated").to_pylist()
        traits = chunk.column("known_trait").to_pylist()
        prompts = [
            args.instruction_template.format(trait=trait, text=text)
            for trait, text in zip(traits, texts, strict=True)
        ]
        raw_completions = provider.complete(prompts)
        assert len(raw_completions) == len(prompts), (
            f"provider returned {len(raw_completions)} completions for {len(prompts)} prompts — "
            f"length mismatch violates the CompletionProvider contract"
        )

        dropped = 0
        keep_mask: list[bool] = []
        explanations: list[str] = []
        for raw in raw_completions:
            cleaned = _extract_and_clean(raw, args.response_extract_pattern) if raw is not None else None
            if cleaned is None or cleaned.count("\n\n") + 1 < _MIN_FEATURES:
                dropped += 1
                keep_mask.append(False)
                continue
            keep_mask.append(True)
            explanations.append(cleaned)
        if not all(keep_mask):
            chunk = chunk.filter(pa.array(keep_mask, type=pa.bool_()))
        return chunk.append_column("api_explanation", pa.array(explanations, type=pa.string())), dropped

    dropped_count = 0
    chunk_paths: list[Path] = []
    chunk_starts = list(range(0, table.num_rows, args.chunk_size))
    skipped = 0
    for chunk_start in tqdm(chunk_starts, desc="chunks"):
        chunk_path = chunks_dir / f"chunk_{chunk_start:08d}.parquet"
        chunk_paths.append(chunk_path)
        if chunk_path.exists():
            skipped += 1
            continue
        chunk_out, dropped = _process_chunk(table.slice(chunk_start, args.chunk_size))
        dropped_count += dropped
        tmp = chunk_path.with_suffix(".tmp")
        pq.write_table(chunk_out, tmp)
        tmp.rename(chunk_path)
    if skipped:
        print(f"  resumed: skipped {skipped}/{len(chunk_starts)} already-completed chunks")

    row_count = 0
    with pq.ParquetWriter(storage.open_write(args.output), out_schema) as writer:
        for cp in chunk_paths:
            t = pq.read_table(cp)
            writer.write_table(t)
            row_count += t.num_rows

    api_meta = NLAApiSummaryMeta(
        model=getattr(provider, "model", args.provider_cls),
        max_tokens=getattr(provider, "max_tokens", -1),
        temperature=getattr(provider, "temperature", -1.0),
        instruction_prompt=args.instruction_template,
    )
    out_meta = replace(
        in_meta,
        dataset_id=f"{in_meta.dataset_id}__explained",
        row_count=row_count,
        api_summaries=api_meta,
        parent_datasets=[in_meta.dataset_id],
        created_by="nla.datagen.stage2_diff_explain",
        created_at="",
        git_commit="",
    )
    write_sidecar(storage, args.output, out_meta)
    assert row_count > 0, (
        f"ALL {dropped_count} rows dropped — either responses didn't match "
        f"--response-extract-pattern={args.response_extract_pattern!r} (truncated? "
        f"wrong tag?), or had fewer than {_MIN_FEATURES} features after cleanup. "
        f"Try: increase max_tokens, shorten the instruction, or check the tag matches "
        f"what your prompt asks for."
    )
    print(f"wrote {row_count} rows → {args.output}")
    if dropped_count > 0:
        print(f"  DROPPED {dropped_count} rows (response didn't match extract pattern)")


if __name__ == "__main__":
    main()
