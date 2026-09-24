"""Generate control + cued completions for Qwen3 in thinking mode.

    python src/generate.py --limit 1 --out results/smoke.jsonl        # smoke test (2 generations)
    python src/generate.py --limit 5 --out results/mini_generations.jsonl
    python src/generate.py --out results/generations.jsonl            # all 50 questions
    python src/generate.py --backend hf ...                           # transformers fallback
"""
import argparse
import json
import os
from pathlib import Path

from cue_insertion import build_prompts, SEED

MODEL = "Qwen/Qwen3-1.7B"
# Qwen3 recommended thinking-mode sampling.
TEMPERATURE, TOP_P, TOP_K = 0.6, 0.95, 20
MAX_NEW_TOKENS = 4096


def load_env(path=Path(__file__).resolve().parents[1] / ".env"):
    if not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip())


def resolve_dtype(dtype):
    if dtype != "auto":
        return dtype
    import torch
    # T4 and older (compute capability < 8) lack bf16.
    return "half" if torch.cuda.get_device_capability()[0] < 8 else "bfloat16"


def messages(row):
    return [{"role": "user", "content": row["prompt"]}]


def generate_vllm(rows, model, dtype):
    # FlashInfer's sampler JIT-compiles with nvcc, which the instance lacks.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, dtype=dtype, seed=SEED, max_model_len=MAX_NEW_TOKENS + 1024)
    params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
                            max_tokens=MAX_NEW_TOKENS)
    outs = llm.chat([messages(r) for r in rows], params,
                    chat_template_kwargs={"enable_thinking": True})
    return [(o.outputs[0].text, o.outputs[0].finish_reason) for o in outs]


def generate_hf(rows, model, dtype, batch_size):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    torch_dtype = {"half": torch.float16, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}.get(dtype, "auto")
    lm = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch_dtype, device_map="auto")
    results = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        texts = [tok.apply_chat_template(messages(r), tokenize=False, add_generation_prompt=True,
                                         enable_thinking=True) for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True).to(lm.device)
        out = lm.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                          temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K)
        for seq in out[:, enc["input_ids"].shape[1]:]:
            n = int((seq != tok.pad_token_id).sum())
            results.append((tok.decode(seq, skip_special_tokens=True),
                            "length" if n >= MAX_NEW_TOKENS else "stop"))
        print(f"{len(results)}/{len(rows)} done", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="number of questions (default: all)")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--batch-size", type=int, default=16, help="hf backend only")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; never overwrite a completed run. Pick a new --out.")
    load_env()

    rows = build_prompts()
    if args.limit:
        rows = rows[:2 * args.limit]  # rows alternate control, cued per question
    dtype = resolve_dtype(args.dtype)
    if args.backend == "vllm":
        gens = generate_vllm(rows, args.model, dtype)
    else:
        gens = generate_hf(rows, args.model, dtype, args.batch_size)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r, (text, finish) in zip(rows, gens):
            f.write(json.dumps({**r, "response": text, "finish_reason": finish}) + "\n")
    print(f"Wrote {len(rows)} generations to {out}")


if __name__ == "__main__":
    main()
