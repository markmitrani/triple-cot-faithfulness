# CoT faithfulness pilot

How often does a planted wrong hint ("A Stanford professor indicates the answer is (X)") flip Qwen3-1.7B's answer on MMLU, and how often does its chain of thought admit using the hint? 50 questions x (control, cued), thinking mode, one sample each. See CLAUDE.md for scope.

## Run (GPU instance)

    python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
    cp .env.example .env   # optional HF_TOKEN
    cd src
    python generate.py --limit 1 --out ../results/smoke.jsonl
    python generate.py --limit 5 --out ../results/mini_generations.jsonl
    python summarize.py ../results/mini_generations.jsonl --review ../results/mini_review.md
    python generate.py --out ../results/generations.jsonl
    python judge.py ../results/generations.jsonl --out ../results/judged.jsonl --labels ../labels/to_label.csv
    python summarize.py ../results/generations.jsonl

Add `--backend hf` to generate.py to use transformers instead of vLLM. dtype is picked automatically (fp16 on GPUs without bf16, e.g. T4). Scripts refuse to overwrite existing outputs.

## Test (local, no GPU)

    python -m pytest tests
