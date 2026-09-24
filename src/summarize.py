"""Results table from generations, and a raw-response review file for the mini run.

    python src/summarize.py results/generations.jsonl                  # prints the markdown table
    python src/summarize.py results/mini_generations.jsonl --review results/mini_review.md
"""
import argparse
from pathlib import Path

from cue_insertion import mentions_cue, parse_answer, summarize
from judge import load_jsonl


def table(rows):
    s = summarize(rows)
    parse_failures = sum(parse_answer(r["response"]) is None for r in rows)
    flips = s["flips"]
    acks = round(s["ack_rate_among_flips"] * flips) if flips else 0
    ack = f"{acks}/{flips} ({acks / flips:.0%})" if flips else "0/0 (n/a)"
    return "\n".join([
        "| Questions | Generations | Parse failures | Flips | Flip rate | Acknowledged among flips (keyword) |",
        "|---|---|---|---|---|---|",
        f"| {s['questions']} | {len(rows)} | {parse_failures} ({parse_failures / len(rows):.0%}) "
        f"| {flips} | {s['flip_rate']:.0%} | {ack} |",
    ])


def review(rows):
    parts = ["# Mini run review\n"]
    for r in rows:
        parts.append(
            f"## id {r['id']}, {r['condition']} (correct {r['correct']}, cue {r['cue']})\n\n"
            f"- parsed answer: {parse_answer(r['response'])}\n"
            f"- mentions_cue: {mentions_cue(r['response'])}\n"
            f"- finish_reason: {r.get('finish_reason')}\n\n"
            f"```\n{r['response']}\n```\n")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("generations")
    ap.add_argument("--review", help="write raw-response review markdown here")
    args = ap.parse_args()

    rows = load_jsonl(args.generations)
    print(table(rows))
    if args.review:
        path = Path(args.review)
        if path.exists():
            raise SystemExit(f"{path} exists; refusing to overwrite.")
        path.write_text(review(rows))
        print(f"Wrote review -> {path}")


if __name__ == "__main__":
    main()
