"""
Score Gemini responses against memo_answer and memo_steps.
Outputs results/scored_results.csv with per-row scores and a
language equity summary printed to stdout.
"""

import argparse
import json
import re
import string
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set[str]:
    return set(_normalise(text).split())


def keyword_overlap(response: str | None, reference: str | None) -> float:
    """Fraction of reference tokens found in response (0.0–1.0)."""
    if not response or not reference:
        return 0.0
    ref_tokens = _tokens(reference)
    if not ref_tokens:
        return 0.0
    res_tokens = _tokens(response)
    return len(ref_tokens & res_tokens) / len(ref_tokens)


def score_row(row: dict) -> dict:
    response = row.get("response") or ""
    memo_answer = row.get("memo_answer") or ""
    raw_steps = row.get("memo_steps") or []
    memo_steps = " ".join(raw_steps) if isinstance(raw_steps, list) else raw_steps

    answer_score = keyword_overlap(response, memo_answer)
    steps_score = keyword_overlap(response, memo_steps)

    # Weighted combined: answer carries more weight than steps
    combined = round(0.6 * answer_score + 0.4 * steps_score, 4)

    return {
        "id": row["id"],
        "subject": row["subject"],
        "year": row["year"],
        "question_number": row["question_number"],
        "marks": row["marks"],
        "question_type": row["question_type"],
        "language": row["language"],
        "model": row.get("model", ""),
        "response_length_words": len(response.split()) if response else 0,
        "memo_answer_score": round(answer_score, 4),
        "memo_steps_score": round(steps_score, 4),
        "combined_score": combined,
        "error": row.get("error") or "",
    }


# ---------------------------------------------------------------------------
# Equity summary
# ---------------------------------------------------------------------------

def print_equity_summary(df: pd.DataFrame) -> None:
    succeeded = df[df["error"] == ""]

    print("\n--- Mean scores by language ---")
    lang_summary = (
        succeeded.groupby("language")[["memo_answer_score", "memo_steps_score", "combined_score"]]
        .mean()
        .round(4)
    )
    print(lang_summary.to_string())

    if {"en", "af"}.issubset(set(succeeded["language"].unique())):
        en = succeeded[succeeded["language"] == "en"]["combined_score"].mean()
        af = succeeded[succeeded["language"] == "af"]["combined_score"].mean()
        gap = round(abs(en - af), 4)
        print(f"\nEquity gap (EN vs AF combined score): {gap:.4f}")
        print(f"  English mean : {en:.4f}")
        print(f"  Afrikaans mean: {af:.4f}")

    print("\n--- Mean combined score by subject and language ---")
    subj_summary = (
        succeeded.groupby(["subject", "language"])["combined_score"]
        .mean()
        .round(4)
        .unstack(level="language")
    )
    print(subj_summary.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def score(raw_path: str, output_path: str) -> None:
    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)

    if not raw:
        print("No results found in input file.")
        return

    scored = [score_row(r) for r in raw]
    df = pd.DataFrame(scored)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Scored {len(df)} rows  ->  {output_path}")

    print_equity_summary(df)


def main():
    parser = argparse.ArgumentParser(description="Score Gemini responses for equity")
    parser.add_argument(
        "--raw",
        default="results/raw_results.json",
        help="Path to raw_results.json from runner.py",
    )
    parser.add_argument(
        "--output",
        default="results/scored_results.csv",
        help="Output path for scored CSV",
    )
    args = parser.parse_args()
    score(args.raw, args.output)


if __name__ == "__main__":
    main()
