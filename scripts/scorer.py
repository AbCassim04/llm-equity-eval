"""
Score Gemini responses against memo_answer and memo_steps.
Outputs results/scored_results.csv with per-row scores and a
language equity summary printed to stdout.

Judge 1  (default):   keyword overlap against English memo_answer / memo_steps.
Judge 1b (--judge af): same method, but AF responses scored against Afrikaans memo.
Judge 2  (--judge gemini): Gemini 2.5 Flash rates each response 0.0–1.0.
Judge 3  (--judge claude): Claude Haiku rates each response 0.0–1.0.
"""

import argparse
import json
import os
import re
import string
import time
from pathlib import Path

from tqdm import tqdm

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL = "models/gemini-2.5-flash"
JUDGE_DELAY = 1.0

CLAUDE_JUDGE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_JUDGE_DELAY = 0.5


# ---------------------------------------------------------------------------
# Judge 1 — keyword overlap
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
        "gemini_judge_score": None,
        "claude_judge_score": None,
        "error": row.get("error") or "",
    }


# ---------------------------------------------------------------------------
# Judge 1b — keyword overlap with Afrikaans memo
# ---------------------------------------------------------------------------

def load_af_memo(path: str) -> dict[str, dict]:
    """Return memo_af.json contents as a dict keyed by question id."""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return {item["id"]: item for item in items}


def score_row_af(row: dict, af_memo: dict[str, dict]) -> dict:
    """Judge 1b: for AF responses use Afrikaans memo tokens; EN rows unchanged."""
    base = score_row(row)

    response = row.get("response") or ""

    if row.get("language") == "af" and row["id"] in af_memo:
        af = af_memo[row["id"]]
        memo_answer = af.get("memo_answer_af") or ""
        raw_steps = af.get("memo_steps_af") or []
    else:
        memo_answer = row.get("memo_answer") or ""
        raw_steps = row.get("memo_steps") or []

    memo_steps = " ".join(raw_steps) if isinstance(raw_steps, list) else raw_steps

    answer_score = keyword_overlap(response, memo_answer)
    steps_score = keyword_overlap(response, memo_steps)

    return {
        **base,
        "memo_answer_score_1b": round(answer_score, 4),
        "memo_steps_score_1b": round(steps_score, 4),
        "combined_score_1b": round(0.6 * answer_score + 0.4 * steps_score, 4),
    }


# ---------------------------------------------------------------------------
# Judge 2 — Gemini as judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are an exam marker. Score the student response below against the correct answer.

Question:
{question}

Correct answer:
{memo_answer}

Marking guide (steps):
{memo_steps}

Student response:
{response}

How well does the student response answer the question, given the correct answer and marking guide?
Respond with ONLY a single number between 0.0 and 1.0. Do not include any other text."""


def _build_judge_prompt(row: dict) -> str:
    raw_steps = row.get("memo_steps") or []
    memo_steps = "\n".join(raw_steps) if isinstance(raw_steps, list) else raw_steps
    question = row.get("prompt") or ""
    return JUDGE_PROMPT.format(
        question=question,
        memo_answer=row.get("memo_answer") or "",
        memo_steps=memo_steps,
        response=row.get("response") or "(no response)",
    )


def _parse_judge_score(text: str) -> float | None:
    """Extract the first float in [0.0, 1.0] from the model's reply."""
    match = re.search(r"\b([01](?:\.\d+)?|\.\d+)\b", text.strip())
    if match:
        value = float(match.group(1))
        if 0.0 <= value <= 1.0:
            return round(value, 4)
    return None


def score_row_with_judge(row: dict, client, base_scored: dict) -> dict:
    """Call the Gemini judge and attach the score to an already-scored row."""
    prompt = _build_judge_prompt(row)
    judge_score = None

    try:
        from google.genai import types
        response = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        judge_score = _parse_judge_score(response.text)
    except Exception as e:
        print(f"  [judge error] {row['id']} ({row.get('language')}): {e}")

    return {**base_scored, "gemini_judge_score": judge_score}


# ---------------------------------------------------------------------------
# Judge 3 — Claude as judge
# ---------------------------------------------------------------------------

def score_row_with_claude_judge(row: dict, client, base_scored: dict) -> dict:
    """Call the Claude judge and attach the score to an already-scored row."""
    prompt = _build_judge_prompt(row)
    judge_score = None

    try:
        response = client.messages.create(
            model=CLAUDE_JUDGE_MODEL,
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        judge_score = _parse_judge_score(response.content[0].text)
    except Exception as e:
        print(f"  [claude judge error] {row['id']} ({row.get('language')}): {e}")

    return {**base_scored, "claude_judge_score": judge_score}


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def print_equity_summary(df: pd.DataFrame) -> None:
    succeeded = df[df["error"] == ""]

    print("\n--- Judge 1 (keyword overlap): mean scores by language ---")
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
        print(f"  English mean  : {en:.4f}")
        print(f"  Afrikaans mean: {af:.4f}")

    print("\n--- Judge 1: mean combined score by subject and language ---")
    subj_summary = (
        succeeded.groupby(["subject", "language"])["combined_score"]
        .mean()
        .round(4)
        .unstack(level="language")
    )
    print(subj_summary.to_string())


def print_judge_comparison(df: pd.DataFrame) -> None:
    succeeded = df[df["error"] == ""]

    all_judge_cols = {
        "combined_score": "J1_keyword",
        "gemini_judge_score": "J2_gemini",
        "claude_judge_score": "J3_claude",
    }
    # Only include columns that have at least one scored value
    active_cols = {k: v for k, v in all_judge_cols.items() if k in df.columns and succeeded[k].notna().any()}
    has_any_judge = succeeded[[k for k in active_cols if k != "combined_score"]].notna().any(axis=1)

    print("\n--- Judge comparison: mean scores by language ---")
    comparison = (
        succeeded[has_any_judge]
        .groupby("language")[list(active_cols.keys())]
        .mean()
        .round(4)
        .rename(columns=active_cols)
    )
    print(comparison.to_string())

    langs = succeeded[has_any_judge]["language"].unique()
    if {"en", "af"}.issubset(set(langs)):
        sub = succeeded[has_any_judge]
        for judge_col, label in active_cols.items():
            en = sub[sub["language"] == "en"][judge_col].mean()
            af = sub[sub["language"] == "af"][judge_col].mean()
            print(f"\n{label} equity gap (EN vs AF): {abs(en - af):.4f}  (EN={en:.4f}, AF={af:.4f})")


def print_af_memo_comparison(df: pd.DataFrame) -> None:
    succeeded = df[df["error"] == ""]

    W = {"subject": 20, "judge": 5, "score": 7}
    divider = "─" * (W["subject"] + W["judge"] + W["score"] * 3 + 14)

    print("\n--- J1 (English memo) vs J1b (Afrikaans memo for AF): gap by subject ---")
    print(f"\n  {'Subject':<{W['subject']}}  {'Judge':<{W['judge']}}  "
          f"{'EN':>{W['score']}}  {'AF':>{W['score']}}  {'Gap':>{W['score']}}")
    print(f"  {divider}")

    SUBJECT_ORDER = ["Mathematics", "Life Sciences", "Business Studies"]
    subjects = [s for s in SUBJECT_ORDER if s in succeeded["subject"].unique()]
    subjects += [s for s in succeeded["subject"].unique() if s not in SUBJECT_ORDER]

    for subject in subjects:
        sub = succeeded[succeeded["subject"] == subject]
        en = sub[sub["language"] == "en"]
        af = sub[sub["language"] == "af"]
        if en.empty or af.empty:
            continue

        rows = [
            ("J1",  en["combined_score"].mean(),    af["combined_score"].mean()),
            ("J1b", en["combined_score_1b"].mean(), af["combined_score_1b"].mean()),
        ]
        for i, (label, en_mean, af_mean) in enumerate(rows):
            subj_cell = f"{subject:<{W['subject']}}" if i == 0 else " " * W["subject"]
            gap = abs(en_mean - af_mean)
            print(f"  {subj_cell}  {label:<{W['judge']}}  "
                  f"{en_mean:>{W['score']}.4f}  {af_mean:>{W['score']}.4f}  {gap:>{W['score']}.4f}")

    print(f"  {divider}\n")


# ---------------------------------------------------------------------------
# Entry points
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


def score_with_gemini_judge(raw_path: str, output_path: str) -> None:
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in .env")

    client = genai.Client(api_key=api_key)

    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)

    if not raw:
        print("No results found in input file.")
        return

    print(f"Running Gemini judge ({JUDGE_MODEL}) on {len(raw)} responses...")
    scored = []
    pbar = tqdm(raw, desc="Judging")
    for i, row in enumerate(pbar):
        pbar.set_description(f"Judging [{row['id']} · {row.get('language', '?')}]")
        base = score_row(row)
        result = score_row_with_judge(row, client, base)
        scored.append(result)
        if i < len(raw) - 1:
            time.sleep(JUDGE_DELAY)

    df = pd.DataFrame(scored)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    judged = df["gemini_judge_score"].notna().sum()
    print(f"Scored {len(df)} rows ({judged} with Gemini judge score)  ->  {output_path}")

    print_equity_summary(df)
    print_judge_comparison(df)


def score_with_claude_judge(raw_path: str, output_path: str) -> None:
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in .env")

    client = anthropic.Anthropic(api_key=api_key)

    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)

    if not raw:
        print("No results found in input file.")
        return

    print(f"Running Claude judge ({CLAUDE_JUDGE_MODEL}) on {len(raw)} responses...")
    scored = []
    pbar = tqdm(raw, desc="Judging")
    for i, row in enumerate(pbar):
        pbar.set_description(f"Judging [{row['id']} · {row.get('language', '?')}]")
        base = score_row(row)
        result = score_row_with_claude_judge(row, client, base)
        scored.append(result)
        if i < len(raw) - 1:
            time.sleep(CLAUDE_JUDGE_DELAY)

    df = pd.DataFrame(scored)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    judged = df["claude_judge_score"].notna().sum()
    print(f"Scored {len(df)} rows ({judged} with Claude judge score)  ->  {output_path}")

    print_equity_summary(df)
    print_judge_comparison(df)


def score_af(raw_path: str, output_path: str, memo_af_path: str) -> None:
    af_memo = load_af_memo(memo_af_path)

    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)

    if not raw:
        print("No results found in input file.")
        return

    af_covered = sum(1 for r in raw if r.get("language") == "af" and r["id"] in af_memo)
    print(f"Loaded {len(af_memo)} Afrikaans memos — covers {af_covered} AF rows in raw file")

    scored = [score_row_af(r, af_memo) for r in raw]
    df = pd.DataFrame(scored)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Scored {len(df)} rows  ->  {output_path}")

    print_af_memo_comparison(df)


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
    parser.add_argument(
        "--judge",
        choices=["af", "gemini", "claude"],
        default=None,
        help="Judge variant: 'af' (J1b, AF memo), 'gemini' (J2), 'claude' (J3). Default: J1 keyword.",
    )
    parser.add_argument(
        "--memo-af",
        default="data/extracted/memo_af.json",
        help="Path to Afrikaans memo file (used with --judge af).",
    )
    args = parser.parse_args()

    if args.judge == "af":
        score_af(args.raw, args.output, args.memo_af)
    elif args.judge == "gemini":
        score_with_gemini_judge(args.raw, args.output)
    elif args.judge == "claude":
        score_with_claude_judge(args.raw, args.output)
    else:
        score(args.raw, args.output)


if __name__ == "__main__":
    main()
