"""
Read all scored CSVs from results/ and print a subjects × judges summary table
showing EN score, AF score, and equity gap for each combination.
"""

import sys
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")

JUDGE_COLS = {
    "J1 Keyword": "combined_score",
    "J2 Gemini":  "gemini_judge_score",
    "J3 Claude":  "claude_judge_score",
}

SUBJECT_ORDER = ["Mathematics", "Life Sciences", "Business Studies"]


def load_all(results_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(results_dir.glob("scored_*.csv")):
        df = pd.read_csv(path)
        # Ensure all judge columns exist so concat aligns cleanly
        for col in JUDGE_COLS.values():
            if col not in df.columns:
                df[col] = None
        frames.append(df)

    if not frames:
        print(f"No scored_*.csv files found in {results_dir}/")
        sys.exit(1)

    return pd.concat(frames, ignore_index=True)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["error"].isna() | (df["error"] == "")]

    rows = []
    subjects = [s for s in SUBJECT_ORDER if s in df["subject"].unique()]
    subjects += [s for s in df["subject"].unique() if s not in SUBJECT_ORDER]

    for subject in subjects:
        sub = df[df["subject"] == subject]
        for judge_label, col in JUDGE_COLS.items():
            with_score = sub[sub[col].notna()]
            if with_score.empty:
                continue

            # Deduplicate: same question may appear in multiple files with
            # identical scores — take mean per (id, language) then average
            per_q = (
                with_score.groupby(["id", "language"])[col]
                .mean()
                .reset_index()
            )

            lang_means = per_q.groupby("language")[col].mean()
            en = lang_means.get("en")
            af = lang_means.get("af")

            if en is None or af is None:
                continue

            rows.append({
                "Subject": subject,
                "Judge":   judge_label,
                "EN":      round(en, 4),
                "AF":      round(af, 4),
                "Gap":     round(abs(en - af), 4),
            })

    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame) -> None:
    COL_W = {"Subject": 20, "Judge": 14, "EN": 7, "AF": 7, "Gap": 7}
    divider = "─" * (sum(COL_W.values()) + len(COL_W) * 3 - 1)

    def row_str(subject, judge, en, af, gap, first_in_group):
        subj_cell = f"{subject:<{COL_W['Subject']}}" if first_in_group else " " * COL_W["Subject"]
        return (
            f"  {subj_cell}   "
            f"{judge:<{COL_W['Judge']}}   "
            f"{en:<{COL_W['EN']}.4f}   "
            f"{af:<{COL_W['AF']}.4f}   "
            f"{gap:<{COL_W['Gap']}.4f}"
        )

    header = (
        f"  {'Subject':<{COL_W['Subject']}}   "
        f"{'Judge':<{COL_W['Judge']}}   "
        f"{'EN':>{COL_W['EN']}}   "
        f"{'AF':>{COL_W['AF']}}   "
        f"{'Gap':>{COL_W['Gap']}}"
    )

    print()
    print(header)
    print(f"  {divider}")

    prev_subject = None
    for _, r in summary.iterrows():
        first = r["Subject"] != prev_subject
        if first and prev_subject is not None:
            print(f"  {divider}")
        print(row_str(r["Subject"], r["Judge"], r["EN"], r["AF"], r["Gap"], first))
        prev_subject = r["Subject"]

    print(f"  {divider}")
    print()


def main():
    df = load_all(RESULTS_DIR)
    summary = build_summary(df)
    if summary.empty:
        print("No scoreable data found.")
        return
    print_summary(summary)


if __name__ == "__main__":
    main()
