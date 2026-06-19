"""
Extract and preprocess evaluation questions from raw source data.
"""

import argparse
import json
import os
from pathlib import Path


DEMOGRAPHIC_GROUPS = ["race", "gender", "age", "socioeconomic_status", "religion"]


def load_raw_questions(input_path: str) -> list[dict]:
    """Load questions from a JSON file or directory of JSON files."""
    path = Path(input_path)
    questions = []

    if path.is_file():
        with open(path) as f:
            data = json.load(f)
        questions = data if isinstance(data, list) else [data]
    elif path.is_dir():
        for file in sorted(path.glob("*.json")):
            with open(file) as f:
                data = json.load(f)
            questions.extend(data if isinstance(data, list) else [data])

    return questions


def validate_question(q: dict) -> bool:
    required = {"id", "text", "demographic_group", "category"}
    return required.issubset(q.keys()) and q["demographic_group"] in DEMOGRAPHIC_GROUPS


def extract(input_path: str, output_path: str) -> None:
    raw = load_raw_questions(input_path)
    valid = [q for q in raw if validate_question(q)]
    skipped = len(raw) - len(valid)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(valid, f, indent=2)

    print(f"Extracted {len(valid)} questions ({skipped} skipped) -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract evaluation questions")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument(
        "--output",
        default="data/extracted/questions_sample.json",
        help="Output JSON file",
    )
    args = parser.parse_args()
    extract(args.input, args.output)


if __name__ == "__main__":
    main()
