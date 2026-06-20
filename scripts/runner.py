"""
Send questions to the Gemini API in both English and Afrikaans.
Saves all responses to results/raw_results.json.
"""

import argparse
import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types
from google.genai.errors import ClientError
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.0))
DELAY = float(os.getenv("API_DELAY", 1.0))


def load_questions(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def query_gemini(client: genai.Client, model_name: str, prompt: str) -> tuple[str, str | None]:
    """Returns (response_text, error). error is None on success."""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=TEMPERATURE),
        )
        return response.text, None
    except ClientError as e:
        if e.code == 404:
            print(
                f"\n[ERROR] 404 — model '{model_name}' not found. "
                "Set GEMINI_MODEL in your .env to a valid model name "
                "(e.g. gemini-2.0-flash) and retry."
            )
        return "", str(e)
    except Exception as e:
        return "", str(e)


def sanity_check(api_key: str, questions: list[dict]) -> None:
    first = questions[0]
    print("=== Sanity check ===")
    print(f"API key loaded : {'*' * 8}{api_key[-4:]}")
    print(f"First question : [{first['id']}] {first['question_en'][:120]}")
    print(f"Total loaded   : {len(questions)} questions")
    print("====================\n")


def run(questions_path: str, output_path: str, model_name: str, run_all: bool) -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in .env")

    client = genai.Client(api_key=api_key)

    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"

    all_questions = load_questions(questions_path)
    sanity_check(api_key, all_questions)

    questions = all_questions if run_all else all_questions[:3]
    if not run_all:
        print(f"Running first {len(questions)} questions (use --all for full set)\n")

    results = []

    # Each question produces two API calls: English + Afrikaans
    total_calls = len(questions) * 2
    pbar = tqdm(total=total_calls, desc="Querying Gemini")

    for q in questions:
        base = {
            "id": q["id"],
            "subject": q["subject"],
            "year": q["year"],
            "question_number": q["question_number"],
            "marks": q["marks"],
            "question_type": q["question_type"],
            "memo_answer": q["memo_answer"],
            "memo_steps": q["memo_steps"],
            "model": model_name,
        }

        for lang, field in [("en", "question_en"), ("af", "question_af")]:
            prompt = q[field]
            response_text, error = query_gemini(client, model_name, prompt)
            results.append({
                **base,
                "language": lang,
                "prompt": prompt,
                "response": response_text if not error else None,
                "error": error,
            })
            pbar.update(1)
            time.sleep(DELAY)

    pbar.close()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    success = sum(1 for r in results if not r["error"])
    print(f"\nDone: {success}/{len(results)} successful  ->  {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Gemini equity evaluation")
    parser.add_argument(
        "--questions",
        default="data/extracted/questions.json",
        help="Path to questions JSON",
    )
    parser.add_argument(
        "--output",
        default="results/raw_results.json",
        help="Output path for raw results",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini model ID (e.g. gemini-1.5-flash, gemini-1.5-pro)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Run all questions (default: first 3 only)",
    )
    args = parser.parse_args()
    run(args.questions, args.output, args.model, args.run_all)


if __name__ == "__main__":
    main()
