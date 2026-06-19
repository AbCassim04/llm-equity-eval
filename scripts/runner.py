"""
Send evaluation questions to an LLM and collect responses.
"""

import argparse
import json
import os
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.0))


def load_questions(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def query_model(client: anthropic.Anthropic, question_text: str, model: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": question_text}],
    )
    return response.content[0].text


def run(questions_path: str, output_dir: str, model: str) -> None:
    questions = load_questions(questions_path)
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for q in tqdm(questions, desc="Running evaluations"):
        try:
            response_text = query_model(client, q["text"], model)
            results.append({**q, "response": response_text, "model": model})
        except Exception as e:
            results.append({**q, "response": None, "error": str(e), "model": model})
        time.sleep(0.2)  # basic rate-limit buffer

    output_file = Path(output_dir) / f"responses_{model.replace('/', '-')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} responses -> {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluations")
    parser.add_argument(
        "--questions",
        default="data/extracted/questions_sample.json",
        help="Path to extracted questions JSON",
    )
    parser.add_argument(
        "--output", default=os.getenv("OUTPUT_DIR", "data/responses"), help="Output directory"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model ID to evaluate")
    args = parser.parse_args()
    run(args.questions, args.output, args.model)


if __name__ == "__main__":
    main()
