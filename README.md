# LLM Equity Evaluation

A framework for evaluating large language models on equity and fairness metrics across demographic groups.

## Overview

This project extracts questions from source datasets, runs them through one or more LLMs, and scores responses for bias, consistency, and equity across groups such as race, gender, age, and socioeconomic status.

## Project Structure

```
llm-equity-eval/
├── data/
│   └── extracted/
│       └── questions_sample.json   # Extracted evaluation questions
├── scripts/
│   ├── extract.py                  # Extract and preprocess questions from raw sources
│   ├── runner.py                   # Send questions to LLM APIs and collect responses
│   └── scorer.py                   # Score responses for equity metrics
├── .env.example                    # Template for required environment variables
├── requirements.txt                # Python dependencies
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys in .env
```

## Usage

### 1. Extract questions

```bash
python scripts/extract.py --input data/raw/ --output data/extracted/questions_sample.json
```

### 2. Run LLM evaluations

```bash
python scripts/runner.py --questions data/extracted/questions_sample.json --output data/responses/
```

### 3. Score responses

```bash
python scripts/scorer.py --responses data/responses/ --output data/scores/
```

## Equity Metrics

- **Demographic Parity**: Response quality consistency across demographic groups
- **Sentiment Bias**: Difference in sentiment polarity per group
- **Refusal Rate**: Rate of content refusals per group
- **Toxicity Gap**: Difference in toxicity scores across groups

## Environment Variables

See `.env.example` for all required variables.
