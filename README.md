# LLM Equity Evaluation — South African Grade 12 Exams

**Author:** Aboobaker Cassim, University of the Witwatersrand

Does a large language model perform equitably across English and Afrikaans? This project evaluates **Gemini 2.5 Flash** on 2025 South African National Senior Certificate (NSC) exam questions, comparing response quality between the two official languages of instruction using three independent scoring methods.

---

## Research Question

> When the same exam question is posed in English and in Afrikaans, does Gemini produce responses of equivalent quality — and does the choice of scoring method affect the answer?

South Africa's NSC exams are set bilingually by the Department of Basic Education. Both language versions are legally equivalent and test identical knowledge. Any performance gap between them reflects a model bias, not a difference in question difficulty.

---

## Dataset

**Source:** South African Department of Basic Education — NSC Past Examination Papers. Available at: https://www.education.gov.za/Curriculum/NationalSeniorCertificate(NSC)Examinations/2025NovemberExamPapers.aspx

| Subject | Questions | Paper |
|---|---|---|
| Mathematics | 48 | 2025 Paper 1 |
| Life Sciences | 47 | 2025 Paper 1 |
| Business Studies | 34 | 2025 Paper 1 |
| **Total** | **129** | |

Each question is stored bilingually with the following fields:

```
id, subject, year, paper, section, question_number, marks,
question_en, question_af, memo_answer, memo_steps, question_type, notes
```

`memo_answer` is the correct answer; `memo_steps` is the marking guide (list of accepted steps/points). Both are in English and are used as the scoring reference across both language variants.

Questions span multiple types: `multiple_choice`, `short_answer`, and `long_answer`.

---

## Project Structure

```
llm-equity-eval/
├── data/
│   └── extracted/
│       ├── questions_math_2025_p1.json
│       ├── questions_bio_2025_p1.json
│       └── questions_bus_2025_p1.json
├── results/                        # Raw API responses and scored CSVs
├── scripts/
│   ├── runner.py                   # Send questions to Gemini, save responses
│   ├── scorer.py                   # Score responses (3 judges)
│   └── summary.py                  # Print subjects × judges summary table
├── .env.example
└── requirements.txt
```

---

## Reproducing the Results

### 1. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Add your API keys to `.env`:

```
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash
```

### 2. Run the model

Send all 129 questions to Gemini in both English and Afrikaans (258 API calls total):

```bash
# Dry run — first 3 questions only (sanity check)
python scripts/runner.py --questions data/extracted/questions_math_2025_p1.json

# Full run
python scripts/runner.py --questions data/extracted/questions_math_2025_p1.json --all --output results/raw_math.json
python scripts/runner.py --questions data/extracted/questions_bio_2025_p1.json  --all --output results/raw_bio.json
python scripts/runner.py --questions data/extracted/questions_bus_2025_p1.json  --all --output results/raw_bus.json
```

The runner prints a sanity check (masked API key, first question preview) before making any calls, and shows a tqdm progress bar during the run.

### 3. Score the responses

Three judges are available:

```bash
# Judge 1 — keyword overlap against memo_answer / memo_steps (no API calls)
python scripts/scorer.py --raw results/raw_math.json --output results/scored_math_j1.csv

# Judge 2 — Gemini 2.5 Flash rates each response 0.0–1.0
python scripts/scorer.py --raw results/raw_math.json --output results/scored_math_j2.csv --judge gemini

# Judge 3 — Claude Haiku rates each response 0.0–1.0
python scripts/scorer.py --raw results/raw_math.json --output results/scored_math_j3.csv --judge claude
```

### 4. View the summary

```bash
python scripts/summary.py
```

---

## Key Findings

Summary across all 129 questions, scored independently by four judges:

```
  Subject            Judge           EN        AF       Gap
  ───────────────────────────────────────────────────────────
  Mathematics        J1 Keyword    0.6712    0.4688    0.2024
                     J1b AF memo   0.6712    0.6752    0.0040
                     J2 Gemini     0.9795    0.9718    0.0077
                     J3 Claude     0.8504    0.8447    0.0057
  ───────────────────────────────────────────────────────────
  Life Sciences      J1 Keyword    0.7360    0.2075    0.5285
                     J1b AF memo   0.7360    0.7280    0.0079
                     J2 Gemini     1.0000    0.9724    0.0276
                     J3 Claude     0.9413    0.9033    0.0380
  ───────────────────────────────────────────────────────────
  Business Studies   J1 Keyword    0.5534    0.2190    0.3344
                     J1b AF memo   0.5534    0.5541    0.0001
                     J2 Gemini     0.8407    0.8324    0.0084
                     J3 Claude     0.7474    0.7641   -0.0168
  ───────────────────────────────────────────────────────────
```

**The scoring method matters more than the language gap.**

- **J1 keyword overlap dramatically overstates the gap** (up to 0.53 in Life Sciences). Afrikaans responses are scored against English memo tokens, so correct Afrikaans answers are penalised regardless of accuracy. This is a measurement artifact, not a real performance gap.
- **Both LLM judges show much smaller gaps (< 0.04)** across all three subjects, and broadly agree with each other on direction and magnitude.
- **Business Studies is the only subject where Afrikaans scores English on J3 Claude** (AF 0.76 > EN 0.74), confirming there is no consistent directional bias.
- **Overall conclusion:** The J1b finding is definitive: the same keyword scorer using Afrikaans memo tokens instead of English ones collapses the gap by up to 478x, confirming the gap was entirely a measurement artefact.

---

## Data Sources

- Department of Basic Education — 2025 November NSC Exam Papers: https://www.education.gov.za/Curriculum/NationalSeniorCertificate(NSC)Examinations/2025NovemberExamPapers.aspx
