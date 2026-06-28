# Badminton Match Winner Prediction

End-to-end predictive analytics pipeline using **BWF World Tour match data**, **PostgreSQL (Supabase)**, **SQL analytics**, **feature engineering**, and **machine learning**.

## Business question

> Can we predict the match winner before play starts using player history, head-to-head, recent form, and fatigue?

## Architecture

```
Raw CSV (BWF)  →  Python ETL  →  Supabase (PostgreSQL)
                                      ↓
                              SQL Analytics + Features
                                      ↓
                              ML Model (sklearn)
                                      ↓
                    Next.js + Vercel (production) / Streamlit (local)
```

## Project structure

```
badminton-predictive-analytics/
├── app/                # Next.js pages (Vercel frontend)
├── api/                # Python serverless functions (Vercel)
├── components/         # React UI
├── data/raw/           # BWF CSV files
├── data/processed/     # Cleaned + feature CSVs
├── data/deploy/        # Player lists for frontend (generated)
├── sql/                # Supabase schema + analytics queries
├── src/                # Python pipeline
├── models/             # Trained .pkl models
├── streamlit/          # Local dev UI (optional)
├── scripts/            # Vercel asset export
├── vercel.json         # Vercel config
└── package.json        # Next.js
```

## Data

BWF World Tour matches (2018–2021), five disciplines:

| File | Discipline | ~Rows |
|------|------------|-------|
| `data/raw/ms.csv` | Men's Singles | 3,761 |
| `data/raw/ws.csv` | Women's Singles | 2,975 |
| `data/raw/md.csv` | Men's Doubles | 2,804 |
| `data/raw/wd.csv` | Women's Doubles | 2,522 |
| `data/raw/xd.csv` | Mixed Doubles | 2,856 |

**Note:** This dataset does not include official BWF rankings. The pipeline uses a **rolling win-rate strength proxy** derived from prior matches (no data leakage).

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Supabase (optional for local ML)

Copy `.env.example` → `.env` and add your Supabase URL + service role key.

Run `sql/schema.sql` in the Supabase SQL Editor first.

### 3. Run the pipeline

```bash
# Explore + clean
python -m src.explore_data
python -m src.clean_data

# Feature engineering (default: Men's Singles)
python -m src.feature_engineering

# Train models
python -m src.train_model

# Load into Supabase (after schema + .env)
python -m src.load_supabase
```

### 4. Run locally (Next.js + Python API)

```bash
# Generate frontend assets + ensure model exists
python scripts/export_vercel_assets.py

# Terminal 1 — Next.js frontend
npm install
npm run dev

# Terminal 2 — Python API (Vercel uses this in production automatically)
# For local dev, use Vercel CLI: npx vercel dev
npx vercel dev
```

Open [http://localhost:3000](http://localhost:3000).

**Streamlit (optional local UI):**

```bash
streamlit run streamlit/streamlit_app.py
```

## Deploy to Vercel

1. Push the repo to GitHub.
2. Import the project at [vercel.com/new](https://vercel.com/new).
3. Vercel auto-detects Next.js — **no root directory change needed**.
4. Default build uses `vercel.json`:
   - Installs Python deps + npm packages
   - Runs `scripts/export_vercel_assets.py` (cleans data, trains model, exports JSON)
   - Builds Next.js
5. Deploy.

**Requirements on Vercel:**
- `data/raw/*.csv` must be in the repo (they are).
- Python 3.x is used for `/api/predict` serverless function.
- No env vars required for basic prediction (Supabase is optional).

**CLI deploy:**

```bash
npm i -g vercel
vercel
```

**Local production preview:**

```bash
npx vercel dev
```

This runs Next.js and Python `/api/*` routes together, matching production.

## Features engineered

| Feature | Description |
|---------|-------------|
| `strength_diff` | Rolling win-rate difference (p1 − p2) |
| `head_to_head_diff` | Prior H2H wins (p1 − p2) |
| `recent_form_p1/p2` | Win rate in last 5 matches |
| `fatigue_p1/p2` | Matches played in prior 7 days |
| `tournament_tier` | 100 / 300 / 500 / 750 / 1000 |
| `round_importance` | Early round → Final (1–5) |

## SQL analytics

See `sql/analytics_queries.sql` for portfolio-ready queries:

- Top players by wins
- Tournament frequency
- Upset rate (weaker player wins)
- Finals vs early rounds (avg sets)

## Resume bullets

- Built an end-to-end predictive analytics pipeline using BWF badminton match data and Supabase PostgreSQL.
- Engineered time-safe features (head-to-head, recent form, fatigue) with no data leakage.
- Trained and compared Logistic Regression, Random Forest, and Gradient Boosting models.
- Designed normalized database schema and SQL analytics layer in Supabase.

## Tech stack

Python · PostgreSQL (Supabase) · SQL · scikit-learn · Next.js · Vercel

## Original data license

See [LICENSE](LICENSE) and the original dataset README for citation requirements.
