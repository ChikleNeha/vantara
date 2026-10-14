# Vantara Customer Behaviour Prediction Platform

Predicts which customers are about to stop buying, how much they are worth, and why, then puts it in a dashboard a marketing team can use.
Built from the PRD (`vantara_doc.docx`) on **React + TailwindCSS + FastAPI + SQLite + Ollama**.

| Layer | What it does |
|---|---|
| `src/` pipeline | Cleans transactions, builds point-in-time features, trains 6 classical + 2 neural churn models, a hurdle CLV model, an autoencoder and K-Means/GMM segments, explains with SHAP + LIME, writes SQLite and `docs/REPORT.md` |
| `api/` FastAPI | Single and batch scoring, customer explorer, segments, revenue forecast, explanations, Ollama assistant. Swagger UI at `/docs` |
| `frontend/` React | Overview, Segments, Risk leaderboard, Customer explorer, Score new data (CSV upload + what-if), Model insights, Ask the analyst |
| Ollama | Rewrites model outputs as plain language and answers questions. It never produces a prediction; every feature falls back to templates if it is off |

## Quick start

```bash
# 1. Python side (3.11+)
pip install -r requirements.txt
python -m src.pipeline              # ~2 min; trains everything, fills data/cbp.sqlite, writes docs/REPORT.md
uvicorn api.main:app --port 8000    # API + docs at http://localhost:8000/docs

# 2. Dashboard (Node 20+), new terminal
cd frontend && npm install && npm run dev    # http://localhost:5173

# 3. Optional: plain-language narratives and the Q&A page
ollama pull llama3.2                # or set OLLAMA_MODEL in .env
```

Or everything with Docker: `docker compose up --build` then open http://localhost:8080 (Ollama stays on the host).

### Use the real dataset
The sandbox that built this had no internet, so the committed results come from a **synthetic generator that mimics the Online Retail II schema** (returns, guest checkouts, admin codes, zero prices, duplicates). The report says so in a banner. To use real data:

1. Download *Online Retail II* from the [UCI repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and save it as `data/raw/online_retail_II.xlsx` (both sheets are read), **or** leave it and the pipeline tries `ucimlrepo` automatically.
2. Run `python -m src.pipeline` again. `DATA_SOURCE=file` (or `config/config.yaml: data.source`) forces a source and fails loudly instead of falling back.

## How the modelling is kept honest
- **No leakage:** features use rows before a cutoff date; the label (no purchase in the next 90 days) uses only the 90 days after. `tests/test_features.py` proves features are unchanged when future rows are removed or altered.
- **Test set touched once.** Tuning is 5-fold CV on the training split; boosted models early-stop on validation; the production model and the decision threshold are picked on validation; the test set is only reported.
- **Class imbalance** is handled with class weights on the training split only.
- **VIF pruning** removes collinear features before modelling (dropped features are listed in the report).
- Every run appends to `models_artifacts/experiment_log.csv`; the comparison table in the report is rebuilt from that log, not typed.

## API at a glance
| Method & path | Purpose |
|---|---|
| `POST /api/predict` | Score one customer record; returns probability, risk band, CLV, segment, SHAP drivers, narrative (`?narrative=llm` uses Ollama) |
| `POST /api/predict/batch` | Upload a CSV of **raw transactions** or **customer summaries**; returns ranked scores |
| `GET /api/customers/leaderboard` | Spend-at-risk ranking with segment/country/value/risk filters and paging |
| `GET /api/customers/{id}` · `/explain?method=shap\|lime` | Profile, recent orders, explanations |
| `GET /api/segments` · `/segments/points` · `/revenue` · `/overview` · `/filters` | Dashboard data |
| `GET /api/model/metadata` · `/explain/global` · `/ollama/status` · `/health` | Model and platform status |
| `POST /api/assistant/ask` | Ollama Q&A over a database summary |

Customer record minimum: `recency_days`, `frequency`, `monetary_total`, `tenure_days`. Everything else defaults to training medians. Bad input returns a readable `422` listing each field.

## Tests and quality
```bash
make test        # pytest (64 tests, 97% line coverage of src/ and api/) + vitest (6 page-render tests)
make lint        # ruff check + format check, zero findings
```
Backend integration tests run a small end-to-end pipeline, then exercise the API (validation, SQL-injection safety, error handling, p95 latency, Ollama-down fallback). Frontend tests render every page against **real API responses** captured from the running backend.

## Where this differs from the PRD
| PRD | Here | Why |
|---|---|---|
| PostgreSQL | SQLite via SQLAlchemy | Your stack choice; swapping back is a connection-string change |
| Streamlit (or React) | React + Tailwind | Your stack choice; the PRD allows React |
| — | Ollama | Added by you: narratives and Q&A, never predictions |
| Single regression for CLV | Hurdle model: P(buy) × E[spend given buy] | about half of customers (54% on the synthetic run) spend nothing in the window; a single regression scored R² 0.17 vs 0.45 for the hurdle on the synthetic run |
| ANN/LSTM served | ANN and autoencoder served from NumPy copies of their weights; LSTM scored in batch | Keeps the API image free of TensorFlow (verified: not imported at runtime) |
| Optional: CatBoost, Transformer, DBSCAN | Not built | Marked optional; listed under future work in the report |

## Known limits (read before trusting numbers)
- **The shipped metrics are on synthetic data.** They prove the pipeline works and show the expected *shape* of results, not real customer behaviour. Re-run on the real file before drawing any business conclusion.
- On the synthetic run, churn AUC lands near 0.80 and CLV R² near 0.45 (target 0.60). The report prints met / not met for each PRD target; nothing is rounded up.
- The dashboard was verified with jsdom render tests plus a production build, not in a real browser, and I have not measured the "< 3 s segment view" target. `docker compose` and live Ollama calls were also not run in the build environment (Ollama is tested with a mocked server). Treat those three as unverified until you run them once.
- Product categories are keyword-derived from descriptions (the dataset has none), so "next category" is coarse.
- "Save as PDF" uses the browser's print dialog (print stylesheet hides navigation); CSV export is built in.

## Layout
```
config/config.yaml   all thresholds, paths, hyperparameters
src/                 data · features · models · segmentation · explainability · serving · pipeline · report
api/                 FastAPI app, routers, schemas
frontend/            React + Tailwind dashboard (+ vitest tests)
tests/               pytest suite · notebooks/01_eda.ipynb · scripts/make_diagrams.py
docs/                REPORT.md · eda_report.md · figures/ (architecture, ER, workflow, SHAP, loss curves)
```
