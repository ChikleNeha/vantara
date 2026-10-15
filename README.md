# Vantara Customer Behaviour Prediction Platform

Predicts which customers are about to stop buying, how much they are worth, and why, then puts it in a dashboard a marketing team can use.
Built on **React + TailwindCSS + FastAPI + SQLite + Ollama**.

| Layer | What it does |
|---|---|
| `src/` pipeline | Cleans transactions, builds point-in-time features, trains 6 classical + 2 neural churn models, a hurdle CLV model, an autoencoder and K-Means/GMM segments, explains with SHAP + LIME, writes SQLite and `docs/REPORT.md` |
| `api/` FastAPI | Single and batch scoring, customer explorer, segments, revenue forecast, explanations, Ollama assistant. Swagger UI at `/docs` |
| `frontend/` React | Overview, Segments, Risk leaderboard, Customer explorer, Score new data (CSV upload + what-if), Model insights, and a chat-style Ask the analyst page. Pastel card design, page and chart animations (disabled automatically for reduced-motion users), Lucide icons |
| Ollama | Rewrites model outputs as plain language and answers questions. It never produces a prediction; every feature falls back to templates if it is off |

## Quick start

```bash
# 1. Python side (3.11 or 3.12; TensorFlow has no wheels for newer versions yet)
python -m pip install -r requirements.txt      # "python -m pip" installs into the SAME Python that runs the pipeline
python -m src.pipeline              # ~2 min; trains everything, fills data/cbp.sqlite, writes docs/REPORT.md
uvicorn api.main:app --port 8000    # API + docs at http://localhost:8000/docs

# 2. Dashboard (Node 20+), new terminal
cd frontend && npm install && npm run dev    # http://localhost:5173

# 3. Optional: plain-language narratives and the Q&A page
ollama pull llama3.2:latest                # or set OLLAMA_MODEL in .env
```

Or everything with Docker: `docker compose up --build` then open http://localhost:8080 (Ollama stays on the host).

### Data source
Place the *Online Retail II* file from the [UCI repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) at `data/raw/online_retail_II.xlsx` (both sheets are read) and run `python -m src.pipeline`. With `data.source: auto` (default) the pipeline uses that file if present, then tries `ucimlrepo`, then falls back to a built-in **synthetic generator** that mimics the dataset's schema (returns, guest checkouts, admin codes, zero prices, duplicates). The data source is printed in `docs/REPORT.md` and shown in the dashboard; a banner appears whenever synthetic data was used. Set `DATA_SOURCE=file` (or `data.source: file`) to fail loudly instead of falling back.

`docs/REPORT.md` is **regenerated on every pipeline run** from the logged metrics, so it always describes the data you actually trained on.

### Troubleshooting
| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'lime'` (or `shap`, `xgboost`, ...) | The package went into a different Python than the one running the pipeline. Use `python -m pip install -r requirements.txt`, and check with `python -c "import shap, xgboost, lightgbm, tensorflow"`. LIME itself is optional: if it is missing the pipeline still finishes and skips only the SHAP-vs-LIME agreement check. |
| TensorFlow fails to install | Use Python 3.11 or 3.12 in a fresh virtual environment. |
| Dashboard shows "The backend isn't ready" | Run the pipeline once, then start `uvicorn api.main:app --port 8000`. |
| Q&A page says the model is not reachable | Start Ollama and run `ollama pull llama3.2` (or set `OLLAMA_MODEL`). Everything else works without it. |

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
make test        # pytest (65 tests, 97% line coverage of src/ and api/) + vitest (7 UI tests)
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
- **Check the data source first.** If the report banner says SYNTHETIC, the numbers only prove the pipeline works. Conclusions need a run on the real file.
- **Targets are reported, not assumed.** `docs/REPORT.md` prints met / not met for each PRD target. On the synthetic run churn AUC was about 0.79 and CLV R² about 0.45 (target 0.60); real data will differ. The test split holds a few hundred to a few thousand customers, so differences of about 0.01 between models are within sampling noise.
- The dashboard is covered by automated render tests against real API responses and a production build; judge the visual design and the "under 3 seconds" segment-view target in your own browser. `docker compose` and live Ollama calls were not run in the build environment (Ollama is tested with a mock server); treat those as unverified until you run them once.
- Product categories are keyword-derived from descriptions (the dataset has none), so "next category" is coarse.
- "Save as PDF" uses the browser's print dialog (print styles hide navigation); CSV export is built in.

## Layout
```
config/config.yaml   all thresholds, paths, hyperparameters
src/                 data · features · models · segmentation · explainability · serving · pipeline · report
api/                 FastAPI app, routers, schemas
frontend/            React + Tailwind dashboard (+ vitest tests)
tests/               pytest suite · notebooks/01_eda.ipynb · scripts/make_diagrams.py
docs/                REPORT.md · eda_report.md · figures/ (architecture, ER, workflow, SHAP, loss curves)
```