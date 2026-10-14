"""FastAPI application: Customer Behaviour Prediction Platform.

Run:  uvicorn api.main:app --reload      (docs at /docs)
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.deps import AppState, load_metrics, log
from api.routers import analytics, assistant, customers, predict, system
from src.serving.db import get_engine
from src.serving.ollama_client import OllamaClient
from src.serving.scoring import ModelBundle
from src.utils.config import get_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model bundle once at startup; start in degraded mode if the pipeline has not run."""
    cfg = get_config()
    bundle = None
    try:
        bundle = ModelBundle(cfg)
        log.info("Model bundle loaded: %s (threshold %.3f)", bundle.meta["production_model"], bundle.threshold)
    except FileNotFoundError:
        log.error("Model artifacts missing in %s. Run `python -m src.pipeline`.", cfg.path("artifacts_dir"))
    app.state.cbp = AppState(cfg, get_engine(cfg.path("db_path")), bundle, load_metrics(cfg.path("artifacts_dir")), OllamaClient(cfg))
    yield


app = FastAPI(
    title="Customer Behaviour Prediction API",
    version="1.0.0",
    description="Churn, lifetime value, segmentation and explanations for Vantara Retail Solutions.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:8080").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    """INFO-level structured request log with latency."""
    rid, t0 = uuid.uuid4().hex[:8], time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    log.info("rid=%s %s %s -> %d in %.0fms", rid, request.method, request.url.path, response.status_code, ms)
    response.headers["X-Request-ID"], response.headers["X-Response-Time-ms"] = rid, f"{ms:.0f}"
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    """Readable 422 messages for malformed customer records."""
    errors = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "problem": e["msg"]} for e in exc.errors()]
    log.warning("Validation failure on %s: %s", request.url.path, errors)
    return JSONResponse(status_code=422, content={"detail": "Invalid request", "errors": errors})


for r in (system.router, predict.router, customers.router, analytics.router, assistant.router):
    app.include_router(r)
