.PHONY: install pipeline api web test lint fmt docker
install: ; pip install -r requirements.txt && cd frontend && npm install
pipeline: ; python -m src.pipeline
api: ; uvicorn api.main:app --reload --port 8000
web: ; cd frontend && npm run dev
test: ; pytest --cov=src --cov=api --cov-report=term-missing && cd frontend && npm test
lint: ; ruff check . && ruff format --check .
fmt: ; ruff format . && ruff check . --fix
docker: ; docker compose up --build
