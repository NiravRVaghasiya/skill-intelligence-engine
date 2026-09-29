# Skill Intelligence Engine — dev shortcuts
# On Windows, run these via `make` (Git Bash / WSL / choco make) or copy the commands.

# override: make setup SOURCE=/path/to/ml-ai-skills (a trailing comment would leak spaces into the value)
SOURCE ?= ../ml-ai-skills

.PHONY: install install-dev setup build query route path eval smoke perf collision test serve demo clean

install:            ## install python deps (flat list: index, rerank, API, eval, tests)
	pip install -r requirements.txt

install-dev:        ## install the package editable with every extra (see pyproject.toml)
	pip install -e ".[all,dev]"

setup:              ## vendor corpus from SOURCE + build indexes
	python scripts/setup_corpus.py --source "$(SOURCE)"

build:              ## (re)build indexes from data/skills
	python -m sie.router --build

query:              ## example query — override Q="..."
	python -m sie.router "$(or $(Q),impute missing values and encode categoricals)"

route:              ## full explained RouteResult as JSON — override Q="..."
	python -m sie.router "$(or $(Q),impute missing values and encode categoricals)" --json

path:               ## learning path — override T=<slug>
	python -m sie.router --path "$(or $(T),rag-evaluation)"

eval:               ## run the baseline-vs-SIE evaluation (rebuilds the index; writes benchmarks/)
	python -m eval.run_eval

smoke:              ## CI regression check: no index, no models, no writes
	python -m eval.run_eval --smoke

perf:               ## machine-dependent latency/quality sweep (writes benchmarks/PERFORMANCE.md)
	python -m eval.perf

collision:          ## reproduce the documented keyword-collision case
	python -m eval.collision --source "$(SOURCE)"

test:               ## run the test suite
	pytest -q

serve:              ## start the FastAPI service (config: SIE_* variables, see .env.example)
	uvicorn sie.api:api --reload

demo:               ## launch the Streamlit demo (pip install -e ".[demo]" first)
	streamlit run demo/app_streamlit.py

clean:              ## remove built indexes and caches
	rm -rf data/chroma .pytest_cache
	find . -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
