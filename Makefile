# Skill Intelligence Engine — dev shortcuts
# On Windows, run these via `make` (Git Bash / WSL / choco make) or copy the commands.

# override: make setup SOURCE=/path/to/ml-ai-skills (a trailing comment would leak spaces into the value)
SOURCE ?= ../ml-ai-skills

.PHONY: install setup build query path eval collision test serve demo clean

install:            ## install python deps
	pip install -r requirements.txt

setup:              ## vendor corpus from SOURCE + build indexes
	python scripts/setup_corpus.py --source "$(SOURCE)"

build:              ## (re)build indexes from data/skills
	python -m sie.router --build

query:              ## example query — override Q="..."
	python -m sie.router "$(or $(Q),impute missing values and encode categoricals)"

path:               ## learning path — override T=<slug>
	python -m sie.router --path "$(or $(T),rag-evaluation)"

eval:               ## run the baseline-vs-SIE evaluation
	python -m eval.run_eval

collision:          ## reproduce the documented keyword-collision case
	python -m eval.collision --source "$(SOURCE)"

test:               ## run the test suite
	pytest -q

serve:              ## start the FastAPI service
	uvicorn sie.api:api --reload

demo:               ## launch the Streamlit demo (pip install streamlit first; not in requirements.txt)
	streamlit run demo/app_streamlit.py

clean:              ## remove built indexes and caches
	rm -rf data/chroma .pytest_cache
	find . -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
