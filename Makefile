# Skill Intelligence Engine — dev shortcuts
# On Windows, run these via `make` (Git Bash / WSL / choco make) or copy the commands.

SOURCE ?= ../ml-ai-skills          # override: make setup SOURCE=/path/to/ml-ai-skills

.PHONY: install setup build query eval test serve demo clean

install:            ## install python deps
	pip install -r requirements.txt

setup:              ## vendor corpus from SOURCE + build indexes
	python scripts/setup_corpus.py --source "$(SOURCE)"

build:              ## (re)build indexes from data/skills
	python -m sie.router --build

query:              ## example query — override Q="..."
	python -m sie.router "$(or $(Q),impute missing values and encode categoricals)"

eval:               ## run the baseline-vs-SIE evaluation
	python -m eval.run_eval

test:               ## run the test suite
	pytest -q

serve:              ## start the FastAPI service
	uvicorn sie.api:api --reload

demo:               ## launch the Streamlit demo
	streamlit run demo/app_streamlit.py

clean:              ## remove built indexes and caches
	rm -rf data/chroma .pytest_cache **/__pycache__
