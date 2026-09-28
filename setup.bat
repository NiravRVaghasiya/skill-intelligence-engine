@echo off
REM One-command setup on Windows. Edit SOURCE if your ml-ai-skills lives elsewhere.
set SOURCE=..\ml-ai-skills
if not "%~1"=="" set SOURCE=%~1
python scripts\setup_corpus.py --source "%SOURCE%"
