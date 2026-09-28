# Corpus goes here

Vendor or symlink the `ml-ai-skills` SKILL.md files into this directory, e.g.:

```bash
git submodule add https://github.com/NiravRVaghasiya/ml-ai-skills.git data/ml-ai-skills
# then point ingest at data/ml-ai-skills, or copy the <slug>/SKILL.md folders here.
```

`sie.ingest.load_corpus` recursively globs for `**/SKILL.md`.
