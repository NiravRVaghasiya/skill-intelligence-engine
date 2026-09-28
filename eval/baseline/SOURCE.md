# Provenance of the vendored keyword baseline

`router.py` and `skills_lib.py` are **unmodified** copies of the keyword router from
[ml-ai-skills](https://github.com/NiravRVaghasiya/ml-ai-skills), MIT-licensed (see `LICENSE`
in this directory, copied from the source repo).

| file | source path | commit | sha256 |
|------|-------------|--------|--------|
| `router.py` | `scripts/router.py` | `8328c600356bcc4b18f1f4a300d6498e66cf3e39` | `15529d76a655a5f0644c7d66de6ec6708144fb340e8a32bbc5624e575c5e11c6` |
| `skills_lib.py` | `scripts/skills_lib.py` | `8328c600356bcc4b18f1f4a300d6498e66cf3e39` | `44b0a45a9e8a1a0c002d994d7de1fd513fcead2de25e3546e4d3d3ba5b098c5b` |

`tests/test_baseline.py` asserts these hashes, so an accidental edit fails CI.
`router.py` is byte-identical at `c54f548` (the commit that introduced it) and at HEAD;
`skills_lib.py` only gained a code-fence dedent helper (`d1bc965`) that routing never calls.

The adapter in `__init__.py` only redirects `load_all_skills()` to a corpus directory — the
scoring, tokenization, stopwords, intent cues, and tie-breaking are the source's own.
