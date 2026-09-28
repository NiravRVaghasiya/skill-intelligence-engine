# Benchmarks — SIE vs keyword baseline

> Run `python -m eval.run_eval` to regenerate. Numbers below are placeholders
> until you run against the vendored `data/skills` corpus.

## Headline

| System | recall@3 | MRR | nDCG@3 |
|--------|:-------:|:---:|:------:|
| Keyword router (baseline) | _TBD_ | _TBD_ | _TBD_ |
| **SIE (hybrid + rerank)** | **_TBD_** | **_TBD_** | **_TBD_** |

## Vocabulary-collision case

The source repo documents a specific keyword-collision failure in `docs/ROUTING.md`.
Record here the query, what the keyword router returns, and what SIE returns — this is
the qualitative money example for the README.

- Query: _..._
- Keyword router top-1: _..._ (wrong)
- SIE top-1: _..._ (correct)
