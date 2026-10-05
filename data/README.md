# Data provenance

This directory contains the two TOFU-derived artifacts used by the experiment:

- `tofu_author_biographies.json`: 10 synthesized fictional-author biographies.
- `tofu_qa_passages.json`: 200 QA records and their synthesized evidence passages (20 questions per author).

They were copied from the existing local `memory/cartridges/data_synthesis` artifacts so this experiment can run independently. The source copies were preserved because other work may still reference them.

SHA-256 checksums at import time:

```text
603cf4799ccb59f05913251f59d2a6dfed260e7bbaa1c7eee37200b7d9ccf4a0  tofu_author_biographies.json
42ac6da06b1faad80cbdfb68a682a9ee5683a67f022c67c02d70d3b335791345  tofu_qa_passages.json
```

`data/benchmarks/quality_dev.jsonl` is downloaded on demand from the official
QuALITY v1.0.1 HTML-stripped development set. The runner selects a seeded
article subset and evaluates every question attached to each selected article.
`data/benchmarks/ruler_16k.jsonl` is the optional prepared RULER subset.
