# Khora GraphRAG Benchmark

[![CI](https://github.com/DeytaHQ/khora-graphrag-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/DeytaHQ/khora-graphrag-benchmark/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Reproducible end-to-end evaluation of the [Khora](https://github.com/DeytaHQ/khora) memory system on the [GraphRAG-Bench (ICLR'26)](https://arxiv.org/abs/2506.05690) benchmark suite. Clone, run, and get a report that compares your run against the published Khora reference baseline.

The benchmark drives Khora through GraphRAG-Bench's three evaluation phases (graph construction, retrieval, answer generation) and scores results with the same paper-aligned LLM-judge methodology the [official GraphRAG-Bench reference implementation](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) uses. Outputs land as JSON, Markdown, and HTML reports.

## Requirements

- **Python 3.13** - the pinned `khora` + `litellm` stack does not ship 3.14 wheels yet, so `make setup` stops if it finds anything else.
- **Docker** with Compose - runs the Postgres+pgvector and Neo4j containers via `make docker-up`. Any Docker-compatible runtime works (Docker Desktop, Rancher Desktop, Podman). You can skip Docker if you bring your own Postgres/Neo4j (see [Configuration](#configuration)).
  - **On macOS**, the Docker VM defaults (often 2 CPU / 2 GB) starve Neo4j and trigger transaction-deadlock retries that drag down retrieval-side metrics. Give the VM **≥4 CPUs and ≥8 GB RAM** (Docker/Rancher Desktop -> Settings -> Resources). If you still see `Transaction failed and will be retried` warnings during ingestion, lower `KGB_MAX_CONCURRENT_DOCUMENTS`.
- **`make`** and **`git`**.
- An **OpenAI API key** - used by the LLM judge (`gpt-4o-mini`) and Khora's extraction/embedding pipeline.

Tested on macOS (Apple Silicon) and Linux (x86_64).

## Quickstart

```bash
# 1. Clone + configure
git clone https://github.com/DeytaHQ/khora-graphrag-benchmark.git
cd khora-graphrag-benchmark
cp .env.example .env  # then edit .env and set OPENAI_API_KEY

# 2. Install + start Postgres/Neo4j
make setup

# 3. Smoke test (~5% sample, ~10-15 min, ~$0.20-0.50)
make run-small

# 4. View the reports
open results/latest/report.html
```

That's it. The JSON / Markdown / HTML reports each contain a comparison against the canonical Khora reference numbers shipped in this repo so you can see immediately how your machine + Khora install compare.

## What's measured

GraphRAG-Bench evaluates the entire graph-RAG pipeline, not just final answers:

| Metric | What it measures |
|---|---|
| `mean_answer_score` | LLM-judged correctness of the generated answer vs gold (F-beta + semantic similarity). |
| `accuracy` | Fraction of questions where the generated answer is considered correct. |
| `coverage` | Fraction of gold-answer facts that appear in the generated answer. |
| `faithfulness` | Fraction of generated statements actually supported by retrieved context. |
| `context_relevance` | 0-2 LLM score for how relevant retrieved context is to the question. |
| `evidence_recall` | Fraction of gold-evidence statements attributable to the retrieved context. |
| `rouge_l` | ROUGE-L F1 between generated and gold answers (text overlap). |
| `cost_usd` | OpenAI API spend for this run (judge + Khora extraction). |
| `runtime_min` | Wall-clock time for the run (minutes). |

Plus phase-1 graph structural metrics: `num_nodes`, `num_edges`, `num_communities`, `avg_degree`.

## Running modes

```bash
make run-small    # ~5% sampling   ~10-15 min   ~$0.20-0.50   smoke test
make run-medium   # ~30% sampling  ~45-60 min   ~$1.50-3.00   balanced iteration
make run          # 100% sampling  ~8-9 hours   ~$2.50-5.00   full validation
```

Sampling is deterministic (seed = 42) so reruns at the same sample mode hit the same questions.

## Configuration

Most knobs live in `.env`. Defaults are picked so a fresh clone runs end-to-end with just `OPENAI_API_KEY` set.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | _required_ | Used by the LLM judge and Khora's extraction pipeline. |
| `POSTGRES_URL` | `postgresql://bench:bench@localhost:5432/khora_graphrag_bench` | Postgres+pgvector connection. Override to point at your own instance. |
| `NEO4J_URL` | `bolt://neo4j:benchbench@localhost:7687` | Neo4j connection. Override to point at your own instance. |
| `KHORA_SPEC` | `khora[accel]==0.21.0` | Pip spec for the Khora library. Override to validate against a different release: `KHORA_SPEC="khora[accel]==0.20.0" make setup`. |
| `BENCH_RESULTS_DIR` | `results` | Where reports get written. |

Bring your own Postgres/Neo4j by setting `POSTGRES_URL` and `NEO4J_URL` in `.env` and skipping `make docker-up`.

## Khora reference baseline (full sampling)

These are the numbers `make run` compares against. Source: `khora[accel]==0.21.0`, `gpt-4o-mini` judge, paper-aligned prompts, full `graphrag_bench_novel` (2010 questions). Quality metrics are the mean of two independent full runs on the harness as of bench commit `d627195` (post PR #3 main: uniform answer prompt and the corrected `coverage` / `evidence_recall` denominators). `runtime_min` and `cost_usd` are from the same two developer-machine full runs.

| metric | value |
|---|---:|
| `mean_answer_score` | 0.694 |
| `accuracy` | 0.799 |
| `coverage` | 0.711 |
| `rouge_l` | 0.439 |
| `faithfulness` | 0.748 |
| `context_relevance` | 0.352 |
| `evidence_recall` | 0.891 |
| `runtime_min` | ~478 |
| `cost_usd` | ~$3.59 |

Numbers vary slightly run-to-run because answer generation is not seeded, so the generated answers, and therefore the judge inputs, differ between runs (the LLM judge itself runs at `temperature=0` with a fixed seed and is disk-cached). Expect roughly ±0.005 noise on most aggregate metrics, plus run-to-run cost variance from judge-cache hits. `faithfulness` is structurally noisier (around ±0.02 across runs) because the judge decomposes the generated answer into per-statement verdicts.

### Reproducibility

`make run` on a clean docker stack reproduces the generation-side metrics (`accuracy`, `mean_answer_score`) within run-to-run variance — roughly ±0.005 — driven by non-deterministic answer generation (the judge runs at `temperature=0` with a fixed seed and is cached).

Retrieval-side metrics (`coverage`, `faithfulness`, `context_relevance`, `evidence_recall`) are sensitive to Neo4j write throughput during ingestion: on a laptop docker stack they can land a few points below the reference. That's a hardware effect, not a regression.

If your numbers deviate materially beyond that, the most common causes are (a) `OPENAI_API_KEY` rate-limit retries dropping judge calls, (b) `make setup` not installing the pinned `khora` version, or (c) the cross-encoder reranker model failing to download from HuggingFace.

Questions that error during scoring (e.g. a persistent judge or embedding failure after retries) are excluded from the aggregates and reported as `error_rate`; the reports show a banner whenever any questions errored. A non-trivial `error_rate` means the run is **not** comparable to the reference baseline.

## CLI

If you prefer skipping `make`, the same flow is available via the installed CLI:

```bash
khora-graphrag-bench run --sample small      # or medium / full
khora-graphrag-bench report                  # regenerate reports from latest run
khora-graphrag-bench report --format html    # regenerate just one format
khora-graphrag-bench report --run-id <id>    # rebuild reports for a specific run
khora-graphrag-bench --help                  # full options
```

## How the comparison numbers stay honest

The `report.md` and `report.html` outputs show your local numbers next to the published reference baseline (the table above, baked into the repo at [`reporters/_reference.py`](src/khora_graphrag_bench/reporters/_reference.py)). When we publish updated numbers, that file gets bumped along with the Khora version pin.

If your numbers come in materially below the reference, the most common culprits are (a) different Khora version, (b) the LLM judge timing out and falling back to plain text, or (c) ingestion not completing (`build_graph` reported 0 nodes). The HTML report surfaces the construction stats up top so the third case is visible at a glance.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `OPENAI_API_KEY is not set` | `.env` not loaded or key absent | `cp .env.example .env`, set `OPENAI_API_KEY` |
| `docker compose up` hangs | Existing containers on the same ports | `make clean-all` then `make setup`, or change `POSTGRES_PORT` / `NEO4J_BOLT_PORT` in `.env` |
| `Graph construction produced 0 entities` | Khora extraction silently failing (usually OpenAI rate-limit or model-name typo) | Check the run log; lower `MAX_CONCURRENT_LLM_CALLS` via `params.max_concurrent_llm_calls` |
| Run aborts mid-way | OpenAI rate limits or transient errors | Re-run; the LLM judge cache (`.cache/khora-graphrag-bench/llm_judge/`) means re-runs only re-execute uncached questions |

## Repo layout

```
khora-graphrag-benchmark/
├── Makefile                          # make setup / run-small / run-medium / run / report / clean
├── docker-compose.yml                # postgres+pgvector + neo4j
├── pyproject.toml                    # khora==0.21.0 pinned by default (override via KHORA_SPEC)
├── .github/workflows/ci.yml          # lint + ty + pytest/coverage + pip-audit
├── .pre-commit-config.yaml           # prek hooks: ruff (check/format) + ty
├── CONTRIBUTING.md                   # dev setup, checks, benchmark-integrity rules
├── tests/unit/                       # unit suite (mocks all external services)
├── scripts/                          # one-off diagnostics
├── src/khora_graphrag_bench/
│   ├── cli.py                        # click CLI
│   ├── adapters/khora.py             # GraphRAGAdapter implementation
│   ├── harness/
│   │   ├── base.py                   # Protocols + dataclasses
│   │   ├── evaluation.py             # Paper-aligned judges (with few-shot prompts)
│   │   ├── runner.py                 # Three-phase execution loop
│   │   └── results.py                # Result dataclasses
│   ├── datasets/
│   │   ├── loader.py                 # Downloads + caches the GraphRAG-Bench JSON
│   │   ├── converters.py             # Raw JSON -> typed dataset model
│   │   └── schema.py                 # GraphRAGDataset, GraphRAGQuestion, ...
│   └── reporters/
│       ├── json_writer.py            # JSON report
│       ├── markdown.py               # Markdown report
│       ├── html.py                   # HTML report
│       └── _reference.py             # Canonical Khora baseline numbers
└── results/                          # generated; one subdir per run + latest symlink
```

## Development

```bash
uv sync --extra dev          # install runtime + dev deps
uv run pre-commit install    # optional: install the prek git hook

# the same checks CI runs:
uv run ruff check src/ tests/          # lint
uv run ruff format --check src/ tests/  # format
uv run ty check src/                    # type check
uv run pytest -n auto                    # unit tests + coverage (floor 80%)
# or run every hook at once:
prek run --all-files
```

Unit tests mock all external services (no Postgres, Neo4j, network, or API
keys) and run in seconds. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full
workflow and benchmark-integrity rules.

## Credits

- [GraphRAG-Bench](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) - the benchmark methodology, dataset, and judge prompts (MIT; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)).
- [Khora](https://github.com/DeytaHQ/khora) - the memory system under evaluation.

## Citation

This harness reproduces the GraphRAG-Bench benchmark introduced in:

```bibtex
@article{xiang2025use,
  title={When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation},
  author={Xiang, Zhishang and Wu, Chuanjie and Zhang, Qinggang and Chen, Shengyuan and Hong, Zijin and Huang, Xiao and Su, Jinsong},
  journal={arXiv preprint arXiv:2506.05690},
  year={2025}
}
```

## License

Apache 2.0. See [LICENSE](LICENSE).
