# Security Policy

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue.

- Use GitHub's [private vulnerability reporting](https://github.com/DeytaHQ/khora-graphrag-benchmark/security/advisories/new), or
- email **security@deyta.ai**.

We aim to acknowledge reports within a few business days.

## Leaked credentials

This benchmark runs with a real `OPENAI_API_KEY` in your local `.env` (which is
gitignored and must never be committed). Before attaching logs, run output, or
report files to an issue or PR, **scrub any API keys**. If you believe you have
committed a key, rotate it immediately and notify us via the channels above.
