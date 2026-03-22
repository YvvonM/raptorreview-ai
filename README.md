# RaptorReview AI

[![Test](https://github.com/dev-k99/raptorreview-ai/actions/workflows/test.yml/badge.svg)](https://github.com/dev-k99/raptorreview-ai/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Powered by Groq](https://img.shields.io/badge/Inference-Groq-orange)](https://groq.com)
[![GitHub Stars](https://img.shields.io/github/stars/dev-k99/raptorreview-ai?style=social)](https://github.com/dev-k99/raptorreview-ai)

A GitHub Action that reviews pull requests using Groq's free-tier inference API.
It posts structured, line-specific code review comments — security issues,
performance regressions, readability problems, missing tests — directly in the
GitHub PR interface. No paid API. No infrastructure.

---

## What it does

- Diffs each PR against its base branch using `git diff origin/<base>...HEAD`
- Sends the diff to Groq (`llama-3.1-70b-versatile` by default)
- Parses the model's structured JSON response and posts:
  - An overall review summary with a categorised finding list
  - Inline comments on the specific lines where issues were found
- Handles large diffs by truncating to ~3 000 tokens and noting the cutoff

---

## Install

### 1. Add the workflow file

Create `.github/workflows/raptor-review.yml` in your repository:

```yaml
name: RaptorReview AI

on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  contents: read
  pull-requests: write

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: dev-k99/raptorreview-ai@v1
        with:
          groq_api_key: ${{ secrets.GROQ_API_KEY }}
          github_token: ${{ secrets.GITHUB_TOKEN }}
```

### 2. Add your Groq API key

Get a free key at [console.groq.com](https://console.groq.com), then add it as a
repository secret:

**Settings → Secrets and variables → Actions → New repository secret**

- Name: `GROQ_API_KEY`
- Value: your key

### 3. Open a pull request

RaptorReview will post its review automatically on every PR open, push, or reopen.

---

## Configuration

All inputs are optional except `groq_api_key`.

| Input | Default | Description |
|---|---|---|
| `groq_api_key` | — | **Required.** Groq API key. Pass via `${{ secrets.GROQ_API_KEY }}`. |
| `github_token` | `${{ github.token }}` | GitHub token for posting comments. The default is sufficient. |
| `model` | `llama-3.1-70b-versatile` | Groq model. Also supports `llama-3.1-8b-instant` for faster, lighter reviews. |
| `temperature` | `0.2` | Sampling temperature `[0.0, 1.0]`. Lower = more consistent output. |
| `max_tokens` | `2048` | Maximum tokens in the model response. |
| `custom_prompt` | — | Fully replace the system prompt. Useful for language- or domain-specific rules. |

Example with all options:

```yaml
- uses: dev-k99/raptorreview-ai@v1
  with:
    groq_api_key:   ${{ secrets.GROQ_API_KEY }}
    github_token:   ${{ secrets.GITHUB_TOKEN }}
    model:          "llama-3.1-8b-instant"
    temperature:    "0.1"
    max_tokens:     "3000"
    custom_prompt:  "You are a Go expert. Focus on goroutine safety and interface misuse."
```

---

## How it works

```
Pull Request event
        |
        v
git diff origin/<base>...HEAD
        |
        v
Truncate to ~3 000 tokens (if needed)
        |
        v
Groq API  --  llama-3.1-70b-versatile
        |
        v
Parse JSON response
  { file, line, severity, title, suggestion, why }
        |
        |-- POST /repos/.../pulls/.../reviews   (summary + inline comments)
        |
        v
Done. CI exits 0 regardless of review outcome.
```

The model is prompted to return a strict JSON schema so the response is always
machine-parseable. Groq's free tier provides sufficient throughput for active
teams at normal PR volume without hitting rate limits.

If Groq is unavailable (network error, quota exhausted, API outage), the action
posts a plain comment explaining the failure and exits with code 0 so your CI
pipeline is never blocked by a review service failure.

---

## Security

- Code diffs are transmitted to Groq's API. Review
  [Groq's privacy policy](https://groq.com/privacy-policy) before enabling this
  on repositories containing regulated or highly sensitive data.
- `GROQ_API_KEY` is consumed as an environment variable and is never written to
  any log output.
- The workflow uses the minimal permission set: `contents: read` and
  `pull-requests: write`. No other scopes are requested.
- The action always exits with code 0. A review failure will never break your
  build or block a merge.

---

## Limitations

- Large diffs are truncated to the first ~3 000 tokens. Open smaller, more
  focused PRs to get complete coverage.
- Inline comment line numbers are provided by the model and are best-effort. If
  a suggested line is not present in the diff, the comment is included in the
  summary section instead of as an inline annotation.
- This is a first-pass automated review, not a replacement for human review.
  It catches common issues quickly; your team still makes the final call.

---

## Roadmap

- [ ] Per-path filtering (skip generated files, lock files, migration files)
- [ ] Configurable minimum severity threshold (e.g. suppress suggestions)
- [ ] Support for Hugging Face Inference API as an alternative provider
- [ ] PR description quality check as an optional second pass

---

## Development

```bash
# Clone and install dependencies
git clone https://github.com/dev-k99/raptorreview-ai.git
cd raptorreview-ai
pip install -r requirements.txt
pip install ruff  # for linting

# Lint
ruff check src/

# Smoke test (no API calls made)
PYTHONPATH=src python -c "import review; print('OK')"
```

The test workflow ([.github/workflows/test.yml](.github/workflows/test.yml)) runs
`ruff check` and the import smoke test on every push and PR against `main`.

---

## Contributing

1. Fork the repository
2. Create a feature branch off `main`
3. Ensure `ruff check src/` passes with zero warnings
4. Open a pull request — RaptorReview AI will review it automatically

---

## License

MIT — see [LICENSE](LICENSE).
