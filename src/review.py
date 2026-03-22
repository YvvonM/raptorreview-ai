"""RaptorReview AI — Groq-powered pull request reviewer.

Entrypoint for the composite GitHub Action. Reads the PR diff, submits it to
the Groq inference API, parses the structured review response, and posts inline
and summary comments back to the pull request via the GitHub REST API.

Environment variables (set by action.yml):
    GROQ_API_KEY          Groq API key (required)
    GITHUB_TOKEN          GitHub token for API calls (required)
    GITHUB_REPOSITORY     owner/repo string (injected by Actions runner)
    GITHUB_EVENT_PATH     path to the event JSON payload (injected by runner)
    INPUT_MODEL           Groq model identifier
    INPUT_TEMPERATURE     sampling temperature (float string)
    INPUT_MAX_TOKENS      max response tokens (int string)
    INPUT_CUSTOM_PROMPT   optional full system prompt override
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import requests
import tiktoken
from groq import APIConnectionError, APIStatusError, Groq

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("raptorreview")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Token budget for the diff sent to the model.
MAX_DIFF_TOKENS: int = 3_000

# Delays (seconds) between retries when Groq returns HTTP 429.
RETRY_DELAYS: list[int] = [2, 5, 15]

# Models tried in order when the primary model is unavailable or decommissioned.
FALLBACK_MODELS: list[str] = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]

# Glob patterns for files excluded from the diff before sending to the model.
# These are noise — generated, minified, or lock files the LLM cannot usefully review.
SKIP_PATTERNS: list[str] = [
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
    "pnpm-lock.yaml",
    "*.min.js",
    "*.min.css",
    "*.pb.go",
    "*.generated.*",
    "*.gen.*",
    "dist/*",
    "build/*",
    ".next/*",
    "node_modules/*",
    "vendor/*",
    "__pycache__/*",
    "*.pyc",
]

SEVERITY_LABELS: dict[str, str] = {
    "critical":   "[CRITICAL]",
    "warning":    "[WARNING]",
    "suggestion": "[SUGGESTION]",
}

DEFAULT_SYSTEM_PROMPT: str = """\
You are a senior staff engineer performing a pull request code review. Be direct, \
technically precise, and constructive. You may be wry when the code clearly warrants \
it, but never cruel. Focus on: security vulnerabilities, performance regressions, \
readability, missing or inadequate tests, and deviations from established best \
practices.

Respond with ONLY a valid JSON object — no markdown fences, no preamble, no \
explanation outside the JSON. The object must conform exactly to this schema:

{
  "comments": [
    {
      "file": "<relative file path, e.g. src/auth.py>",
      "line": <integer line number in the NEW version of the file>,
      "severity": "<critical|warning|suggestion>",
      "title": "<concise issue title, max 80 chars>",
      "suggestion": "<concrete fix or alternative — code snippet preferred>",
      "why": "<explanation of impact>"
    }
  ],
  "summary": "<overall review paragraph>"
}

Set "line" to 0 if a comment does not map to a specific line. The "comments" \
array may be empty if the diff is clean. Only report issues clearly visible \
in the diff — never fabricate problems."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ReviewComment:
    file: str
    line: int
    severity: str
    title: str
    suggestion: str
    why: str

    def to_markdown(self) -> str:
        label = SEVERITY_LABELS.get(self.severity, f"[{self.severity.upper()}]")
        parts: list[str] = [
            f"**{label} {self.title}**",
            "",
            self.why,
        ]
        if self.suggestion:
            parts += [
                "",
                "**Suggestion:**",
                f"```\n{self.suggestion}\n```",
            ]
        return "\n".join(parts)


@dataclass
class ReviewResult:
    comments: list[ReviewComment] = field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base (GPT-4 / Llama-compatible encoding).

    Falls back to a character-based estimate if tiktoken is unavailable.
    """
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:  # noqa: BLE001
        return len(text) // 4


# ---------------------------------------------------------------------------
# Diff acquisition and conditioning
# ---------------------------------------------------------------------------


def get_diff(base_ref: str) -> str:
    """Run git diff against the PR base branch and return the unified diff."""
    cmd = ["git", "diff", f"origin/{base_ref}...HEAD"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as exc:
        log.warning("git diff exited %d: %s", exc.returncode, exc.stderr.strip())
        return ""


def filter_diff(diff: str) -> tuple[str, list[str]]:
    """Remove noise files from the unified diff.

    Splits the diff on 'diff --git' headers, checks each file path against
    SKIP_PATTERNS, and reassembles only the sections worth reviewing.

    Returns:
        (filtered_diff, skipped_file_paths)
    """
    sections = re.split(r"(?=^diff --git )", diff, flags=re.MULTILINE)
    kept: list[str] = []
    skipped: list[str] = []

    for section in sections:
        if not section.strip():
            continue
        header = section.split("\n")[0]
        match = re.match(r"^diff --git a/.+ b/(.+)$", header)
        if match:
            filepath = match.group(1)
            basename = os.path.basename(filepath)
            if any(
                fnmatch.fnmatch(filepath, p) or fnmatch.fnmatch(basename, p)
                for p in SKIP_PATTERNS
            ):
                skipped.append(filepath)
                continue
        kept.append(section)

    return "".join(kept), skipped


def truncate_diff(diff: str) -> tuple[str, bool, int]:
    """Truncate the diff to MAX_DIFF_TOKENS using binary search on line boundaries.

    Returns:
        (diff_text, was_truncated, final_token_count)
    """
    token_count = count_tokens(diff)
    if token_count <= MAX_DIFF_TOKENS:
        return diff, False, token_count

    lines = diff.split("\n")
    lo, hi = 0, len(lines)
    while lo < hi:
        mid = (lo + hi) // 2
        if count_tokens("\n".join(lines[:mid])) <= MAX_DIFF_TOKENS:
            lo = mid + 1
        else:
            hi = mid

    cut = "\n".join(lines[: lo - 1])
    notice = (
        "\n\n[DIFF TRUNCATED — only the first ~3,000 tokens were analyzed. "
        "Consider breaking this PR into smaller, focused changes.]"
    )
    final = cut + notice
    return final, True, count_tokens(final)


def parse_diff_stats(diff: str) -> tuple[int, int, int]:
    """Extract summary statistics from a unified diff.

    Returns:
        (files_changed, lines_added, lines_removed)
    """
    files = set(re.findall(r"^\+\+\+ b/(.+)$", diff, re.MULTILINE))
    added = sum(
        1 for ln in diff.split("\n")
        if ln.startswith("+") and not ln.startswith("+++")
    )
    removed = sum(
        1 for ln in diff.split("\n")
        if ln.startswith("-") and not ln.startswith("---")
    )
    return len(files), added, removed


def parse_diff_hunks(diff: str) -> dict[str, set[int]]:
    """Parse a unified diff and return the set of new-file line numbers per file.

    Only lines that appear as additions (+) in the diff are included — these
    are the only positions the GitHub Reviews API accepts for inline comments
    with side=RIGHT.
    """
    file_lines: dict[str, set[int]] = {}
    current_file: str | None = None
    new_line_num: int = 0

    for line in diff.split("\n"):
        if line.startswith("diff --git "):
            m = re.match(r"^diff --git a/.+ b/(.+)$", line)
            if m:
                current_file = m.group(1)
                file_lines.setdefault(current_file, set())
                new_line_num = 0
        elif line.startswith("@@") and current_file:
            m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if m:
                new_line_num = int(m.group(1)) - 1
        elif current_file:
            if line.startswith("+") and not line.startswith("+++"):
                new_line_num += 1
                file_lines[current_file].add(new_line_num)
            elif not line.startswith("-") and not line.startswith("\\"):
                new_line_num += 1

    return file_lines


# ---------------------------------------------------------------------------
# CODEOWNERS
# ---------------------------------------------------------------------------


def load_codeowners_patterns(repo_root: str = ".") -> list[str]:
    """Read ownership patterns from CODEOWNERS, if present.

    Checks the three standard locations GitHub recognises.
    Returns a list of path patterns (leading slash stripped).
    """
    candidates = [
        os.path.join(repo_root, "CODEOWNERS"),
        os.path.join(repo_root, ".github", "CODEOWNERS"),
        os.path.join(repo_root, "docs", "CODEOWNERS"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            patterns: list[str] = []
            with open(path, encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if raw and not raw.startswith("#"):
                        patterns.append(raw.split()[0].lstrip("/"))
            log.info("Loaded %d CODEOWNERS pattern(s) from %s.", len(patterns), path)
            return patterns
    return []


def apply_codeowners_boost(
    comments: list[ReviewComment],
    patterns: list[str],
) -> list[ReviewComment]:
    """Escalate severity by one level for findings in CODEOWNERS-tracked files.

    suggestion -> warning -> critical. Critical stays critical.
    """
    if not patterns:
        return comments

    order = ["suggestion", "warning", "critical"]
    boosted: list[ReviewComment] = []

    for c in comments:
        sev = c.severity
        for pattern in patterns:
            if fnmatch.fnmatch(c.file, pattern) or fnmatch.fnmatch(
                c.file, f"**/{pattern}"
            ):
                idx = order.index(sev) if sev in order else 0
                sev = order[min(idx + 1, len(order) - 1)]
                log.debug(
                    "CODEOWNERS boost: %s:%d  %s -> %s",
                    c.file, c.line, c.severity, sev,
                )
                break
        boosted.append(
            ReviewComment(
                file=c.file,
                line=c.line,
                severity=sev,
                title=c.title,
                suggestion=c.suggestion,
                why=c.why,
            )
        )

    return boosted


# ---------------------------------------------------------------------------
# Groq interaction
# ---------------------------------------------------------------------------


def _call_model(
    client: Groq,
    diff: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
) -> dict[str, Any]:
    """Call a single Groq model with exponential backoff on rate limits."""
    user_content = f"Review the following git diff:\n\n```diff\n{diff}\n```"
    delays = [0] + RETRY_DELAYS

    last_exc: Exception | None = None
    for attempt, delay in enumerate(delays, start=1):
        if delay:
            log.info(
                "Rate-limit backoff: waiting %ds before attempt %d/%d.",
                delay, attempt, len(delays),
            )
            time.sleep(delay)

        try:
            log.info(
                "Groq request: model=%s  attempt=%d/%d", model, attempt, len(delays)
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            raw: str = response.choices[0].message.content
            log.debug("Response preview: %.300s", raw)
            return json.loads(raw)

        except APIStatusError as exc:
            last_exc = exc
            if exc.status_code == 429 and attempt < len(delays):
                log.warning("Groq 429 on attempt %d/%d. Will retry.", attempt, len(delays))
                continue
            raise

        except APIConnectionError as exc:
            last_exc = exc
            if attempt < len(delays):
                log.warning(
                    "Connection error on attempt %d/%d: %s. Will retry.",
                    attempt, len(delays), exc,
                )
                continue
            raise

        except (json.JSONDecodeError, ValueError) as exc:
            log.error("Failed to parse model response as JSON: %s", exc)
            raise

    raise RuntimeError("Exhausted all retry attempts.") from last_exc


def call_groq(
    client: Groq,
    diff: str,
    primary_model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
) -> tuple[dict[str, Any], str]:
    """Call Groq with an automatic model fallback chain.

    Tries primary_model first, then each entry in FALLBACK_MODELS in order,
    skipping any model Groq reports as decommissioned or not found.

    Returns:
        (parsed_response_dict, model_that_succeeded)
    """
    chain = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]

    for model in chain:
        try:
            payload = _call_model(
                client, diff, model, temperature, max_tokens, system_prompt
            )
            log.info("Review completed with model: %s", model)
            return payload, model
        except APIStatusError as exc:
            decommissioned = exc.status_code == 400 and any(
                kw in str(exc).lower()
                for kw in ("decommissioned", "model_not_found", "not found")
            )
            if decommissioned:
                log.warning(
                    "Model %s is unavailable (HTTP %d). Trying next in chain.",
                    model, exc.status_code,
                )
                continue
            raise

    raise RuntimeError(f"All models in fallback chain exhausted: {chain}")


def parse_review(payload: dict[str, Any]) -> ReviewResult:
    """Coerce the raw model output into a typed ReviewResult.

    Malformed individual entries are logged and skipped rather than crashing
    the entire review run.
    """
    result = ReviewResult(summary=payload.get("summary", ""))
    valid_severities = {"critical", "warning", "suggestion"}

    for item in payload.get("comments", []):
        try:
            severity = str(item.get("severity", "suggestion")).lower()
            if severity not in valid_severities:
                log.debug("Normalising unknown severity %r to 'suggestion'.", severity)
                severity = "suggestion"

            result.comments.append(
                ReviewComment(
                    file=str(item["file"]),
                    line=int(item.get("line", 0)),
                    severity=severity,
                    title=str(item.get("title", "(no title)")),
                    suggestion=str(item.get("suggestion", "")),
                    why=str(item.get("why", "")),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("Discarding malformed comment entry %r: %s", item, exc)

    return result


# ---------------------------------------------------------------------------
# GitHub REST API helpers
# ---------------------------------------------------------------------------


def _gh_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    return s


_GH_API = "https://api.github.com"


def post_pr_review(
    session: requests.Session,
    repo: str,
    pull_number: int,
    commit_sha: str,
    body: str,
    inline_comments: list[dict[str, Any]],
) -> None:
    """Submit a pull request review via the GitHub Reviews API.

    If inline comments cause a 422 (line not present in the diff), the request
    is retried without them so the summary body is never silently lost.
    """
    url = f"{_GH_API}/repos/{repo}/pulls/{pull_number}/reviews"
    payload: dict[str, Any] = {
        "commit_id": commit_sha,
        "body":      body,
        "event":     "COMMENT",
    }
    if inline_comments:
        payload["comments"] = inline_comments

    resp = session.post(url, json=payload)
    if resp.status_code in (200, 201):
        log.info("Review posted with %d inline comment(s).", len(inline_comments))
        return

    if resp.status_code == 422 and inline_comments:
        log.warning(
            "Review rejected with inline comments (422); retrying without them. "
            "API: %s",
            resp.text[:400],
        )
        payload.pop("comments")
        resp = session.post(url, json=payload)
        if resp.status_code in (200, 201):
            log.info("Summary-only review posted.")
            return

    log.error(
        "Failed to post review: HTTP %d — %s",
        resp.status_code, resp.text[:400],
    )


def post_issue_comment(
    session: requests.Session,
    repo: str,
    issue_number: int,
    body: str,
) -> None:
    """Post a plain issue/PR comment. Used as a last-resort fallback."""
    url = f"{_GH_API}/repos/{repo}/issues/{issue_number}/comments"
    resp = session.post(url, json={"body": body})
    if resp.status_code == 201:
        log.info("Fallback issue comment posted.")
    else:
        log.error(
            "Failed to post issue comment: HTTP %d — %s",
            resp.status_code, resp.text[:200],
        )


# ---------------------------------------------------------------------------
# Review formatting
# ---------------------------------------------------------------------------


def build_review_body(
    result: ReviewResult,
    was_truncated: bool,
    token_count: int,
    model: str,
    files_changed: int,
    lines_added: int,
    lines_removed: int,
    skipped_files: list[str],
) -> str:
    """Compose the top-level review body from the structured result."""
    parts: list[str] = ["## RaptorReview AI", ""]

    if was_truncated:
        parts += [
            "> **Note:** The diff exceeded the review budget (~3,000 tokens) and "
            "was truncated. Only the first portion of this PR was analyzed.",
            "",
        ]

    if skipped_files:
        skipped_display = ", ".join(f"`{f}`" for f in skipped_files[:5])
        if len(skipped_files) > 5:
            skipped_display += f" and {len(skipped_files) - 5} more"
        parts += [
            f"> **Skipped {len(skipped_files)} noise file(s):** {skipped_display}",
            "",
        ]

    if result.summary:
        parts += [result.summary, ""]

    if result.comments:
        parts += ["---", "", f"**{len(result.comments)} finding(s):**", ""]
        for c in result.comments:
            label = SEVERITY_LABELS.get(c.severity, c.severity.upper())
            line_ref = f":{c.line}" if c.line > 0 else ""
            parts.append(f"- {label} **{c.title}** — `{c.file}{line_ref}`")
        parts.append("")

    parts += [
        "---",
        f"Reviewed **{files_changed} file(s)** &nbsp;·&nbsp; "
        f"`+{lines_added} -{lines_removed}` &nbsp;·&nbsp; "
        f"Model: `{model}` &nbsp;·&nbsp; "
        f"Tokens: ~{token_count:,}",
        "",
        "_Posted by [RaptorReview AI](https://github.com/dev-k99/raptorreview-ai)_",
    ]
    return "\n".join(parts)


def build_inline_comments(
    result: ReviewResult,
    diff: str,
) -> list[dict[str, Any]]:
    """Convert ReviewComment objects into GitHub Reviews API comment dicts.

    Uses the parsed diff hunk map to validate line numbers. When the LLM
    returns a line close to (but not exactly on) a changed line, the comment
    is snapped to the nearest valid line within a 5-line tolerance.
    """
    diff_line_map = parse_diff_hunks(diff)
    inline: list[dict[str, Any]] = []

    for c in result.comments:
        if c.line <= 0:
            continue

        valid_lines = diff_line_map.get(c.file)
        if not valid_lines:
            log.debug("Skipping inline comment for %r (file not in diff hunks).", c.file)
            continue

        if c.line in valid_lines:
            target_line = c.line
        else:
            closest = min(valid_lines, key=lambda x: abs(x - c.line))
            if abs(closest - c.line) > 5:
                log.debug(
                    "Skipping inline comment for %r:%d "
                    "(nearest diff line %d exceeds tolerance).",
                    c.file, c.line, closest,
                )
                continue
            log.debug(
                "Snapping inline comment %r:%d -> diff line %d.",
                c.file, c.line, closest,
            )
            target_line = closest

        inline.append(
            {
                "path": c.file,
                "line": target_line,
                "side": "RIGHT",
                "body": c.to_markdown(),
            }
        )

    return inline


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    # ------------------------------------------------------------------
    # Read configuration from environment
    # ------------------------------------------------------------------
    groq_api_key  = os.environ.get("GROQ_API_KEY",         "").strip()
    github_token  = os.environ.get("GITHUB_TOKEN",          "").strip()
    repo          = os.environ.get("GITHUB_REPOSITORY",     "")
    event_path    = os.environ.get("GITHUB_EVENT_PATH",     "")
    model         = os.environ.get("INPUT_MODEL",           "llama-3.3-70b-versatile").strip()
    temperature   = float(os.environ.get("INPUT_TEMPERATURE", "0.2"))
    max_tokens    = int(os.environ.get("INPUT_MAX_TOKENS",    "2048"))
    custom_prompt = os.environ.get("INPUT_CUSTOM_PROMPT",   "").strip() or None

    # ------------------------------------------------------------------
    # Validate required secrets
    # ------------------------------------------------------------------
    missing = [
        name
        for name, val in [("GROQ_API_KEY", groq_api_key), ("GITHUB_TOKEN", github_token)]
        if not val
    ]
    if missing:
        log.error(
            "Required environment variable(s) not set: %s. "
            "Configure them as repository secrets.",
            ", ".join(missing),
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load PR event payload
    # ------------------------------------------------------------------
    if not event_path or not os.path.isfile(event_path):
        log.error("GITHUB_EVENT_PATH not found: %r", event_path)
        sys.exit(1)

    with open(event_path, encoding="utf-8") as fh:
        event = json.load(fh)

    pr = event.get("pull_request")
    if not pr:
        log.info("Event payload has no 'pull_request' key — nothing to review.")
        sys.exit(0)

    pull_number: int = pr["number"]
    head_sha: str    = pr["head"]["sha"]
    base_ref: str    = pr["base"]["ref"]

    log.info(
        "PR #%d  head=%s  base_ref=%s  repo=%s",
        pull_number, head_sha[:8], base_ref, repo,
    )

    # ------------------------------------------------------------------
    # Acquire, filter, and truncate the diff
    # ------------------------------------------------------------------
    raw_diff = get_diff(base_ref)
    if not raw_diff.strip():
        log.info("Empty diff — no review needed.")
        sys.exit(0)

    filtered_diff, skipped_files = filter_diff(raw_diff)
    if skipped_files:
        log.info("Filtered out %d noise file(s): %s", len(skipped_files), skipped_files)

    if not filtered_diff.strip():
        log.info("Diff is empty after filtering noise files — no review needed.")
        sys.exit(0)

    diff, was_truncated, token_count = truncate_diff(filtered_diff)
    files_changed, lines_added, lines_removed = parse_diff_stats(diff)
    log.info(
        "Diff: %d tokens  %d file(s)  +%d -%d%s",
        token_count, files_changed, lines_added, lines_removed,
        "  (truncated)" if was_truncated else "",
    )

    # ------------------------------------------------------------------
    # Load CODEOWNERS for severity boosting
    # ------------------------------------------------------------------
    codeowners_patterns = load_codeowners_patterns()

    # ------------------------------------------------------------------
    # Call Groq and parse the structured response
    # ------------------------------------------------------------------
    gh            = _gh_session(github_token)
    groq_client   = Groq(api_key=groq_api_key)
    system_prompt = custom_prompt or DEFAULT_SYSTEM_PROMPT

    try:
        payload, model_used = call_groq(
            client=groq_client,
            diff=diff,
            primary_model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        review = parse_review(payload)

        if codeowners_patterns:
            review.comments = apply_codeowners_boost(
                review.comments, codeowners_patterns
            )

        log.info(
            "Review parsed: %d comment(s), summary %d chars, model %s.",
            len(review.comments), len(review.summary), model_used,
        )

    except Exception as exc:  # noqa: BLE001
        log.exception("Groq review failed: %s", exc)
        fallback_body = (
            "## RaptorReview AI\n\n"
            "The automated review could not be completed due to an API error.\n\n"
            f"**Error:** `{type(exc).__name__}: {exc}`\n\n"
            "**Troubleshooting:**\n"
            "- Verify that `GROQ_API_KEY` is set and has remaining quota at "
            "[console.groq.com](https://console.groq.com).\n"
            "- As a fallback, you can swap the API call in `src/review.py` to target "
            "the [Hugging Face Inference API](https://huggingface.co/inference-api).\n\n"
            "_Review skipped. CI pipeline is unblocked._"
        )
        post_issue_comment(gh, repo, pull_number, fallback_body)
        sys.exit(0)

    # ------------------------------------------------------------------
    # Post results to GitHub
    # ------------------------------------------------------------------
    review_body = build_review_body(
        result=review,
        was_truncated=was_truncated,
        token_count=token_count,
        model=model_used,
        files_changed=files_changed,
        lines_added=lines_added,
        lines_removed=lines_removed,
        skipped_files=skipped_files,
    )
    inline_cmts = build_inline_comments(review, raw_diff)

    log.info("Posting review (%d inline comment(s)).", len(inline_cmts))
    post_pr_review(
        session=gh,
        repo=repo,
        pull_number=pull_number,
        commit_sha=head_sha,
        body=review_body,
        inline_comments=inline_cmts,
    )


if __name__ == "__main__":
    main()
