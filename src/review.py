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

# ~3 000 tokens at average 4 chars/token.  Keeps us well inside free-tier
# context limits while still covering the majority of real-world PRs.
MAX_DIFF_CHARS: int = 12_000

# Delays (seconds) between retries when Groq returns HTTP 429.
RETRY_DELAYS: list[int] = [2, 5, 15]

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
      "line": <integer line number in the new version of the file>,
      "severity": "<critical|warning|suggestion>",
      "title": "<concise issue title, max 80 chars>",
      "suggestion": "<concrete fix or alternative — code snippet preferred>",
      "why": "<explanation of impact>"
    }
  ],
  "summary": "<overall review paragraph>"
}

Set "line" to 0 if a comment does not map to a specific line. The "comments" \
array may be empty if the diff is clean. Never fabricate issues that are not \
present in the diff."""


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
# Diff acquisition
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


def truncate_diff(diff: str) -> tuple[str, bool]:
    """Return (diff, was_truncated).

    When the diff exceeds MAX_DIFF_CHARS, it is cut at a line boundary so the
    model never receives a half-formed hunk.
    """
    if len(diff) <= MAX_DIFF_CHARS:
        return diff, False

    cut = diff[:MAX_DIFF_CHARS]
    last_nl = cut.rfind("\n")
    if last_nl > 0:
        cut = cut[:last_nl]

    notice = (
        "\n\n[DIFF TRUNCATED — only the first ~3 000 tokens were analyzed. "
        "Consider breaking this PR into smaller, focused changes.]"
    )
    return cut + notice, True


# ---------------------------------------------------------------------------
# Groq interaction
# ---------------------------------------------------------------------------


def call_groq(
    client: Groq,
    diff: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
) -> dict[str, Any]:
    """Submit the diff to Groq and return the parsed JSON response dict.

    Retries automatically on HTTP 429 (rate limit) and transient connection
    errors, using the delays defined in RETRY_DELAYS.  Raises on all other
    API errors or if all attempts are exhausted.
    """
    user_content = f"Review the following git diff:\n\n```diff\n{diff}\n```"
    delays = [0] + RETRY_DELAYS  # first attempt has no delay

    last_exc: Exception | None = None
    for attempt, delay in enumerate(delays, start=1):
        if delay:
            log.info(
                "Rate-limit backoff: waiting %ds before attempt %d/%d.",
                delay, attempt, len(delays),
            )
            time.sleep(delay)

        try:
            log.info("Groq request: model=%s  attempt=%d/%d", model, attempt, len(delays))
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
                log.warning("Groq returned 429 on attempt %d. Will retry.", attempt)
                continue
            raise

        except APIConnectionError as exc:
            last_exc = exc
            if attempt < len(delays):
                log.warning("Groq connection error on attempt %d: %s. Will retry.", attempt, exc)
                continue
            raise

        except (json.JSONDecodeError, ValueError) as exc:
            # response_format=json_object should make this extremely rare,
            # but surface it clearly if it happens.
            log.error("Failed to parse model response as JSON: %s", exc)
            raise

    raise RuntimeError("Exhausted all Groq retry attempts.") from last_exc


def parse_review(payload: dict[str, Any]) -> ReviewResult:
    """Coerce the raw model output into a typed ReviewResult.

    Malformed individual comment entries are logged and skipped rather than
    crashing the entire review run.
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

    If inline comments cause a 422 (e.g. a line number that is not present in
    the diff), the request is retried without inline comments so the overall
    summary is never silently lost.
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
            "Review rejected with inline comments (422). "
            "Retrying with summary body only. API response: %s",
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


def build_review_body(result: ReviewResult, was_truncated: bool) -> str:
    """Compose the top-level review body from the structured result."""
    parts: list[str] = ["## RaptorReview AI", ""]

    if was_truncated:
        parts += [
            "> **Note:** The diff exceeded the review budget (~3 000 tokens) and "
            "was truncated. Only the first portion of this PR was analyzed.",
            "",
        ]

    if result.summary:
        parts += [result.summary, ""]

    if result.comments:
        parts += ["---", "", f"**{len(result.comments)} finding(s):**", ""]
        for c in result.comments:
            label = SEVERITY_LABELS.get(c.severity, c.severity.upper())
            parts.append(f"- {label} **{c.title}** — `{c.file}:{c.line}`")
        parts.append("")

    parts += [
        "---",
        "_Posted by [RaptorReview AI](https://github.com/raptorreview-ai/raptorreview-ai)_",
    ]
    return "\n".join(parts)


def build_inline_comments(
    result: ReviewResult,
    diff: str,
) -> list[dict[str, Any]]:
    """Convert ReviewComment objects to GitHub Reviews API comment dicts.

    Only includes comments where line > 0 and the file is present in the diff.
    Comments that cannot be placed inline are captured in the review body
    summary list instead.
    """
    diff_files: set[str] = set(re.findall(r"^\+\+\+ b/(.+)$", diff, re.MULTILINE))

    inline: list[dict[str, Any]] = []
    for c in result.comments:
        if c.line <= 0:
            continue
        if c.file not in diff_files:
            log.debug("Skipping inline comment for %r (file not in diff).", c.file)
            continue
        inline.append(
            {
                "path": c.file,
                "line": c.line,
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
    model         = os.environ.get("INPUT_MODEL",           "llama-3.1-70b-versatile").strip()
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
    # Acquire and condition the diff
    # ------------------------------------------------------------------
    raw_diff = get_diff(base_ref)
    if not raw_diff.strip():
        log.info("Empty diff — no review needed.")
        sys.exit(0)

    diff, was_truncated = truncate_diff(raw_diff)
    log.info(
        "Diff: %d chars%s",
        len(diff), " (truncated)" if was_truncated else "",
    )

    # ------------------------------------------------------------------
    # Call Groq and parse the structured response
    # ------------------------------------------------------------------
    gh          = _gh_session(github_token)
    groq_client = Groq(api_key=groq_api_key)
    system_prompt = custom_prompt or DEFAULT_SYSTEM_PROMPT

    try:
        payload = call_groq(
            client=groq_client,
            diff=diff,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        review = parse_review(payload)
        log.info(
            "Review parsed: %d comment(s), summary %d chars.",
            len(review.comments), len(review.summary),
        )

    except Exception as exc:  # noqa: BLE001 — intentional broad catch; errors here must not block CI
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
    review_body  = build_review_body(review, was_truncated)
    inline_cmts  = build_inline_comments(review, raw_diff)

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
