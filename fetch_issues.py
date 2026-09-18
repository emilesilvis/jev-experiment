"""Freeze a small, balanced Flutter issue dataset from the GitHub REST API."""

import html
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from common import TEAMS, request_json, write_json


def sanitize_text(text):
    """Apply the same leakage-removal rules regardless of the issue's target."""
    text = html.unescape(text or "")
    text = re.sub(r"<!--.*?(?:-->|$)", "", text, flags=re.DOTALL)
    # Keep link text, which often describes the bug, but never the destination.
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(
        r"\b(?:[a-z][a-z0-9+.-]*://[^\s<>]+|www\.[^\s<>]+|github\.com(?:/[^\s<>]*)?)",
        "", text, flags=re.IGNORECASE,
    )
    # Explicit metadata and ownership statements are not part of the bug report.
    text = re.sub(
        r"^\s*[#>*_-]*\s*[\"']?(?:(?:github|issue|current|suggested|expected|applied)\s+)?"
        r"(?:labels?|owning team|assigned team|owner|team)[\"'*_]*\s*[:=].*$",
        "", text, flags=re.IGNORECASE | re.MULTILINE,
    )
    text = re.sub(
        r"^(?=[^\n]*\b(?:framework|tools?|ios|android)\b)[^\n]*"
        r"(?:\b(?:owned|triaged|assigned|routed)\s+(?:by|to)|\bbelongs?\s+to)[^\n]*$",
        "", text, flags=re.IGNORECASE | re.MULTILINE,
    )
    text = re.sub(r"@flutter/[\w-]+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bteam[-_ ](?:framework|tool|ios|android)\b", "", text,
                  flags=re.IGNORECASE)
    text = re.sub(r"\b(?:framework|tools?|ios|android)\s+team\b", "", text,
                  flags=re.IGNORECASE)
    # Common Flutter label prefixes; avoid stripping arbitrary code properties.
    text = re.sub(r"\b(?:a|p|c|e|f|r|t|d):\s*[\w.-]+", "", text,
                  flags=re.IGNORECASE)
    text = re.sub(r"\bissues?\s+(?:no\.?\s*|number\s*)?#?\d+\b", "", text,
                  flags=re.IGNORECASE)
    text = re.sub(r"(?:(?:[\w.-]+/)?[\w.-]+)?#\d+\b", "", text)
    text = re.sub(r"\bb/\d+\b", "", text, flags=re.IGNORECASE)
    # Preserve indentation in code/logs, while tidying trailing spaces and gaps.
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")


def make_model_input(title, body, max_chars):
    return (f"Title: {sanitize_text(title)}\n\n"
            f"Description:\n{sanitize_text(body)}")[:max_chars]


def target_for_issue(issue):
    """Other labels are allowed, but precisely one of our four teams must match."""
    if "pull_request" in issue:
        return None
    labels = {label["name"] for label in issue.get("labels", [])}
    matches = labels.intersection(TEAMS)
    return next(iter(matches)) if len(matches) == 1 else None


def fetch_team_issues(team, count, start_date, end_date, max_chars, headers):
    selected = []
    seen = set()
    page = 1
    while len(selected) < count:
        batch = request_json(
            "GET", "https://api.github.com/repos/flutter/flutter/issues",
            headers=headers,
            params={
                "state": "all", "labels": team, "sort": "created",
                "direction": "desc", "per_page": 100, "page": page,
                # GitHub's `since` filters updated_at; created_at is checked below.
                "since": f"{start_date.isoformat()}T00:00:00Z",
            },
        )
        if not isinstance(batch, list):
            raise ValueError("GitHub returned an unexpected issue response.")
        if not batch:
            break
        reached_start = False
        for issue in batch:
            created = date.fromisoformat(issue["created_at"][:10])
            if created < start_date:
                reached_start = True
                break  # The API returns issues in descending creation order.
            if created > end_date or target_for_issue(issue) != team:
                continue
            if issue["number"] in seen:
                continue
            seen.add(issue["number"])
            selected.append({
                "number": issue["number"],
                "title": issue["title"],
                "body": issue.get("body") or "",
                "labels": [label["name"] for label in issue["labels"]],
                "created_at": issue["created_at"],
                "updated_at": issue["updated_at"],
                "target": team,
                "model_input": make_model_input(
                    issue["title"], issue.get("body"), max_chars),
            })
            if len(selected) == count:
                break
        if reached_start or len(batch) < 100:
            break
        page += 1
    if len(selected) != count:
        raise ValueError(
            f"Only {len(selected)} eligible {team} issues in "
            f"{start_date} through {end_date}; need {count}. "
            "Widen SINCE_DATE or lower ISSUES_PER_TEAM. No dataset was saved."
        )
    return selected


def main():
    output = Path(os.getenv("DATA_DIR", "data")) / "issues.json"
    if output.exists():
        raise ValueError(f"{output} already exists. Use another DATA_DIR for a new snapshot.")
    now = datetime.now(timezone.utc)
    count = int(os.getenv("ISSUES_PER_TEAM", "50"))
    max_chars = int(os.getenv("MAX_INPUT_CHARS", "12000"))
    start = date.fromisoformat(os.getenv("SINCE_DATE", str(now.date() - timedelta(days=180))))
    end = date.fromisoformat(os.getenv("UNTIL_DATE", str(now.date())))
    if count < 1 or max_chars < 100 or start > end:
        raise ValueError("Require ISSUES_PER_TEAM > 0, MAX_INPUT_CHARS >= 100, and SINCE_DATE <= UNTIL_DATE.")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    issues = []
    for team in TEAMS:
        print(f"Fetching {count} {team} issues...", flush=True)
        issues.extend(fetch_team_issues(team, count, start, end, max_chars, headers))
    if len({issue["number"] for issue in issues}) != len(issues):
        raise ValueError("An issue changed teams during fetching. Rerun to get a consistent sample.")
    issues.sort(key=lambda issue: issue["number"])
    write_json(output, {
        "metadata": {
            "repository": "flutter/flutter",
            "fetched_at": now.isoformat(),
            "since_date": start.isoformat(),
            "until_date": end.isoformat(),
            "issues_per_team": count,
            "max_input_chars": max_chars,
            "selection": "Newest created issues per team, inclusive UTC creation dates; "
                         "exclude pull requests and issues with multiple target labels.",
            "sanitization_version": 2,
            "sanitization": "Identical title/body rules remove HTML comments, URLs, issue "
                            "references, explicit label metadata, target team label names, "
                            "Flutter team handles and common structured label tokens. "
                            "Combined input is truncated after sanitization.",
        },
        "issues": issues,
    })
    print(f"Saved {len(issues)} issues to {output}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from None
