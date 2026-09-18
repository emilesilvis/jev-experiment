"""A few shared constants and JSON/HTTP helpers."""

import json
from pathlib import Path

import requests


# These descriptions are experiment choices, shared by both classifiers.
TEAMS = {
    "team-framework": "Flutter's Dart UI framework: widgets, layout, painting, gestures, and animation.",
    "team-tool": "Flutter developer tools: flutter CLI, project creation, builds, hot reload, and tooling.",
    "team-ios": "Flutter's iOS integration: iOS embedding, platform behavior, and iOS-specific problems.",
    "team-android": "Flutter's Android integration: Android embedding, platform behavior, and Android-specific problems.",
}


def validate_task(task):
    """Check a frozen classification rubric before it reaches either provider."""
    if not isinstance(task, dict) or set(task) != {"name", "instructions", "labels"}:
        raise ValueError("Task must contain name, instructions, and labels.")
    if any(not isinstance(task[key], str) or not task[key].strip()
           for key in ("name", "instructions")):
        raise ValueError("Task name and instructions must be nonempty strings.")
    labels = task["labels"]
    if not isinstance(labels, dict) or len(labels) < 2:
        raise ValueError("Task must define at least two labels.")
    if any(not isinstance(label, str) or not label.strip()
           or not isinstance(description, str) or not description.strip()
           for label, description in labels.items()):
        raise ValueError("Label names and descriptions must be nonempty strings.")


def request_json(method, url, **kwargs):
    """One request, no hidden retries (which would change latency and cost)."""
    try:
        response = requests.request(method, url, timeout=60, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        status = exc.response.status_code
        hint = " Check API credentials." if status in (401, 403) else ""
        if status in (429, 529):
            hint = " Rate limited; wait before rerunning."
        # Do not log request headers, keys, or arbitrary provider response bodies.
        raise RuntimeError(f"HTTP {status} from {url.split('?')[0]}.{hint}") from None
    except (requests.exceptions.JSONDecodeError, ValueError):
        raise RuntimeError(f"Invalid JSON from {url.split('?')[0]}.") from None
    except requests.RequestException:
        raise RuntimeError(f"Network error calling {url.split('?')[0]}; try again later.") from None


def write_json(path, value):
    """Replace the file only after the complete JSON has been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
