import unittest
from datetime import date
from unittest.mock import patch

from fetch_issues import fetch_team_issues, make_model_input, sanitize_text, target_for_issue


def issue(number, labels, created="2026-09-10T12:00:00Z", **extra):
    return {
        "number": number, "title": f"Widget fails on iOS #{number}",
        "body": "Android build fails", "labels": [{"name": name} for name in labels],
        "created_at": created, "updated_at": "2026-09-18T12:00:00Z", **extra,
    }


class FetchTests(unittest.TestCase):
    def test_exactly_one_target_and_no_pull_requests(self):
        self.assertEqual(target_for_issue(issue(1, ["team-framework", "a: quality"])),
                         "team-framework")
        self.assertIsNone(target_for_issue(issue(2, ["team-framework", "team-tool"])))
        self.assertIsNone(target_for_issue(issue(3, ["bug"])))
        self.assertIsNone(target_for_issue(issue(4, ["team-ios"], pull_request={})))

    def test_sanitization_removes_identifiers_and_preserves_bug_terms(self):
        source = """Widget build fails on iOS and Android.
Labels: team-framework, a: quality
**Team:** iOS
This issue is owned by the framework team.
@flutter/team-ios owns this. team-android p:firebase_auth a:crash
See flutter/flutter#123, #456, issue 789 and issue #900.
Read [Widget documentation](https://example.com/123) or https://github.com/foo/bar/issues/42.
<!-- assigned team-tool -->
<!-- unfinished comment team-framework"""
        clean = sanitize_text(source)
        self.assertIn("Widget build fails on iOS and Android.", clean)
        self.assertIn("Widget documentation", clean)
        for leakage in ["team-", "@flutter", "firebase_auth", "a:crash", "123", "456",
                        "789", "900", "https://", "Labels:", "assigned", "unfinished",
                        "**Team", "owned by"]:
            self.assertNotIn(leakage, clean)

    def test_input_uses_sanitized_title_and_body_and_has_fixed_cap(self):
        clean = make_model_input("team-tool #42: Build error", "Labels: team-ios\n" + "x" * 100, 60)
        self.assertEqual(len(clean), 60)
        self.assertIn("Build error", clean)
        self.assertNotIn("team-", clean)
        self.assertNotIn("42", clean)

    def test_sanitization_preserves_code_indentation(self):
        source = "    config:\n      child: Widget()  \n\n\n\n    return child\t\n"
        self.assertEqual(sanitize_text(source),
                         "    config:\n      child: Widget()\n\n    return child")

    def test_sanitization_handles_encoded_comments_urls_and_google_issue_refs(self):
        source = ("&lt;!-- hidden team assignment --&gt;\n"
                  "See HTTPS://EXAMPLE.COM/1 and gs://bucket/logs or GitHub.COM/foo/bar.\n"
                  "Related b/123456 and B/789. &lt;!-- unclosed hidden comment")
        clean = sanitize_text(source)
        self.assertIn("See", clean)
        self.assertIn("Related", clean)
        for leakage in ["hidden", "EXAMPLE", "bucket", "GitHub", "123456", "789"]:
            self.assertNotIn(leakage, clean)

    @patch("fetch_issues.request_json")
    def test_selects_latest_eligible_created_issues_within_window(self, request):
        request.return_value = [
            issue(9, ["team-ios"], "2026-09-19T01:00:00Z"),
            issue(8, ["team-ios"], pull_request={}),
            issue(7, ["team-ios", "team-tool"]),
            issue(6, ["team-ios", "bug"]),
            issue(5, ["team-ios"]),
            issue(4, ["team-ios"], "2026-08-31T23:59:59Z"),
        ]
        selected = fetch_team_issues("team-ios", 2, date(2026, 9, 1), date(2026, 9, 18), 1000, {})
        self.assertEqual([item["number"] for item in selected], [6, 5])
        self.assertEqual(selected[0]["labels"], ["team-ios", "bug"])
        self.assertIn("#6", selected[0]["title"])  # Originals stay auditable.
        self.assertNotIn("#6", selected[0]["model_input"])
        self.assertEqual(request.call_args.kwargs["params"]["sort"], "created")

    @patch("fetch_issues.request_json")
    def test_shortage_fails_instead_of_returning_partial_sample(self, request):
        request.return_value = [issue(1, ["team-tool"])]
        with self.assertRaisesRegex(ValueError, "Only 1 eligible"):
            fetch_team_issues("team-tool", 2, date(2026, 9, 1), date(2026, 9, 18), 1000, {})


if __name__ == "__main__":
    unittest.main()
