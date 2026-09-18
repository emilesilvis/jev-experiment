import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_data import (TASKS, balanced_sample, deduplicate, example,
                            fetch_source, freeze_bytes, parse_ag_news, parse_snips,
                            parse_trec, prepare_dataset)


class ParsingTests(unittest.TestCase):
    def test_ag_news_maps_all_classes_and_preserves_article_text(self):
        data = ('1,"Title, with comma","Quoted ""news""\\nMore #42 https://example.com"\n'
                '2,"Sports headline","Sports detail"\n'
                '3,"Business headline","Business detail"\n'
                '4,"Tech headline","Tech detail"\n').encode()
        rows = parse_ag_news(data)
        self.assertEqual([row["target"] for row in rows], list(TASKS["ag_news"]["labels"]))
        self.assertEqual(rows[0]["model_input"],
                         'Title, with comma\n\nQuoted "news"\nMore #42 https://example.com')
        self.assertEqual(rows[0]["source_id"], "test.csv:1")
        self.assertNotIn("World", rows[0]["model_input"])
        with self.assertRaisesRegex(ValueError, "Invalid AG News"):
            parse_ag_news(b'0,"Title","Description"')

    def test_trec_removes_fine_and_coarse_labels_only(self):
        data = b'ABBR:exp What does SQL stand for ?\nNUM:date When was he born ?\n'
        rows = parse_trec(data)
        self.assertEqual([row["target"] for row in rows], ["ABBR", "NUM"])
        self.assertEqual(rows[0]["model_input"], "What does SQL stand for ?")
        self.assertEqual(rows[1]["number"], 2)
        with self.assertRaisesRegex(ValueError, "Unknown TREC"):
            parse_trec(b'WRONG:exp What is this ?')

    def test_snips_reassembles_text_without_entities_or_intent(self):
        data = json.dumps({"PlayMusic": [{"data": [
            {"text": "Play "}, {"text": "Beyoncé", "entity": "artist"},
            {"text": " on "}, {"text": "Spotify", "entity": "service"}]}]}).encode()
        row = parse_snips(data, "PlayMusic")[0]
        self.assertEqual(row["model_input"], "Play Beyoncé on Spotify")
        self.assertEqual(row["target"], "PlayMusic")
        self.assertEqual(row["number"], 4000001)
        other = json.dumps({"GetWeather": [{"data": [{"text": "Weather?"}]}]}).encode()
        self.assertNotEqual(row["number"], parse_snips(other, "GetWeather")[0]["number"])
        with self.assertRaisesRegex(ValueError, "Unexpected SNIPS"):
            parse_snips(data, "GetWeather")

    def test_empty_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Empty input"):
            example(1, " \n", "World", "row1")


class SamplingTests(unittest.TestCase):
    def rows(self, labels=("a", "b"), per_class=10):
        return [example(i * per_class + j, f"{label} text{j}", label, f"{label}:{j}")
                for i, label in enumerate(labels) for j in range(1, per_class + 1)]

    def test_balanced_sampling_has_exact_counts_and_repeatable_selection(self):
        rows = self.rows()
        selected = balanced_sample(rows, ("a", "b"), 4, 17)
        self.assertEqual(len(selected), 8)
        self.assertEqual(sum(row["target"] == "a" for row in selected), 4)
        self.assertEqual(selected, balanced_sample(list(reversed(rows)), ("a", "b"), 4, 17))
        self.assertNotEqual(selected, balanced_sample(rows, ("a", "b"), 4, 18))
        self.assertEqual(len({row["number"] for row in selected}), 8)

    def test_shortage_fails_instead_of_shrinking_a_class(self):
        with self.assertRaisesRegex(ValueError, "Only 10 unique a examples; need 11"):
            balanced_sample(self.rows(), ("a", "b"), 11, 17)
        with self.assertRaisesRegex(ValueError, "Unexpected target"):
            balanced_sample(self.rows(), ("a",), 1, 17)

    def test_duplicates_and_development_overlap_cannot_enter_test_sample(self):
        rows = [example(1, "same", "a", "1"), example(2, "same", "a", "2"),
                example(3, "development", "b", "3"), example(4, "new", "b", "4")]
        unique, audit = deduplicate(rows, excluded_texts={"development"})
        self.assertEqual([row["number"] for row in unique], [1, 4])
        self.assertEqual(audit["duplicate_rows_removed"], 1)
        self.assertEqual(audit["development_overlap_removed"], 1)
        self.assertEqual(audit["development_examples"], 1)
        self.assertEqual(len(balanced_sample(unique, ("a", "b"), 1, 0)), 2)

    def test_conflicting_duplicate_targets_are_not_arbitrarily_resolved(self):
        with self.assertRaisesRegex(ValueError, "conflicting target"):
            deduplicate([example(1, "same", "a", "1"), example(2, "same", "b", "2")])


class FreezingTests(unittest.TestCase):
    def test_existing_snapshots_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            freeze_bytes(path, b"original")
            freeze_bytes(path, b"original")
            with self.assertRaisesRegex(ValueError, "different contents"):
                freeze_bytes(path, b"replacement")
            self.assertEqual(path.read_bytes(), b"original")

    def test_cached_unversioned_sources_require_correct_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "data.txt").write_bytes(b"modified source")
            with patch("benchmark_data.requests.get") as request:
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    fetch_source(directory, "data.txt", "https://example.com/data",
                                 expected_sha256="0" * 64)
                request.assert_not_called()

    @patch("benchmark_data.load_sources")
    def test_snapshot_records_counts_task_and_sampling_and_refuses_changed_seed(self, loader):
        labels = TASKS["ag_news"]["labels"]
        rows = SamplingTests().rows(labels, 6)
        loader.return_value = rows, [{"url": "source", "sha256": "a" * 64}], {}, "test", 2
        with tempfile.TemporaryDirectory() as directory:
            saved = prepare_dataset("ag_news", directory, seed=11)
            self.assertEqual(saved["task"], TASKS["ag_news"])
            self.assertEqual(saved["metadata"]["sampling"], "class_balanced")
            self.assertEqual(saved["metadata"]["selected_count"], 8)
            self.assertEqual(saved["metadata"]["selected_class_counts"],
                             {label: 2 for label in labels})
            self.assertEqual(prepare_dataset("ag_news", directory, seed=11), saved)
            with self.assertRaisesRegex(ValueError, "different contents"):
                prepare_dataset("ag_news", directory, seed=12)

    @patch("benchmark_data.load_sources")
    def test_full_trec_retains_natural_distribution_and_documents_dedup(self, loader):
        rows = [example(1, "What is it ?", "DESC", "1"),
                example(2, "What is it ?", "DESC", "2"),
                example(3, "Where is it ?", "LOC", "3"),
                example(4, "Why is it ?", "DESC", "4")]
        loader.return_value = rows, [], {}, "TREC 10 test", None
        with tempfile.TemporaryDirectory() as directory:
            saved = prepare_dataset("trec", directory)
            self.assertEqual(saved["metadata"]["sampling"], "natural")
            self.assertEqual(saved["metadata"]["source_count"], 4)
            self.assertEqual(saved["metadata"]["selected_count"], 3)
            self.assertEqual(saved["metadata"]["deduplication"]["duplicate_rows_removed"], 1)
            self.assertEqual(saved["metadata"]["selected_class_counts"], {"DESC": 2, "LOC": 1})


if __name__ == "__main__":
    unittest.main()
