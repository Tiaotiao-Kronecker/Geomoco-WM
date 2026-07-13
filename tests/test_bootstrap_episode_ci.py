from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.bootstrap_episode_ci import episode_bootstrap_report, parse_filters


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class BootstrapEpisodeCiTests(unittest.TestCase):
    def test_lower_is_better_gain_is_baseline_minus_candidate(self) -> None:
        baseline_rows = [
            {"window_id": "w0", "episode_id": "e0", "mse": 1.0},
            {"window_id": "w1", "episode_id": "e0", "mse": 1.2},
            {"window_id": "w2", "episode_id": "e1", "mse": 2.0},
            {"window_id": "w3", "episode_id": "e1", "mse": 2.2},
        ]
        candidate_rows = [
            {"window_id": "w0", "episode_id": "e0", "mse": 0.5},
            {"window_id": "w1", "episode_id": "e0", "mse": 0.7},
            {"window_id": "w2", "episode_id": "e1", "mse": 1.5},
            {"window_id": "w3", "episode_id": "e1", "mse": 1.7},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            baseline_path = Path(tmp_dir) / "baseline.jsonl"
            candidate_path = Path(tmp_dir) / "candidate.jsonl"
            write_jsonl(baseline_path, baseline_rows)
            write_jsonl(candidate_path, candidate_rows)

            report = episode_bootstrap_report(
                baseline_path=baseline_path,
                candidate_path=candidate_path,
                metrics=["mse"],
                num_bootstrap=200,
                seed=3,
            )

        metric = report["metrics"]["mse"]
        self.assertAlmostEqual(metric["observed_gain"], 0.5, delta=1e-9)
        self.assertGreater(metric["ci_low"], 0.0)
        self.assertFalse(metric["crosses_zero"])
        self.assertTrue(metric["reliable_positive"])

    def test_episode_mean_weights_episodes_equally(self) -> None:
        baseline_rows = [
            {"window_id": "w0", "episode_id": "long", "mse": 1.0},
            {"window_id": "w1", "episode_id": "long", "mse": 1.0},
            {"window_id": "w2", "episode_id": "short", "mse": 10.0},
        ]
        candidate_rows = [
            {"window_id": "w0", "episode_id": "long", "mse": 0.0},
            {"window_id": "w1", "episode_id": "long", "mse": 0.0},
            {"window_id": "w2", "episode_id": "short", "mse": 0.0},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            baseline_path = Path(tmp_dir) / "baseline.jsonl"
            candidate_path = Path(tmp_dir) / "candidate.jsonl"
            write_jsonl(baseline_path, baseline_rows)
            write_jsonl(candidate_path, candidate_rows)

            window_weighted = episode_bootstrap_report(
                baseline_path=baseline_path,
                candidate_path=candidate_path,
                metrics=["mse"],
                num_bootstrap=20,
                seed=7,
                episode_weighting="window_weighted",
            )
            episode_mean = episode_bootstrap_report(
                baseline_path=baseline_path,
                candidate_path=candidate_path,
                metrics=["mse"],
                num_bootstrap=20,
                seed=7,
                episode_weighting="episode_mean",
            )

        self.assertAlmostEqual(
            window_weighted["metrics"]["mse"]["observed_gain"],
            4.0,
            delta=1e-9,
        )
        self.assertAlmostEqual(
            episode_mean["metrics"]["mse"]["observed_gain"],
            5.5,
            delta=1e-9,
        )

    def test_filter_keeps_matching_rows_only(self) -> None:
        baseline_rows = [
            {"window_id": "w0", "episode_id": "e0", "event_type": "transition_close", "mse": 1.0},
            {"window_id": "w1", "episode_id": "e1", "event_type": "sustain_open", "mse": 5.0},
        ]
        candidate_rows = [
            {"window_id": "w0", "episode_id": "e0", "event_type": "transition_close", "mse": 0.0},
            {"window_id": "w1", "episode_id": "e1", "event_type": "sustain_open", "mse": 0.0},
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            baseline_path = Path(tmp_dir) / "baseline.jsonl"
            candidate_path = Path(tmp_dir) / "candidate.jsonl"
            write_jsonl(baseline_path, baseline_rows)
            write_jsonl(candidate_path, candidate_rows)

            report = episode_bootstrap_report(
                baseline_path=baseline_path,
                candidate_path=candidate_path,
                metrics=["mse"],
                filters=parse_filters(["event_type=transition_close"]),
                num_bootstrap=20,
            )

        self.assertEqual(report["num_windows"], 1)
        self.assertEqual(report["num_episodes"], 1)
        self.assertAlmostEqual(report["metrics"]["mse"]["observed_gain"], 1.0, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
