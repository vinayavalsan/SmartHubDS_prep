"""Offline strategy-summary and won-bid regression checks; no Slack delivery."""

import ast
import os
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pandas as pd

from smarthub.monitoring import strategy_comparison as comparison


class StrategyComparisonTests(unittest.TestCase):
    def setUp(self):
        self.start, self.end = comparison.report_bounds("2026-10-15T05:00Z")
        self.frame = pd.DataFrame(
            [
                (1, 11, 14, 6, "2026-10-15T01:00Z", 1.0, True, False),
                (2, 11, 14, 6, "2026-10-15T02:00Z", 3.0, False, False),
                (3, 11, 7, 6, "2026-10-15T03:00Z", 4.0, "", False),
                (4, 23, 7, 6, "2026-10-15T03:00Z", 5.0, True, False),
                (5, 11, 14, 6, "2026-10-15T04:00Z", 0.0, True, False),
                (6, 11, 14, 6, "2026-10-15T04:00Z", 9.0, True, True),
                (7, 11, 14, 1, "2026-10-15T04:00Z", 9.0, True, False),
                (8, 11, 14, 6, "2026-10-16T00:00Z", 9.0, True, False),
            ],
            columns=[
                "id",
                "account_id",
                "bidding_strategy_id",
                "lead_type_id",
                "created_at",
                "bid",
                "won",
                "erred",
            ],
        )

    def summary(self, frame=None):
        return comparison.build_summary(
            self.frame if frame is None else frame,
            lead_type_id=6,
            ml_strategy_ids=[14],
            start=self.start,
            end=self.end,
        )

    def test_population_averages_and_zero_wins(self):
        summary = self.summary(pd.concat([self.frame, self.frame.iloc[[0]]]))
        self.assertEqual(len(summary), 2)
        ml = summary.iloc[0]
        self.assertEqual(ml.opportunities, 2)
        self.assertEqual(ml.winrate, 0.5)
        self.assertEqual(ml.avg_bid, 2.0)
        self.assertEqual(ml.avg_bid_on_wins, 1.0)
        self.assertTrue(pd.isna(summary.iloc[1].avg_bid_on_wins))

    def test_utc_bounds(self):
        start, end = comparison.report_bounds("2026-10-16T23:00-05:00")
        self.assertEqual(str(start), "2026-10-17 00:00:00+00:00")
        self.assertEqual(str(end), "2026-10-17 04:00:00+00:00")

    def test_cli_sends_by_default_without_extra_flag(self):
        with (
            patch.object(sys, "argv", ["strategy_comparison", "--lead-type-id", "6"]),
            patch.object(comparison, "check_once") as check,
        ):
            comparison.main()
            check.assert_called_once_with(
                lead_type_id=6, as_of=None, dry_run=False, resend=True
            )

    def test_blank_line_between_accounts(self):
        second = self.frame.copy()
        second["id"] += 100
        second["account_id"] = 22
        summary = self.summary(pd.concat([self.frame, second], ignore_index=True))
        payloads = comparison.build_payloads(
            summary,
            lead_type_id=6,
            start=self.start,
            end=self.end,
            min_opportunities=100,
        )
        table = payloads[0]["blocks"][1]["text"]
        self.assertEqual(table["type"], "mrkdwn")
        self.assertRegex(table["text"], r"\n\n +22 ")
        self.assertTrue(table["text"].startswith("```\n"))
        self.assertIn("Meas WR", table["text"])

    def test_four_hour_windows_and_exact_boundary(self):
        for anchor in ("2026-10-16T16:00Z", "2026-10-16T19:59Z"):
            start, end = comparison.report_bounds(anchor)
            self.assertEqual(str(start), "2026-10-16 12:00:00+00:00")
            self.assertEqual(str(end), "2026-10-16 16:00:00+00:00")
        with self.assertRaises(ValueError):
            comparison.report_bounds(window_hours=5)

    def test_slack_table_is_bounded_and_informational(self):
        summary = pd.concat([self.summary()] * 30, ignore_index=True)
        payloads = comparison.build_payloads(
            summary,
            lead_type_id=6,
            start=self.start,
            end=self.end,
            min_opportunities=100,
        )
        self.assertEqual(len(payloads), 3)
        for payload in payloads:
            self.assertNotIn("@here", payload["text"])
            self.assertIn("14 ML*", payload["text"])
            self.assertIn("N/A", payload["text"])
            self.assertEqual(payload["blocks"][0]["text"]["type"], "mrkdwn")
            self.assertIn("*SmartHub", payload["blocks"][0]["text"]["text"])
            self.assertIn(
                "· strategy-comparison · SUMMARY · auto (6)*", payload["text"]
            )
            self.assertEqual(payload["blocks"][-1]["type"], "context")
            self.assertIn("env:", payload["blocks"][-1]["elements"][0]["text"])
            self.assertIn("UTC", payload["blocks"][-1]["elements"][0]["text"])
            for block in payload["blocks"]:
                if "text" in block:
                    self.assertLess(len(block["text"]["text"]), 3000)

    def test_shared_table_layout_includes_env_and_never_delivers(self):
        with (
            patch.dict(os.environ, {"SLACK_ENV_LABEL": "staging"}),
            patch.object(comparison.notifications, "_post") as post,
        ):
            payload = comparison.build_payloads(
                self.summary(),
                lead_type_id=6,
                start=self.start,
                end=self.end,
                min_opportunities=100,
            )[0]
            self.assertTrue(
                payload["text"].startswith(":white_check_mark: *SmartHub STAGING")
            )
            self.assertIn("env: `staging`", payload["text"])
            self.assertNotIn("<!here>", payload["text"])
            post.assert_not_called()

    def test_dry_run_and_delivery_deduplication(self):
        with TemporaryDirectory() as tmp:
            with (
                patch.object(comparison, "_load_raw", return_value=self.frame),
                patch.object(comparison, "_send", return_value=True) as sender,
                patch.object(comparison.paths, "data_dir", return_value=Path(tmp)),
            ):
                comparison.check_once(lead_type_id=6, as_of=self.end, dry_run=True)
                sender.assert_not_called()
                self.assertFalse(list(Path(tmp).rglob("*.json")))
                comparison.check_once(lead_type_id=6, as_of=self.end)
                comparison.check_once(lead_type_id=6, as_of=self.end)
                self.assertEqual(sender.call_count, 1)

    def test_failed_delivery_retries(self):
        with TemporaryDirectory() as tmp:
            with (
                patch.object(comparison, "_load_raw", return_value=self.frame),
                patch.object(comparison, "_send", side_effect=[False, True]) as sender,
                patch.object(comparison.paths, "data_dir", return_value=Path(tmp)),
            ):
                with self.assertRaises(RuntimeError):
                    comparison.check_once(lead_type_id=6, as_of=self.end)
                comparison.check_once(lead_type_id=6, as_of=self.end)
                self.assertEqual(sender.call_count, 2)

    def test_manual_resend_and_dry_run_stays_offline(self):
        with TemporaryDirectory() as tmp:
            with (
                patch.object(comparison, "_load_raw", return_value=self.frame),
                patch.object(comparison, "_send", return_value=True) as sender,
                patch.object(comparison.paths, "data_dir", return_value=Path(tmp)),
            ):
                comparison.check_once(lead_type_id=6, as_of=self.end)
                comparison.check_once(lead_type_id=6, as_of=self.end, resend=True)
                self.assertEqual(sender.call_count, 2)
                state_file = next(Path(tmp).rglob("*.json"))
                before = state_file.read_bytes()
                comparison.check_once(
                    lead_type_id=6, as_of=self.end, resend=True, dry_run=True
                )
                self.assertEqual(sender.call_count, 2)
                self.assertEqual(state_file.read_bytes(), before)

    def test_stale_data_does_not_send(self):
        with (
            patch.object(comparison, "_load_raw", return_value=self.frame.iloc[:4]),
            patch.object(comparison, "_send") as sender,
        ):
            with self.assertRaisesRegex(ValueError, "stale"):
                comparison.check_once(lead_type_id=6, as_of=self.end)
            sender.assert_not_called()

    def test_partial_delivery_retries_only_outstanding_page(self):
        frames = []
        for index in range(11):
            frame = self.frame.copy()
            frame["id"] += index * 100
            frame["account_id"] += index * 100
            frames.append(frame)
        frame = pd.concat(frames, ignore_index=True)
        with TemporaryDirectory() as tmp:
            with (
                patch.object(comparison, "_load_raw", return_value=frame),
                patch.object(
                    comparison, "_send", side_effect=[True, False, True]
                ) as sender,
                patch.object(comparison.paths, "data_dir", return_value=Path(tmp)),
            ):
                with self.assertRaises(RuntimeError):
                    comparison.check_once(lead_type_id=6, as_of=self.end)
                comparison.check_once(lead_type_id=6, as_of=self.end)
                self.assertEqual(sender.call_count, 3)
                self.assertEqual(sender.call_args_list[1], sender.call_args_list[2])


class IndependentWorkflowTests(unittest.TestCase):
    """Check real wrapper functions without starting Prefect or external work."""

    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def load_function(self, relative_path, name, namespace):
        source = self.root / relative_path
        tree = ast.parse(source.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        function.decorator_list = []
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        return namespace[name]

    def test_prediction_refresh_has_separate_type_watermarks(self):
        module = types.ModuleType("smarthub.data_pull.prediction_logs")
        module.pull_and_persist_prediction_logs = Mock(
            return_value={"prediction_rows": 1}
        )
        resolve = Mock(return_value=("2026-10-16 12:00:00", "2026-10-16 13:00:00"))
        variable = Mock()
        from smarthub.core.lead_types import lead_type_name

        namespace = {
            "get_run_logger": Mock(),
            "resolve_window": resolve,
            "Variable": variable,
            "resolve_lead_type_name": lead_type_name,
        }
        function = self.load_function(
            "src/smarthub/data_pull/flow.py", "prediction_log_refresh_flow", namespace
        )
        with patch.dict(sys.modules, {module.__name__: module}):
            function(lead_type_id=6)
            function(lead_type_id=1)
        self.assertEqual(
            [call.args[0] for call in resolve.call_args_list],
            [
                "prediction_log_last_pull_timestamp_auto",
                "prediction_log_last_pull_timestamp_home",
            ],
        )
        self.assertEqual(
            [
                call.args[2]
                for call in module.pull_and_persist_prediction_logs.call_args_list
            ],
            [6, 1],
        )

    def test_data_pull_does_not_call_monitoring_or_summary(self):
        frame = pd.DataFrame({"id": [1]})
        namespace = {
            "get_run_logger": Mock(),
            "_utc_now_naive": Mock(),
            "resolve_lead_type_name": Mock(return_value="auto"),
            "watermark_variable": Mock(return_value="watermark"),
            "resolve_window": Mock(return_value=("start", "end")),
            "Variable": Mock(),
            "fetch": Mock(return_value=frame),
            "coerce": lambda raw: raw,
            "validate": Mock(),
            "persist": Mock(return_value={}),
            "update_watermark": Mock(return_value="end"),
            "_report": Mock(),
            "_notify_success": Mock(),
        }
        function = self.load_function(
            "src/smarthub/data_pull/flow.py", "data_pull_flow", namespace
        )
        result = function(lead_type_id=6)
        self.assertEqual(result["rows"], 1)

    def test_summary_task_has_no_pull_dependency(self):
        check = Mock(return_value=pd.DataFrame({"account_id": [11, 22]}))
        function = self.load_function(
            "src/smarthub/monitoring/flow.py",
            "send_strategy_summary",
            {"check_strategy_comparison": check},
        )
        self.assertEqual(function(6), 2)
        check.assert_called_once_with(lead_type_id=6, as_of=None)

    def test_yaml_delays_home_by_two_minutes(self):
        import yaml

        deployments = {
            item["name"]: item
            for item in yaml.safe_load((self.root / "prefect.yaml").read_text())[
                "deployments"
            ]
        }
        expected = {
            "data-pull": ("0 * * * *", "2 * * * *"),
            "prediction-log-refresh": ("15 * * * *", "17 * * * *"),
            "build-features": ("5 */4 * * *", "7 */4 * * *"),
            "model-degradation": ("20 * * * *", "22 * * * *"),
            "strategy-comparison": ("25 */4 * * *", "27 */4 * * *"),
        }
        for name, crons in expected.items():
            item = deployments[name]
            self.assertEqual(
                item["schedules"],
                [
                    {
                        "cron": crons[0],
                        "timezone": "UTC",
                        "parameters": {"lead_type_id": 6},
                    },
                    {
                        "cron": crons[1],
                        "timezone": "UTC",
                        "parameters": {"lead_type_id": 1},
                    },
                ],
            )
            self.assertEqual(item["concurrency_limit"], 1)
            source, function = item["entrypoint"].split(":")
            tree = ast.parse((self.root / source).read_text())
            self.assertTrue(
                any(
                    isinstance(node, ast.FunctionDef) and node.name == function
                    for node in tree.body
                )
            )

    def test_training_deployment_matches_etl_branch(self):
        import yaml

        payload = yaml.safe_load((self.root / "prefect.yaml").read_text())
        training = next(
            item for item in payload["deployments"] if item["name"] == "train-model"
        )
        self.assertEqual(
            training["entrypoint"], "src/smarthub/train_and_predict/flow.py:train_flow"
        )
        self.assertNotIn("hpo_schedule", training["parameters"])
        self.assertEqual(
            [row["cron"] for row in training["schedules"]], ["0 10 * * *", "2 10 * * *"]
        )
        self.assertTrue(all(row["timezone"] == "UTC" for row in training["schedules"]))
        self.assertEqual(
            [row["parameters"]["lead_type_id"] for row in training["schedules"]], [6, 1]
        )


if __name__ == "__main__":
    unittest.main()
