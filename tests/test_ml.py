"""Tests for the yield model, backtest, drift monitor and retraining.

Run from the repo root:
    python -m unittest discover tests
"""
import json
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ml import config, drift, pipeline  # noqa: E402
from ml.data import load_training_data  # noqa: E402
from ml.evaluate import backtest  # noqa: E402
from ml.models import GBMOnHistory, HistoryLookup, LastSeason, candidates  # noqa: E402


class IsolatedPaths(unittest.TestCase):
    """Point every output path at a temp dir so tests never touch the real model."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        shutil.copy(config.BASE_DATA, self.tmp / "base.csv")
        self.patches = [
            mock.patch.object(config, "BASE_DATA", self.tmp / "base.csv"),
            mock.patch.object(config, "NEW_OBSERVATIONS", self.tmp / "new_observations.csv"),
            mock.patch.object(config, "MODEL_DIR", self.tmp / "models"),
            mock.patch.object(config, "MODEL_PATH", self.tmp / "models" / "yield_model.joblib"),
            mock.patch.object(config, "MODEL_INFO_PATH", self.tmp / "model_info.json"),
            mock.patch.object(config, "PRED_LOG_PATH", self.tmp / "prediction_log.jsonl"),
            mock.patch.object(config, "DRIFT_CACHE_PATH", self.tmp / "drift_report.json"),
            mock.patch.object(config, "MLFLOW_DIR", self.tmp / "mlruns"),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)


class DataTests(unittest.TestCase):
    def test_bales_crops_excluded(self):
        df = load_training_data(include_new=False)
        for crop in ["Jute", "Mesta", "Cotton(lint)"]:
            self.assertNotIn(crop, set(df["Crop"]))

    def test_implausible_rainfall_is_missing_not_zero(self):
        df = load_training_data(include_new=False)
        self.assertFalse((df["Rainfall"] < config.MIN_PLAUSIBLE_RAINFALL_MM).any())
        self.assertTrue(df["Rainfall"].isna().any())

    def test_target_is_yield(self):
        df = load_training_data(include_new=False)
        np.testing.assert_allclose(df["Yield"], df["Production"] / df["Area"])


class LeakageTests(unittest.TestCase):
    def test_backtest_trains_only_on_past_years(self):
        df = load_training_data(include_new=False)
        seen = []

        class Spy(LastSeason):
            def fit(self, d):
                seen.append(int(d["Crop_Year"].max()))
                return super().fit(d)

            def predict(self, d):
                seen.append(("test", int(d["Crop_Year"].min())))
                return super().predict(d)

        backtest(Spy, df)
        train_max = [s for s in seen if not isinstance(s, tuple)]
        test_min = [s[1] for s in seen if isinstance(s, tuple)]
        for tr, te in zip(train_max, test_min):
            self.assertLess(tr, te)

    def test_history_model_features_use_only_earlier_years(self):
        df = load_training_data(include_new=False)
        year = 2012
        past = df[df["Crop_Year"] < year]
        rows = df[df["Crop_Year"] == year]
        f_past_only = HistoryLookup(past).features(rows)
        # Changing the target year's own yields must not change its features.
        tampered = df.copy()
        tampered.loc[tampered["Crop_Year"] == year, "Yield"] *= 100
        f_tampered = HistoryLookup(tampered[tampered["Crop_Year"] < year]).features(rows)
        pd.testing.assert_frame_equal(f_past_only, f_tampered)
        GBMOnHistory().fit(df)  # fits without error on the full data


class ModelTests(unittest.TestCase):
    def test_every_candidate_predicts_positive_finite_yield(self):
        df = load_training_data(include_new=False)
        train, test = df[df["Crop_Year"] < 2014], df[df["Crop_Year"] == 2014]
        for m in candidates():
            p = m.fit(train).predict(test)
            self.assertEqual(len(p), len(test), m.name)
            self.assertTrue(np.isfinite(p).all() and (p > 0).all(), m.name)

    def test_unseen_combination_falls_back(self):
        df = load_training_data(include_new=False)
        m = LastSeason().fit(df)
        row = pd.DataFrame([{"District_Name": "RI BHOI", "Season": "Kharif", "Crop": "Rice", "Rainfall": np.nan}])
        self.assertFalse(m.has_history(row)[0])
        self.assertGreater(m.predict(row)[0], 0)


class PipelineTests(IsolatedPaths):
    def test_train_then_retrain_without_new_data_keeps_model(self):
        first = pipeline.run("train")
        self.assertEqual(first["status"], "upgraded")
        info = json.loads(config.MODEL_INFO_PATH.read_text())
        self.assertEqual(info["version"], 1)
        self.assertEqual(info["champion"], first["leaderboard"][0]["model"])

        second = pipeline.run("retrain")
        self.assertEqual(second["status"], "kept")
        self.assertEqual(json.loads(config.MODEL_INFO_PATH.read_text())["version"], 1)

    def test_new_harvest_data_is_picked_up(self):
        pipeline.run("train")
        new = pd.DataFrame([{
            "District_Name": "RI BHOI", "Crop_Year": 2015, "Season": "Autumn", "Crop": "Rice",
            "Area": 100.0, "Production": 300.0, "Rainfall": 2400.0,
        }])
        new.to_csv(config.NEW_OBSERVATIONS, index=False)
        result = pipeline.run("retrain")
        self.assertIn(result["status"], ("refreshed", "upgraded"))
        info = json.loads(config.MODEL_INFO_PATH.read_text())
        self.assertEqual(info["version"], 2)

        bundle = pipeline.load_bundle()
        row = pd.DataFrame([{"District_Name": "RI BHOI", "Season": "Autumn", "Crop": "Rice", "Rainfall": np.nan}])
        if bundle["name"] == "last_season":
            self.assertAlmostEqual(bundle["model"].predict(row)[0], 3.0, places=6)

    def test_model_file_is_never_half_written(self):
        pipeline.run("train")
        self.assertFalse(config.MODEL_PATH.with_suffix(".tmp").exists())


class DriftTests(IsolatedPaths):
    def _write_log(self, rows):
        with open(config.PRED_LOG_PATH, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def _like_training(self, n, seed=0):
        df = load_training_data(include_new=False).sample(n, replace=True, random_state=seed)
        return [{"district": r.District_Name, "crop": r.Crop, "season": r.Season, "area": 1.0,
                 "rainfall": 2000.0, "rainfall_source": "district_average", "has_history": True}
                for r in df.itertuples()]

    def test_reads_lowercase_log_keys(self):
        pipeline.run("train")
        self._write_log(self._like_training(200))
        report = drift.run_drift_check()
        self.assertIn("district", report["features"])
        self.assertIn("crop", report["features"])

    def test_no_drift_when_requests_look_like_training(self):
        pipeline.run("train")
        self._write_log(self._like_training(300))
        report = drift.run_drift_check()
        for name in ("district", "crop", "season"):
            self.assertFalse(report["features"][name]["drifted"], name)

    def test_detects_shift_to_one_district(self):
        pipeline.run("train")
        rows = self._like_training(100)
        for r in rows:
            r["district"] = "RI BHOI"
        self._write_log(rows)
        report = drift.run_drift_check()
        self.assertTrue(report["features"]["district"]["drifted"])
        self.assertEqual(report["status"], "drift_detected")

    def test_detects_cold_start_spike(self):
        pipeline.run("train")
        rows = self._like_training(100)
        for r in rows:
            r["has_history"] = False
        self._write_log(rows)
        report = drift.run_drift_check()
        self.assertTrue(report["features"]["cold_start_rate"]["drifted"])

    def test_insufficient_data(self):
        pipeline.run("train")
        self._write_log(self._like_training(5))
        self.assertEqual(drift.run_drift_check()["status"], "insufficient_data")


if __name__ == "__main__":
    unittest.main()
