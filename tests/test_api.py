"""API tests: run with Flask installed (flask-cors optional).

    python -m unittest discover tests
"""
import sys
import types
import unittest
import warnings

warnings.filterwarnings("ignore")

try:
    import flask_cors  # noqa: F401
except ImportError:  # CORS is irrelevant to these tests
    stub = types.ModuleType("flask_cors")
    stub.CORS = lambda app, *a, **k: app
    sys.modules["flask_cors"] = stub

try:
    import app as api
except Exception as e:  # pragma: no cover
    api = None
    IMPORT_ERROR = e


@unittest.skipIf(api is None or api.bundle is None, "app or trained model not available")
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.c = api.app.test_client()
        self._log = api.ml_config.PRED_LOG_PATH
        self._log_existed = self._log.exists()
        self._log_before = self._log.read_text() if self._log_existed else ""

    def tearDown(self):
        # Leave the prediction log as it was.
        if self._log_existed:
            self._log.write_text(self._log_before)
        elif self._log.exists():
            self._log.unlink()

    def post(self, path, body):
        r = self.c.post(path, json=body)
        return r.status_code, r.get_json()

    def test_yield_does_not_depend_on_farm_size(self):
        yields = {self.post("/predict", {"district": "RI BHOI", "season": "Autumn", "crop": "Rice", "area": a})[1]["yield"]
                  for a in (0.2, 1, 10, 100, 1000)}
        self.assertEqual(len(yields), 1)

    def test_production_scales_with_area(self):
        _, one = self.post("/predict", {"district": "RI BHOI", "season": "Autumn", "crop": "Rice", "area": 1})
        _, ten = self.post("/predict", {"district": "RI BHOI", "season": "Autumn", "crop": "Rice", "area": 10})
        self.assertAlmostEqual(ten["production"], one["production"] * 10, delta=0.05)

    def test_bales_crop_rejected(self):
        status, body = self.post("/predict", {"district": "WEST GARO HILLS", "season": "Kharif", "crop": "Jute", "area": 1})
        self.assertEqual(status, 400)
        self.assertIn("bales", body["error"])

    def test_bad_inputs_rejected(self):
        self.assertEqual(self.post("/predict", {"district": "NOWHERE", "season": "Kharif", "crop": "Rice", "area": 1})[0], 400)
        self.assertEqual(self.post("/predict", {"district": "RI BHOI", "season": "Autumn", "crop": "Rice", "area": 0})[0], 400)

    def test_cold_start_is_flagged(self):
        _, body = self.post("/predict", {"district": "RI BHOI", "season": "Kharif", "crop": "Rice", "area": 1})
        self.assertFalse(body["has_history"])
        self.assertIsNotNone(body["confidence_note"])

    def test_recommend_ranks_by_revenue_among_crops_grown_there(self):
        _, body = self.post("/recommend", {"district": "RI BHOI", "season": "Kharif", "area": 1, "slope": "Gentle"})
        revenues = [r["revenue_est"] for r in body["recommendations"]]
        self.assertEqual(revenues, sorted(revenues, reverse=True))
        grown = {c for d, c, s in api.bundle["support"] if d == "RI BHOI" and s == "Kharif"}
        self.assertTrue(all(r["crop"] in grown for r in body["recommendations"]))


if __name__ == "__main__":
    unittest.main()
