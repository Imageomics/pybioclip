import os
import tempfile
import unittest
from bioclip import recorder
import tempfile


class DummyClassifier:
    def __init__(self, model_str="resnet", pretrained_str="imagenet", device="cpu"):
        self.model_str = model_str
        self.pretrained_str = pretrained_str
        self.device = device
        self.recorder = None

    def set_recorder(self, rec):
        self.recorder = rec


class TestRecorder(unittest.TestCase):
    def test_attach_prediction_recorder_sets_recorder(self):
        clf = DummyClassifier()
        rec = recorder.attach_prediction_recorder(clf, foo="bar")
        self.assertIsInstance(rec, recorder.PredictionRecorder)
        self.assertIs(clf.recorder, rec)
        self.assertEqual(rec.top_level_settings["foo"], "bar")

    def test_add_prediction_records_data(self):
        clf = DummyClassifier()
        rec = recorder.PredictionRecorder(clf, batch_size=8)
        rec.add_prediction(["img1.png", "img2.png"], label="cat", score=0.9)
        self.assertEqual(len(rec.predictions), 1)
        pred = rec.predictions[0]
        self.assertEqual(pred["images"], ["img1.png", "img2.png"])
        self.assertEqual(pred["details"]["label"], "cat")
        self.assertEqual(pred["details"]["score"], 0.9)
        self.assertIsNotNone(rec.end)

    def test_save_recorded_predictions_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            clf = DummyClassifier()
            rec = recorder.attach_prediction_recorder(clf, foo="bar")
            rec.add_prediction(["img1.png"], label="dog")
            out_path = os.path.join(tmpdirname, "report.txt")
            recorder.save_recorded_predictions(clf, out_path)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r") as f:
                content = f.read()
            self.assertIn("Prediction Log", content)
            self.assertIn("dog", content)
            self.assertIn("img1.png", content)

    def test_save_recorded_predictions_raises_without_recorder(self):
        clf = DummyClassifier()
        clf.recorder = None
        with self.assertRaises(ValueError):
            recorder.save_recorded_predictions(clf, "dummy.txt")

    def test_prediction_log_report_format_key(self):
        self.assertEqual(recorder.PredictionLogReport.format_key("foo_bar"), "Foo Bar")
        self.assertEqual(recorder.PredictionLogReport.format_key("batch"), "Batch")

    def test_prediction_log_report_adds_top_level_info(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            clf = DummyClassifier(model_str="clip", pretrained_str=None, device="cuda")
            rec = recorder.PredictionRecorder(clf, alpha=0.5, beta=None)
            rec.add_prediction(["img.png"], label="cat")
            report = recorder.PredictionLogReport(rec)
            out_path = os.path.join(tmpdirname, "report2.txt")
            report.save(out_path)
            with open(out_path, "r") as f:
                text = f.read()
            self.assertIn("Model: clip", text)
            self.assertIn("Device: cuda", text)
            self.assertIn("Alpha: 0.5", text)
            self.assertNotIn("Beta:", text)  # None values are skipped

    def test_prediction_log_report_adds_multiple_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            clf = DummyClassifier()
            rec = recorder.PredictionRecorder(clf)
            rec.add_prediction(["a.png"], label="a")
            rec.add_prediction(["b.png"], label="b")
            report = recorder.PredictionLogReport(rec)
            out_path = os.path.join(tmpdirname, "report3.txt")
            report.save(out_path)
            with open(out_path, "r") as f:
                text = f.read()
            self.assertIn("a.png", text)
            self.assertIn("b.png", text)
            self.assertEqual(text.count("Images:"), 2)


if __name__ == "__main__":
    unittest.main()
