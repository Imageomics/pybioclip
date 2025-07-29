import os
import tempfile
import unittest
from bioclip import recorder
import json


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

    def test_save_recorded_predictions_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            clf = DummyClassifier()
            rec = recorder.attach_prediction_recorder(clf, foo="bar")
            rec.add_prediction(["img1.png"], label="dog")
            out_path = os.path.join(tmpdirname, "report.json")
            recorder.save_recorded_predictions(clf, out_path)
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r") as f:
                data = json.load(f)
            self.assertIn("pybioclip_version", data)
            self.assertIn("predictions", data)
            self.assertEqual(data["predictions"][0]["images"], ["img1.png"])
            self.assertEqual(data["predictions"][0]["details"]["label"], "dog")

    def test_prediction_log_report_format_key(self):
        self.assertEqual(recorder.PredictionLogReport.format_key("foo_bar"), "Foo Bar")
        self.assertEqual(recorder.PredictionLogReport.format_key("batch"), "Batch")

    def test_prediction_log_report_adds_top_level_info(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            clf = DummyClassifier(model_str="clip", pretrained_str=None, device="cuda")
            rec = recorder.PredictionRecorder(clf, alpha=0.5, beta=None)
            rec.add_prediction(["img.png"], label="cat")
            report = recorder.PredictionLogReport(rec, command_line="bioclip predict image.jpeg")
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
            report = recorder.PredictionLogReport(rec, command_line="bioclip predict image.jpeg")
            out_path = os.path.join(tmpdirname, "report3.txt")
            report.save(out_path)
            with open(out_path, "r") as f:
                text = f.read()
            self.assertIn("a.png", text)
            self.assertIn("b.png", text)
            self.assertEqual(text.count("Images:"), 2)

    def test_create_dictionary_contains_expected_keys(self):
        clf = DummyClassifier(model_str="clip", pretrained_str="openai", device="cuda")
        rec = recorder.PredictionRecorder(clf, foo="bar", batch_size=4)
        rec.add_prediction(["img1.png"], label="cat", score=0.95)
        report = recorder.PredictionLogReport(rec, command_line="python script.py")
        d = report.create_dictionary()
        self.assertIn("pybioclip_version", d)
        self.assertIn("start", d)
        self.assertIn("end", d)
        self.assertIn("command_line", d)
        self.assertIn("model", d)
        self.assertIn("pretrained", d)
        self.assertIn("device", d)
        self.assertIn("top_level_settings", d)
        self.assertIn("predictions", d)
        self.assertEqual(d["model"], "clip")
        self.assertEqual(d["pretrained"], "openai")
        self.assertEqual(d["device"], "cuda")
        self.assertEqual(d["top_level_settings"]["foo"], "bar")
        self.assertEqual(d["top_level_settings"]["batch_size"], 4)
        self.assertEqual(d["predictions"][0]["images"], ["img1.png"])
        self.assertEqual(d["predictions"][0]["details"]["label"], "cat")
        self.assertEqual(d["predictions"][0]["details"]["score"], 0.95)

    def test_create_report_json_output(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            clf = DummyClassifier()
            rec = recorder.PredictionRecorder(clf, foo="bar")
            rec.add_prediction(["img1.png"], label="dog")
            out_path = os.path.join(tmpdirname, "report.json")
            rec.create_report(out_path, command_line="python test.py")
            self.assertTrue(os.path.exists(out_path))
            with open(out_path, "r") as f:
                data = json.load(f)
            self.assertIn("pybioclip_version", data)
            self.assertIn("predictions", data)
            self.assertEqual(data["predictions"][0]["images"], ["img1.png"])
            self.assertEqual(data["predictions"][0]["details"]["label"], "dog")
            self.assertEqual(data["command_line"], "python test.py")

    def test_add_prediction_updates_end_time(self):
        clf = DummyClassifier()
        rec = recorder.PredictionRecorder(clf)
        before = rec.start
        rec.add_prediction(["img1.png"], label="cat")
        self.assertIsNotNone(rec.end)
        self.assertGreaterEqual(rec.end, before)

    def test_attach_prediction_recorder_passes_top_level_settings(self):
        clf = DummyClassifier()
        rec = recorder.attach_prediction_recorder(clf, param1=123, param2="abc")
        self.assertEqual(rec.top_level_settings["param1"], 123)
        self.assertEqual(rec.top_level_settings["param2"], "abc")

    def test_save_recorded_predictions_raises_when_json_exists(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            clf = DummyClassifier()
            rec = recorder.attach_prediction_recorder(clf)
            rec.add_prediction(["img1.png"], label="dog")
            out_path = os.path.join(tmpdirname, "report.json")
            with open(out_path, "w") as f:
                json.dump({"existing": "data"}, f)
            with self.assertRaises(ValueError):
                recorder.save_recorded_predictions(clf, out_path)


if __name__ == "__main__":
    unittest.main()
