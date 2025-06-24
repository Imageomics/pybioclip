from .__about__ import __version__ as pybioclip_version
from datetime import datetime


def attach_prediction_recorder(classifier: object, **top_level_settings):
    """
    Attaches a PredictionRecorder to the classifier.

    Args:
        classifier (object): The classifier instance to attach the recorder to.
        **top_level_settings: Additional settings to be recorded.

    Returns:
        PredictionRecorder: An instance of PredictionRecorder attached to the classifier.
    """
    recorder = PredictionRecorder(classifier, **top_level_settings)
    classifier.set_recorder(recorder)
    return recorder


def save_recorded_predictions(classifier: object, path: str):
    """
    Appends recorded predictions from the classifier to a report file.

    Args:
        classifier (object): The classifier instance with recorded predictions.
        path (str): The file path where the report will be saved.
    """
    if classifier.recorder:
        classifier.recorder.create_report(path)
    else:
        raise ValueError("The classifier does not have a recorder attached.")


class PredictionRecorder:
    """
    A class to record predictions made by a classifier.
    It stores metadata about the model, device, and settings used during the predictions,
    and allows for the addition of prediction data.
    The predictions can be saved to a report file.
    """

    def __init__(self, classifier: object, **top_level_settings):
        self.model_str = classifier.model_str
        self.pretrained_str = classifier.pretrained_str
        self.device = classifier.device
        self.predictions = []
        self.top_level_settings = top_level_settings
        self.start = datetime.now().isoformat(timespec="seconds")
        self.end = None
        classifier.set_recorder(self)

    def add_prediction(self, images, **kwargs):
        """
        Adds a prediction to the recorder.

        Args:
            prediction (dict): A dictionary containing the prediction data.
        """
        self.predictions.append({"images": images, "details": kwargs})
        self.end = datetime.now().isoformat(timespec="seconds")

    def create_report(self, path: str):
        """
        Creates a report of the predictions and saves it to the specified path.

        Args:
            path (str): The file path where the report will be saved.
        """
        report = PredictionLogReport(self)
        report.save(path)


class PredictionLogReport:
    """
    A class to generate a report of the predictions made by the PredictionRecorder.
    The report includes metadata about the model, device, and settings used during the predictions.
    """

    def __init__(self, recorder: PredictionRecorder):
        self.recorder = recorder
        self.start = recorder.start
        self.end = recorder.end

    @staticmethod
    def format_key(key):
        return " ".join([word.capitalize() for word in key.split("_")])

    def _add_title(self, f):
        title = f"** Prediction Log - pybioclip v{pybioclip_version} - {self.start} to {self. end} **"
        f.write("*" * len(title) + "\n")
        f.write(f"{title}\n")
        f.write("*" * len(title) + "\n")

    def _add_top_level_info(self, f):
        f.write(f"Model: {self.recorder.model_str}\n")
        if self.recorder.pretrained_str:
            f.write(f"Pretrained: {self.recorder.pretrained_str}\n")
        f.write(f"Device: {self.recorder.device}\n")
        for k, v in self.recorder.top_level_settings.items():
            if v is not None:
                f.write(f"{self.format_key(k)}: {v}\n")

    def _add_predictions(self, f):
        print("\n")
        for pred_dict in self.recorder.predictions:
            for k, v in pred_dict["details"].items():
                f.write(f"{self.format_key(k)}: {v}\n")
            images = pred_dict["images"]
            f.write("Images:\n" + "\n".join(images) + "\n")

    def save(self, path: str):
        with open(path, "a") as f:
            self._add_title(f)
            self._add_top_level_info(f)
            self._add_predictions(f)
            f.write("\n")
