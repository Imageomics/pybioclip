"""Records predictions made by a classifier and saves the output to a file."""
from .__about__ import __version__ as pybioclip_version
from datetime import datetime
from typing import Union
import sys
import json
import os


def is_json_file(path: str) -> bool:
    return path.endswith(".json")


def verify_recorder_path(path: str):
    """
    Ensure the path can either be appended to or is a non-existent JSON file.
    """
    if is_json_file(path):
        if os.path.exists(path):
            raise ValueError(f"File {path} already exists. Please choose a different file name.")
    else:
        return True


def attach_prediction_recorder(classifier: object, **top_level_settings):
    """
    Attach a PredictionRecorder to the classifier instance that will record metadata and subsequent predictions.
    Call save_recorded_predictions to save the recorded predictions to a file.

    Args:
        classifier (object): The classifier (such as TreeOfLifeClassifier) instance to attach the recorder to.
        **top_level_settings: Additional settings to be recorded.

    Returns:
        PredictionRecorder: An instance of PredictionRecorder attached to the classifier.
    """
    recorder = PredictionRecorder(classifier, **top_level_settings)
    classifier.set_recorder(recorder)
    return recorder


def save_recorded_predictions(classifier: object, path: str, include_command_line: bool = True):
    """
    Saves recorded predictions from the classifier to a file.
    Before calling this function, ensure that the classifier has a recorder attached
    using attach_prediction_recorder. Saves the recorder's data to the specified file path in 
    either JSON or plain text format. If the file extension is '.json', the data is serialized
    as JSON. Otherwise, the data is appended in a human-readable text format.

    Args:
        classifier (object): The classifier instance (such as TreeOfLifeClassifier) with recorded predictions.
        path (str): The file path where the report will be saved.
        include_command_line (bool): When True includes the python command line in the log file.

    Raises:
        ValueError: If the output path extension is .json and the file already exists.
    """
    if classifier.recorder:
        command_line = " ".join(sys.argv) if include_command_line else None
        classifier.recorder.create_report(path, command_line=command_line)
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

    def add_prediction(self, images, **kwargs):
        """
        Adds a prediction entry to the recorder.
        Parameters:
            images: The images associated with the prediction.
            **kwargs: Additional details related to the prediction.
        """
        self.predictions.append({"images": images, "details": kwargs})
        self.end = datetime.now().isoformat(timespec="seconds")

    def create_report(self, path: str, command_line: Union[str, None]):
        """
        Creates a report of the predictions and saves it to the specified path.

        Args:
            path (str): The file path where the report will be saved.
            command_line (str): The command line used to run the predictions.
        """
        report = PredictionLogReport(self, command_line)
        report.save(path)


class PredictionLogReport:
    """
    A class to generate a report of the predictions made by the PredictionRecorder.
    The report includes metadata about the model, device, and settings used during the predictions.
    """

    def __init__(self, recorder: PredictionRecorder, command_line: Union[str, None]):
        self.recorder = recorder
        self.start = recorder.start
        self.end = recorder.end
        self.command_line = command_line

    def create_dictionary(self):
        """
        Creates a dictionary representation of the report.

        Returns:
            dict: A dictionary containing the report data.
        """
        return {
            "pybioclip_version": pybioclip_version,
            "start": self.start,
            "end": self.end,
            "command_line": self.command_line,
            "model": self.recorder.model_str,
            "pretrained": self.recorder.pretrained_str,
            "device": self.recorder.device,
            "top_level_settings": self.recorder.top_level_settings,
            "predictions": self.recorder.predictions
        }

    @staticmethod
    def format_key(key):
        return " ".join([word.capitalize() for word in key.split("_")])

    def _add_title(self, f):
        title = f"** Prediction Log - pybioclip v{pybioclip_version} - {self.start} to {self.end} **"
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
        for pred_dict in self.recorder.predictions:
            for k, v in pred_dict["details"].items():
                f.write(f"{self.format_key(k)}: {v}\n")
            images = pred_dict["images"]
            f.write("Images:\n" + "\n".join(images) + "\n")

    def save(self, path: str):
        """
        Saves the recorder's data to the specified file path in either JSON or plain text format.
        If the file extension is '.json', the data is serialized as JSON. Otherwise, the data is appended
        in a human-readable text format.
        Args:
            path (str): The file path where the data should be saved.
        Raises:
            ValueError: If the output path extension is .json and the file already exists.
            IOError: If there is an error writing to the file.
        """
        verify_recorder_path(path)
        if is_json_file(path):
            with open(path, "w") as f:
                json.dump(self.create_dictionary(), f, indent=4)
        else:
            with open(path, "a") as f:
                self._add_title(f)
                if self.command_line:
                    f.write(f"Command Line: {self.command_line}\n")
                self._add_top_level_info(f)
                self._add_predictions(f)
                f.write("\n")
