"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import os
import json
from interactive_compression.utils import standardize_json


class LocalModelMonitor:
    """
    A class that manages a local directory containing descriptions, operations, and
    metrics for each generated model.
    """

    def __init__(
        self,
        monitor_directory,
        metadata_filename="models.json",
        model_dirname="models",
        model_output_dirname="models/outputs",
        checkpoint_type="torch",  # pickle, torch, (others todo)
        outputs_type="pickle",  # json, pickle, numpy
    ):
        self.monitor_directory = monitor_directory
        self.model_dirname = model_dirname
        self.model_output_dirname = model_output_dirname
        self.metadata_filename = metadata_filename
        self.checkpoint_type = checkpoint_type
        self.outputs_type = outputs_type

        if not os.path.isdir(self.monitor_directory):
            os.mkdir(self.monitor_directory)
        if not os.path.isdir(os.path.join(self.monitor_directory, self.model_dirname)):
            os.mkdir(os.path.join(self.monitor_directory, self.model_dirname))
        if not os.path.isdir(
            os.path.join(self.monitor_directory, self.model_output_dirname)
        ):
            os.mkdir(os.path.join(self.monitor_directory,
                     self.model_output_dirname))

        self.operations = {}
        self.metrics = {}
        self.models = {}
        self.load_metadata()

    def make_info_dictionary(self):
        """
        Returns all model metadata, operations, and metrics as a JSON-serializable
        dictionary."""
        return standardize_json(
            {
                "operations": list(self.operations.values()),
                "metrics": list(self.metrics.values()),
                "models": list(self.models.values()),
            }
        )

    def load_metadata(self):
        if not os.path.exists(
            os.path.join(self.monitor_directory, self.metadata_filename)
        ):
            return

        with open(
            os.path.join(self.monitor_directory, self.metadata_filename), "r"
        ) as file:
            metadata = json.load(file)
        self.operations = {op["name"]: op for op in metadata["operations"]}
        self.metrics = {m["name"]: m for m in metadata["metrics"]}
        self.models = {m["id"]: m for m in metadata["models"]}

    def save_metadata(self):
        with open(
            os.path.join(self.monitor_directory, self.metadata_filename), "w"
        ) as file:
            json.dump(
                self.make_info_dictionary(),
                file,
                indent=2,
            )

    def set_operation(self, name, **parameters):
        for param_name, param_options in parameters.items():
            for option_name, option in param_options.items():
                if option_name not in ("min", "max", "format", "options", "type"):
                    raise ValueError(
                        f"Unexpected operation parameter option '{option_name}' for parameter '{param_name}'"
                    )
        self.operations[name] = {"name": name, "parameters": parameters}
        self.save_metadata()

    def set_metric(
        self, name, primary=False, format=None, unit=None, min=None, max=None
    ):
        self.metrics[name] = {
            k: v
            for k, v in {
                "name": name,
                "primary": primary,
                "format": format,
                "unit": unit,
                "min": min,
                "max": max,
            }.items()
            if v is not None
        }
        self.save_metadata()

    def save_model(
        self, id, metrics, base_id=None, operation=None, checkpoint=None, outputs=None
    ):
        for metric_name in metrics:
            if metric_name not in self.metrics:
                raise ValueError(
                    f"Unknown metric '{metric_name}'. Define it first using set_metric()."
                )

        model_dict = {
            "id": id,
            "base": base_id,
            "operation": operation,
            "metrics": metrics,
        }

        self.models[id] = {k: v for k,
                           v in model_dict.items() if v is not None}
        self.save_metadata()

        if checkpoint is not None:
            if self.checkpoint_type == "pickle":
                import pickle

                path = os.path.join(
                    self.monitor_directory, self.model_dirname, f"{id}.pkl"
                )
                print(f"Saving checkpoint in pickle file '{path}'")

                with open(path, "wb") as file:
                    pickle.dump(checkpoint, file)

            elif self.checkpoint_type == "torch":
                import torch

                path = os.path.join(
                    self.monitor_directory, self.model_dirname, f"{id}.pth"
                )
                print(f"Saving checkpoint in pytorch state dict file '{path}'")

                torch.save(checkpoint, path)

        if outputs is not None:
            if self.outputs_type == "pickle":
                import pickle

                path = os.path.join(
                    self.monitor_directory, self.model_output_dirname, f"{id}.pkl"
                )
                print(f"Saving model outputs in pickle file '{path}'")

                with open(path, "wb") as file:
                    pickle.dump(outputs, file)

            elif self.outputs_type == "json":
                path = os.path.join(
                    self.monitor_directory, self.model_output_dirname, f"{id}.json"
                )
                print(f"Saving model outputs in JSON file '{path}'")

                with open(path, "w") as file:
                    json.dump(outputs, file)

            elif self.outputs_type == "numpy":
                import numpy as np

                path = os.path.join(
                    self.monitor_directory, self.model_output_dirname, f"{id}.npy"
                )
                print(f"Saving model outputs in numpy array file '{path}'")

                np.save(path, outputs)

    def has_model_checkpoint(self, id):
        if self.checkpoint_type == "pickle":
            path = os.path.join(self.monitor_directory,
                                self.model_dirname, f"{id}.pkl")
        elif self.checkpoint_type == "torch":
            path = os.path.join(self.monitor_directory,
                                self.model_dirname, f"{id}.pth")
        else:
            return False
        return os.path.exists(path)

    def load_model_checkpoint(self, id, **load_kwargs):
        """Loads a model checkpoint, or None if it doesn't exist."""
        if self.checkpoint_type == "pickle":
            import pickle

            path = os.path.join(self.monitor_directory,
                                self.model_dirname, f"{id}.pkl")
            if not os.path.exists(path):
                return None

            print(f"Loading checkpoint from pickle file '{path}'")

            with open(path, "rb") as file:
                return pickle.load(file, **load_kwargs)

        elif self.checkpoint_type == "torch":
            import torch

            path = os.path.join(self.monitor_directory,
                                self.model_dirname, f"{id}.pth")
            if not os.path.exists(path):
                return None
            print(f"Loading checkpoint from pytorch state dict file '{path}'")

            return torch.load(path, **load_kwargs)

        return None

    def get_model_outputs(self, id):
        """Loads a set of model outputs, or None if it doesn't exist."""
        if self.outputs_type == "pickle":
            import pickle

            path = os.path.join(
                self.monitor_directory, self.model_output_dirname, f"{id}.pkl"
            )
            if not os.path.exists(path):
                return None
            print(f"Loading model outputs from pickle file '{path}'")

            with open(path, "rb") as file:
                return pickle.load(file)

        elif self.outputs_type == "json":
            path = os.path.join(
                self.monitor_directory, self.model_output_dirname, f"{id}.json"
            )
            if not os.path.exists(path):
                return None
            print(f"Loading model outputs from JSON file '{path}'")

            with open(path, "r") as file:
                return json.load(file)

        elif self.outputs_type == "numpy":
            import numpy as np

            path = os.path.join(
                self.monitor_directory, self.model_output_dirname, f"{id}.npy"
            )
            if not os.path.exists(path):
                return None
            print(f"Loading model outputs from numpy array file '{path}'")

            return np.load(path)

    def __contains__(self, model_id):
        """
        Returns whether the model with the given ID has already been created.
        """
        return model_id in self.models
