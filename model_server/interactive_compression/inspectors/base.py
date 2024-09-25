"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import os
import json
import collections


class Prediction:
    def __init__(self, id, label, pred, classes=None):
        self.id = id
        self.label = label
        self.pred = pred
        self.classes = classes

    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "pred": self.pred,
            "classes": self.classes or [],
        }


class TaskLogger:
    """
    A helper class for writing out JSON files from task results.
    """

    def __init__(self, output_path=None):
        self.output_path = output_path

    def progress(self, message, progress_val=None):
        """
        Logs that the task is in progress with a message and optional
        determinate progress value.

        :param message: String message to show in the UI representing
            what the task is doing
        :param progress_val: If provided, a number from 0 to 1 indicating
            the numerical progress of the task
        """
        with open(os.path.join(self.output_path, "output.json"), "w") as file:
            json.dump(
                {
                    "status": "running",
                    "message": message,
                    **({"progress": progress_val} if progress_val is not None else {}),
                },
                file,
            )

    def complete(self, result):
        """
        Writes the given output as a successful task run.

        :param result: A JSON-serializable object to be written to file.
        """
        with open(os.path.join(self.output_path, "output.json"), "w") as file:
            json.dump(
                {
                    "status": "complete",
                    "result": result,
                },
                file,
            )


class ModelInspectorBase(TaskLogger):
    """
    A base class for classes that implement fundamental inspection operations for models to
    be displayed in the visualization tool.
    """

    def __init__(self):
        self.output_path = None

    def sparsity(self, model_id):
        """
        Returns information about whether each parameter in the model is zero or not.

        :param model_id: A string ID representing a model to load.

        :returns: A dictionary of the form:
            ```
            {
                "module.submodule.layer": {
                    "module_name": str,
                    "parameter_name": str,
                    "num_zeros": int,
                    "num_parameters": int,
                    "sparsity_mask": {
                        "path": str,
                        "format": "hdf5", # currently only hdf5 is supported
                        "dataset_name": str
                    }
                },
                # ...
            }
            ```
        """
        raise NotImplementedError

    def weights(self, model_id):
        raise NotImplementedError

    def predictions(self, model_id):
        """
        Generates or loads predictions for a specific model.

        :param model_id: A string ID representing a model to load.

        :returns: A dictionary of the form:
            ```
            {
                "predictions": [
                    {
                        "id": int,
                        "label": str,
                        "pred": str,
                        "classes": List[str],
                    },
                    #...
                ],
                "comparisons": {
                    "Comparison Name": {
                        "comparison": "difference" | "abs_difference" | "difference_argmax" | "abs_difference_argmax" | "ratio" | "ratio_argmax" | "mse" | "kl_divergence",
                        "vectors": {
                            "format": "hdf5",
                            "path": str,
                            "dataset_name": str
                        }
                    },
                    # ...
                }
            }
            ```
        """
        raise NotImplementedError

    def activations(self, model_id):
        """
        Computes output tensors for all modules in the network and saves them
        to HDF5 file(s). The first dimension of each tensor should be the batch
        dimension, which will be averaged through after comparison.

        :param model_id: A string ID representing a model to load.

        :returns: A dictionary of the form:
            ```
            {
                "module.submodule.layer": {
                    "module_name": str,
                    "parameter_name": str,
                    "num_parameters": int,
                    "activations": {
                        "path": str,
                        "format": "hdf5", # currently only hdf5 is supported
                        "dataset_name": str
                    }
                },
                # ...
            }
            ```
        """
        raise NotImplementedError

    def instances(self, ids):
        raise NotImplementedError

    def __call__(self, args, output_path):
        self.output_path = output_path

        try:
            if args["command"] == "sparsity":
                self.progress(f"Retrieving sparsity for model {args['model_id']}")
                results = self.sparsity(args["model_id"])
                self.complete(results)

            elif args["command"] == "weights":
                self.progress(f"Retrieving weights for model {args['model_id']}")
                results = self.weights(args["model_id"])
                self.complete(results)

            elif args["command"] == "activations":
                self.progress("Computing activations")
                results = self.activations(args["model_id"])
                self.complete(results)

            elif args["command"] == "predictions":
                self.progress("Computing predictions")
                results = self.predictions(args["model_id"])
                self.complete(results)

            elif args["command"] == "instances":
                self.progress("Retrieving instances")
                results = self.instances(args["ids"])
                self.complete(results)
            else:
                with open(os.path.join(output_path, "output.json"), "w") as file:
                    json.dump(
                        {
                            "status": "error",
                            "message": f"Unrecognized command {args['command']}",
                        },
                        file,
                    )
        except Exception as e:
            with open(os.path.join(output_path, "output.json"), "w") as file:
                json.dump({"status": "error", "message": str(e)}, file)
            raise e
