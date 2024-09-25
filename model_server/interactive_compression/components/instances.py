"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import numpy as np
import h5py
import traitlets
from ..sockets import SocketRequestable, socket_route, socket_async_route, bound_method
from .. import utils
import os
import json
from apscheduler.jobstores.base import JobLookupError


def calculate_prediction_differences(args, output_dir):
    results = args["results"]
    base_model_id = args["base_model_id"]
    model_ids = args["model_ids"]
    base_model_index = model_ids.index(base_model_id)

    # Currently only supports hdf5 format

    all_differences = [
        {
            "id": item["id"],
            "label": item["label"],
            "classes": item["classes"],
            "predictions": {},
            "comparisons": {},
        }
        for item in results[base_model_index]["predictions"]
    ]

    # Populate model predictions (no vectors required)
    for model_id, result in zip(model_ids, results):
        for output_row, model_pred in zip(all_differences, result["predictions"]):
            assert (
                output_row["id"] == model_pred["id"]
            ), "Instance IDs do not match between model prediction outputs"
            output_row["predictions"][model_id] = {
                "pred": model_pred["pred"],
            }

    print("all_differences has", len(all_differences))

    # Populate comparisons for each defined comparison
    base_comp_info = results[base_model_index]["comparisons"]
    for comparison_name, comparison_info in base_comp_info.items():
        assert all(
            comparison_name in result["comparisons"]
            and result["comparisons"][comparison_name]["comparison"]
            == comparison_info["comparison"]
            for result in results
        ), f"Models have inconsistent or missing comparison information for comparison '{comparison_name}'"

        vector_files = [
            h5py.File(result["comparisons"][comparison_name]["vectors"]["path"], "r")
            for result in results
        ]

        comparison_function = comparison_info["comparison"]

        base_vectors = vector_files[base_model_index][
            comparison_info["vectors"]["dataset_name"]
        ][:].astype(np.float64)

        for model_id, result, vector_file in zip(model_ids, results, vector_files):
            model_vectors = vector_file[
                result["comparisons"][comparison_name]["vectors"]["dataset_name"]
            ][:].astype(np.float64)
            assert len(model_vectors) == len(
                base_vectors
            ), f"Expected comparison vectors to have same shape, got {model_vectors.shape} and {base_vectors.shape}"
            for output_row, model_vector, base_vector in zip(
                all_differences, model_vectors, base_vectors
            ):
                difference = 0
                raw_value = None
                if comparison_function == "mse":
                    difference = ((model_vector - base_vector) ** 2).mean()
                elif comparison_function == "kl_divergence":
                    difference = max(
                        (
                            base_vector
                            * np.log2((base_vector + 1e-6) / (model_vector + 1e-6))
                        ).sum(),
                        0,
                    )
                elif comparison_function == "ratio":
                    # mean will be taken if the vector has more than one element
                    raw_value = model_vector
                    difference = model_vector / (base_vector + 1e-6)
                    if isinstance(raw_value, np.ndarray):
                        raw_value = raw_value.mean()
                    if isinstance(difference, np.ndarray):
                        difference = difference.mean()
                elif comparison_function == "ratio_argmax":
                    logit_idx = np.argmax(base_vector)
                    base_logit = base_vector.max()
                    difference = model_vector[logit_idx] / (base_logit + 1e-6)
                    raw_value = model_vector[logit_idx]
                elif comparison_function.endswith("difference"):
                    # mean will be taken if the vector has more than one element
                    raw_value = model_vector
                    difference = model_vector - base_vector
                    if isinstance(raw_value, np.ndarray):
                        raw_value = raw_value.mean()
                    if isinstance(difference, np.ndarray):
                        difference = difference.mean()
                    if comparison_function.startswith("abs_"):
                        difference = np.abs(difference)
                elif comparison_function.endswith("difference_argmax"):
                    logit_idx = np.argmax(base_vector)
                    base_logit = base_vector.max()
                    difference = model_vector[logit_idx] - base_logit
                    raw_value = model_vector[logit_idx]
                    if comparison_function.startswith("abs_"):
                        difference = np.abs(difference)
                else:
                    raise ValueError(
                        f"Unsupported comparison function '{comparison_function}'"
                    )
                output_row["comparisons"].setdefault(comparison_name, {})
                output_row["comparisons"][comparison_name][model_id] = {
                    "difference": difference
                }
                if raw_value is not None:
                    output_row["comparisons"][comparison_name][model_id][
                        "value"
                    ] = raw_value

        for file in vector_files:
            file.close()

    with open(os.path.join(output_dir, "output.json"), "w") as file:
        json.dump(
            {
                "status": "complete",
                "result": {
                    "predictions": utils.standardize_json(all_differences),
                    "type": "prediction_comparisons",
                    "comparisons": results[0]["comparisons"],
                },
            },
            file,
        )


class InstanceDetailComponent(traitlets.HasTraits, SocketRequestable):
    """
    Class that shows details about a small number of models (as many
    as can be loaded in CPU).
    """

    model_list = traitlets.List([]).tag(sync=True)
    base_model = traitlets.Unicode("").tag(sync=True)
    comparison_models = traitlets.List([]).tag(sync=True)
    has_task_runner = traitlets.Bool(False).tag(sync=True)

    error_message = traitlets.Unicode("").tag(sync=True)
    loading_message = traitlets.Unicode("").tag(sync=True)
    loading_progress = traitlets.Float(0.0).tag(sync=True)

    predictions = traitlets.List([]).tag(sync=True)
    comparison_names = traitlets.List([]).tag(sync=True)
    comparison_options = traitlets.Dict({}).tag(sync=True)

    def __init__(self, models, task_runner, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_list = [model["id"] for model in models]
        self.has_task_runner = task_runner is not None
        self.task_runner = task_runner
        self.connected = False
        self.active_task_ids = set()

    @bound_method
    @socket_async_route("get_instances")
    def get_instances(self, callback, args):
        instance_ids = list(set(args["ids"]))

        def receive_results(task_id, results):
            if results["status"] == "error":
                callback({"error": results.get("message", "Error loading instances")})
                return
            elif task_id not in self.active_task_ids:
                # The job was stopped
                print("Instance task not active")
                return
            elif results["status"] == "complete":
                print("Completing instance task")
                callback(results["result"])
                self.active_task_ids.remove(task_id)

        result = self.task_runner.run_task(
            {"command": "instances", "ids": instance_ids}, receive_results
        )
        self.active_task_ids.add(result["task_id"])
        receive_results(result["task_id"], result)

    @bound_method
    @socket_route("compute_predictions")
    def compute_predictions(self, args):
        if self.loading_message:
            print("Already loading")
            return
        if not self.task_runner:
            self.error_message = "No task runner is configured"
            return
        model_ids = args["comparison_models"]
        base_model_id = args["base_model"]
        if not len(model_ids) or not base_model_id:
            self.error_message = "No model IDs or base model provided"
            return

        def receive_results(task_ids, intermediate_results):
            if any(result["status"] == "error" for result in intermediate_results):
                self.error_message = "One or more tasks failed to run"
                self.active_task_ids -= set(task_ids)
                self.loading_message = ""
            elif not all(task_id in self.active_task_ids for task_id in task_ids):
                # The job was stopped
                print("Canceling processing for instance details")
                return
            elif len(intermediate_results) == len(model_ids) and all(
                result["status"] == "complete" for result in intermediate_results
            ):
                print("Received results", task_ids, len(intermediate_results))
                self.active_task_ids -= set(task_ids)

                def receive_processed_results(task_id, results):
                    if results["status"] == "error":
                        self.predictions = []
                        self.active_task_ids -= set([task_id])
                        self.error_message = results.get(
                            "message", "Post-processing failed"
                        )
                        self.loading_message = ""
                        return
                    elif task_id not in self.active_task_ids:
                        # The job was stopped
                        print("Canceling post-processing for instance details")
                        return
                    elif results["status"] != "complete":
                        self.loading_message = results.get(
                            "message", "Aggregating results"
                        )
                        self.loading_progress = (
                            len(model_ids) + 1 + results.get("progress", 0.0)
                        ) / (len(model_ids) + 2)
                        return
                    final_result = results["result"]
                    print(
                        "Completing instance details task,",
                        self.active_task_ids,
                        task_id,
                        len(final_result["predictions"]),
                    )
                    self.active_task_ids -= set([task_id])
                    self.comparison_models = model_ids
                    self.base_model = base_model_id
                    self.predictions = final_result["predictions"]
                    self.comparison_names = sorted(final_result["comparisons"].keys())
                    self.comparison_options = {
                        k: v["options"]
                        for k, v in final_result["comparisons"].items()
                        if "options" in v
                    }
                    self.loading_message = ""

                initial_result = self.task_runner.run_task(
                    {
                        "command": "calculate_prediction_differences",
                        "model_ids": model_ids,
                        "base_model_id": base_model_id,
                        "results": [r["result"] for r in intermediate_results],
                    },
                    receive_processed_results,
                    target_fn=calculate_prediction_differences,
                )
                self.active_task_ids.add(initial_result["task_id"])
                receive_processed_results(initial_result["task_id"], initial_result)
            else:
                running_tasks = [
                    r
                    for r in intermediate_results
                    if r["status"] in ("running", "waiting")
                ]
                if len(running_tasks):
                    self.loading_message = running_tasks[0].get(
                        "message", "Loading predictions"
                    )
                    if len(running_tasks) > 1:
                        self.loading_message += f" and {len(running_tasks) - 1} other task{'s' if len(running_tasks) - 1 > 1 else ''}"
                elif len(running_tasks) == 0:
                    self.loading_message = "Retrieving results"
                self.loading_progress = 1 - (len(running_tasks) + 1) / (
                    len(model_ids) + 2
                )

        print("Starting predictions job")
        initial_results = self.task_runner.run_tasks(
            [
                {"command": "predictions", "model_id": model_id}
                for model_id in model_ids
            ],
            receive_results,
        )
        task_ids = [t["task_id"] for t in initial_results]
        self.active_task_ids |= set(task_ids)
        receive_results(task_ids, initial_results)

    @traitlets.observe("comparison_models", "base_model")
    def changed_prediction_arguments(self, change):
        # Without computing the predictions, check whether the predictions
        # have already been computed
        model_ids = (
            change.new if change.name == "comparison_models" else self.comparison_models
        )
        base_model = change.new if change.name == "base_model" else self.base_model
        initial_results = self.task_runner.run_tasks(
            [
                {"command": "predictions", "model_id": model_id}
                for model_id in model_ids
            ],
            None,
            dry_run=True,
        )
        if not all(
            result is not None and result["status"] == "complete"
            for result in initial_results
        ):
            self.predictions = []
            return

        final_result = self.task_runner.run_task(
            {
                "command": "calculate_prediction_differences",
                "model_ids": model_ids,
                "base_model_id": base_model,
                "results": [r["result"] for r in initial_results],
            },
            None,
            target_fn=calculate_prediction_differences,
            dry_run=True,
        )
        if not (final_result is not None and final_result["status"] == "complete"):
            self.predictions = []
            return

        self.predictions = final_result["result"]["predictions"]
        self.comparison_names = sorted(final_result["result"]["comparisons"].keys())
        self.comparison_options = {
            k: v["options"]
            for k, v in final_result["result"]["comparisons"].items()
            if "options" in v
        }
        self.loading_message = ""

    @bound_method
    @socket_route("stop_updating")
    def stop_updating_predictions(self, _):
        if not self.loading_message:
            print("Not loading predictions")
            return
        for task_id in self.active_task_ids:
            try:
                self.task_runner.stop_task(task_id)
            except JobLookupError:
                print(f"No job with task ID {task_id}")
        self.active_task_ids = set()
        self.loading_message = ""
        self.loading_progress = 1.0
        self.error_message = ""

    def connect(self):
        print("Connected")
        self.connected = True
        if len(self.comparison_models) and self.base_model:
            self.compute_predictions(
                {
                    "comparison_models": self.comparison_models,
                    "base_model": self.base_model,
                }
            )
