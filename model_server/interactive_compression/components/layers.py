"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import numpy as np
import h5py
import traitlets
from ..sockets import SocketRequestable, socket_route, socket_async_route, bound_method
from .. import utils
from ..inspectors.base import TaskLogger
import os
import json
from apscheduler.jobstores.base import JobLookupError


def calculate_sparsity_changes(base_model_id, model_ids, results):
    base_model_idx = model_ids.index(base_model_id)
    assert base_model_idx >= 0, "Base model not in list of model IDs"

    open_files = {}
    leaf_nodes = {}
    for module_name in results[base_model_idx]["result"]:
        leaf_nodes[module_name] = {
            "parent": ".".join(module_name.split(".")[:-1])
            if "." in module_name
            else "",
            "children": [],
            "data": {},
        }

        # Read base model
        module_info = results[base_model_idx]["result"][module_name]
        base_zeros = module_info["num_zeros"]

        # Include base model stats
        leaf_nodes[module_name]["data"][base_model_id] = {
            "num_zeros": base_zeros,
            "num_parameters": module_info["num_parameters"],
        }

        assert "sparsity_mask" in module_info
        sparsity_mask_info = module_info["sparsity_mask"]
        if sparsity_mask_info["format"] == "hdf5":
            if sparsity_mask_info["path"] not in open_files:
                open_files[sparsity_mask_info["path"]] = h5py.File(
                    sparsity_mask_info["path"], "r"
                )
            base_tensor = open_files[sparsity_mask_info["path"]][
                sparsity_mask_info["dataset_name"]
            ][:]
        else:
            raise ValueError(
                f"Unknown sparsity_mask format {sparsity_mask_info['format']}"
            )

        for model_id, result in zip(model_ids, results):
            if model_id == base_model_id:
                continue
            module_info = result["result"][module_name]

            # Read sparsity mask from h5 file
            assert "sparsity_mask" in module_info
            sparsity_mask_info = module_info["sparsity_mask"]
            if sparsity_mask_info["format"] == "hdf5":
                if sparsity_mask_info["path"] not in open_files:
                    open_files[sparsity_mask_info["path"]] = h5py.File(
                        sparsity_mask_info["path"], "r"
                    )
                tensor = open_files[sparsity_mask_info["path"]][
                    sparsity_mask_info["dataset_name"]
                ][:]
            else:
                raise ValueError(
                    f"Unknown sparsity_mask format {sparsity_mask_info['format']}"
                )

            # Compare to base model
            leaf_nodes[module_name]["data"][model_id] = {
                "num_zeros": module_info["num_zeros"] - base_zeros,
                "num_parameters": module_info["num_parameters"],
                "zero_base_only": int(((1 - base_tensor) * tensor).sum()),
                "zero_model_only": int((base_tensor * (1 - tensor)).sum()),
                "zero_both": int(((1 - base_tensor) * (1 - tensor)).sum()),
            }

    for file in open_files.values():
        file.close()

    # Add parents and children
    tree_nodes = {**leaf_nodes}
    for module_name in leaf_nodes:
        curr = module_name
        while curr:
            # Add to tree nodes with final data comparison
            parent_id = tree_nodes[curr]["parent"]
            if not parent_id:
                break
            if parent_id not in tree_nodes:
                tree_nodes[parent_id] = {
                    "parent": ".".join(parent_id.split(".")[:-1])
                    if "." in parent_id
                    else "",
                    "children": [],
                    "data": {id: {} for id in [base_model_id] + model_ids},
                }

            if curr not in tree_nodes[parent_id]["children"]:
                tree_nodes[parent_id]["children"].append(curr)
            for model_id in model_ids:
                for field, val in tree_nodes[module_name]["data"][model_id].items():
                    tree_nodes[parent_id]["data"][model_id][field] = (
                        tree_nodes[parent_id]["data"][model_id].get(field, 0) + val
                    )
            curr = tree_nodes[curr]["parent"]

    return {
        "nodes": tree_nodes,
        "type": "sparsity_mask",
        "comparison_models": model_ids,
        "base_model": base_model_id,
    }


def calculate_weight_changes(args, output_dir):
    results = args["results"]
    base_model_id = args["base_model_id"]
    model_ids = args["model_ids"]
    base_model_idx = model_ids.index(base_model_id)
    assert base_model_idx >= 0, "Base model ID not found in list of models"

    output = TaskLogger(output_dir)

    open_files = {}
    leaf_nodes = {}

    results_by_module = results[base_model_idx]["result"]

    # First compute the histogram bins by iterating over the base model and calculating
    # the min and max values
    min_value = 1e12
    max_value = -1e12
    output.progress("Computing histogram bins")
    for module_idx, module_name in enumerate(results_by_module):
        module_info = results[base_model_idx]["result"][module_name]

        assert "weights" in module_info
        weight_info = module_info["weights"]
        if weight_info["format"] == "hdf5":
            if weight_info["path"] not in open_files:
                open_files[weight_info["path"]] = h5py.File(weight_info["path"], "r")
            base_tensor = open_files[weight_info["path"]][weight_info["dataset_name"]][
                :
            ]
            min_value = min(min_value, base_tensor.min())
            max_value = max(max_value, base_tensor.max())

    hist_bins = utils.choose_histogram_bins(min_value, max_value, n_bins=30)
    diff_hist_bins = np.concatenate([np.linspace(0, 1, 6), [1e12]])
    print("diff bins:", diff_hist_bins)

    for module_idx, module_name in enumerate(results_by_module):
        output.progress(
            f"Aggregating {module_name}", module_idx / len(results_by_module)
        )

        leaf_nodes[module_name] = {
            "parent": ".".join(module_name.split(".")[:-1])
            if "." in module_name
            else "",
            "children": [],
            "data": {},
        }

        # Read base model
        module_info = results[base_model_idx]["result"][module_name]

        assert "weights" in module_info
        weight_info = module_info["weights"]
        if weight_info["format"] == "hdf5":
            if weight_info["path"] not in open_files:
                open_files[weight_info["path"]] = h5py.File(weight_info["path"], "r")
            base_tensor = open_files[weight_info["path"]][weight_info["dataset_name"]][
                :
            ]
            shape = base_tensor.shape
            base_tensor = base_tensor.flatten()
            base_hist = np.histogram(base_tensor, bins=hist_bins)[0]
            # Include base model stats
            leaf_nodes[module_name]["data"][base_model_id] = {
                "sum": np.abs(base_tensor).astype(np.float64).sum(),
                "square_sum": (base_tensor**2).astype(np.float64).sum(),
                "shape": shape,
                "num_parameters": len(base_tensor),
                "weight_histogram": {
                    "bins": hist_bins,
                    "values": base_hist,
                },
            }
        else:
            raise ValueError(f"Unknown weights format {weight_info['format']}")

        for model_id, result in zip(model_ids, results):
            if model_id == base_model_id:
                continue
            module_info = result["result"][module_name]

            # Read sparsity mask from h5 file
            assert "weights" in module_info
            weight_info = module_info["weights"]
            if weight_info["format"] == "hdf5":
                if weight_info["path"] not in open_files:
                    open_files[weight_info["path"]] = h5py.File(
                        weight_info["path"], "r"
                    )
                tensor = open_files[weight_info["path"]][weight_info["dataset_name"]][:]
                shape = tensor.shape
                tensor = tensor.flatten()
            else:
                raise ValueError(f"Unknown weights format {weight_info['format']}")

            # Compare to base model
            weight_diff_histograms = np.zeros(
                (len(hist_bins) - 1, len(diff_hist_bins) - 1), dtype=np.uint64
            )
            norm_diffs = np.abs(tensor - base_tensor) / np.abs(base_tensor)
            norm_diffs = np.where(
                np.isinf(norm_diffs), 1e9, np.where(np.isnan(norm_diffs), 0, norm_diffs)
            )
            for i in range(len(diff_hist_bins) - 1):
                weight_diff_histograms[:, i] = np.histogram(
                    tensor[
                        (norm_diffs >= diff_hist_bins[i])
                        & (norm_diffs < diff_hist_bins[i + 1])
                    ],
                    bins=hist_bins,
                )[0]
            leaf_nodes[module_name]["data"][model_id] = {
                "sum": np.abs(tensor).astype(np.float64).sum(),
                "square_sum": (tensor**2).astype(np.float64).sum(),
                "shape": shape,
                "num_parameters": len(tensor),
                "weight_histogram": {
                    "bins": hist_bins,
                    "values": np.histogram(tensor, bins=hist_bins)[0],
                },
                "difference_sum": np.abs(tensor - base_tensor).sum(),
                "weight_difference_histogram": {
                    "bins": [hist_bins, diff_hist_bins],
                    "values": weight_diff_histograms,
                },
            }

    for file in open_files.values():
        file.close()

    # Add parents and children
    tree_nodes = {**leaf_nodes}
    for module_name in leaf_nodes:
        curr = module_name
        while curr:
            # Add to tree nodes with final data comparison
            parent_id = tree_nodes[curr]["parent"]
            if not parent_id:
                break
            if parent_id not in tree_nodes:
                tree_nodes[parent_id] = {
                    "parent": ".".join(parent_id.split(".")[:-1])
                    if "." in parent_id
                    else "",
                    "children": [],
                    "data": {id: {} for id in [base_model_id] + model_ids},
                }

            if curr not in tree_nodes[parent_id]["children"]:
                tree_nodes[parent_id]["children"].append(curr)
            for model_id in model_ids:
                parent_item = tree_nodes[parent_id]["data"][model_id]
                current_item = tree_nodes[module_name]["data"][model_id]
                parent_item["sum"] = parent_item.get("sum", 0) + current_item["sum"]
                parent_item["num_parameters"] = (
                    parent_item.get("num_parameters", 0)
                    + current_item["num_parameters"]
                )
                parent_item["square_sum"] = (
                    parent_item.get("square_sum", 0) + current_item["square_sum"]
                )
                parent_item.setdefault(
                    "weight_histogram",
                    {
                        "bins": hist_bins,
                        "values": np.zeros(len(hist_bins) - 1),
                    },
                )
                parent_item["weight_histogram"]["values"] += current_item[
                    "weight_histogram"
                ]["values"]
                if model_id != base_model_id:
                    parent_item["difference_sum"] = (
                        parent_item.get("difference_sum", 0)
                        + current_item["difference_sum"]
                    )
                    parent_item.setdefault(
                        "weight_difference_histogram",
                        {
                            "bins": [hist_bins, diff_hist_bins],
                            "values": np.zeros(
                                (len(hist_bins) - 1, len(diff_hist_bins) - 1),
                                dtype=np.uint64,
                            ),
                        },
                    )
                    parent_item["weight_difference_histogram"][
                        "values"
                    ] += current_item["weight_difference_histogram"]["values"]
            curr = tree_nodes[curr]["parent"]

    output.complete(
        {
            "nodes": utils.standardize_json(tree_nodes),
            "type": "weight_changes",
            "comparison_models": model_ids,
            "base_model": base_model_id,
        }
    )


def calculate_activations(args, output_dir):
    results = args["results"]
    base_model_id = args["base_model_id"]
    model_ids = args["model_ids"]
    base_model_idx = model_ids.index(base_model_id)
    assert base_model_idx >= 0, "Base model ID not found in list of models"

    output = TaskLogger(output_dir)

    results_by_module = results[base_model_idx]["result"]

    nodes = {}

    # First compute the histogram bins by iterating over the base model and calculating
    # the min and max values
    hist_bins = {}
    output.progress("Computing histogram bins")
    for module_idx, module_name in enumerate(results_by_module):
        output.progress(
            f"Computing histogram bins for {module_name}",
            module_idx / len(results_by_module) * 0.5,
        )
        module_info = results[base_model_idx]["result"][module_name]

        assert "activations" in module_info
        weight_info = module_info["activations"]
        if weight_info["format"] == "hdf5":
            file = h5py.File(weight_info["path"], "r")
            for dataset_name in file.keys():
                base_tensor = file[dataset_name][:].astype(np.float64)
                shape = base_tensor.shape
                base_tensor = base_tensor.flatten()
                hist_bins[dataset_name] = utils.choose_histogram_bins(
                    base_tensor[np.isfinite(base_tensor)].min(), base_tensor[np.isfinite(base_tensor)].max(), n_bins=20
                )
                base_hist = np.histogram(base_tensor, bins=hist_bins[dataset_name])[0]

                nodes[dataset_name] = {
                    "parent": ".".join(dataset_name.split(".")[:-1])
                    if "." in dataset_name
                    else "",
                    "children": [],
                    "has_activations": True,
                    "data": {
                        base_model_id: {
                            "sum": np.abs(base_tensor).sum(),
                            "square_sum": (base_tensor**2).sum(),
                            "shape": shape,
                            "num_activations": len(base_tensor),
                            "histogram": {
                                "bins": hist_bins[dataset_name],
                                "values": base_hist,
                            },
                        }
                    },
                }

            file.close()

    diff_hist_bins = np.concatenate([np.linspace(0, 1, 6), [1e12]])

    # Now iterate over modules again and compute histograms of differences
    for module_idx, module_name in enumerate(results_by_module):
        output.progress(
            f"Computing differences for {module_name}",
            0.5 + module_idx / len(results_by_module) * 0.5,
        )
        module_info = results[base_model_idx]["result"][module_name]

        assert "activations" in module_info
        weight_info = module_info["activations"]
        if weight_info["format"] != "hdf5":
            raise ValueError("hdf5 format is required for activations")
        base_file = h5py.File(weight_info["path"], "r")
        base_tensors = {}
        for dataset_name in base_file.keys():
            base_tensors[dataset_name] = (
                base_file[dataset_name][:].flatten().astype(np.float64)
            )

        for model_id, result in zip(model_ids, results):
            if model_id == base_model_id:
                continue
            module_info = result["result"][module_name]

            # Read sparsity mask from h5 file
            assert "activations" in module_info
            weight_info = module_info["activations"]
            if weight_info["format"] != "hdf5":
                raise ValueError("hdf5 format is required for activations")

            file = h5py.File(weight_info["path"], "r")
            for dataset_name in file.keys():
                tensor = file[dataset_name][:].astype(np.float64)
                shape = tensor.shape
                tensor = tensor.flatten()

                # Compare to base model
                weight_diff_histograms = np.zeros(
                    (len(hist_bins[dataset_name]) - 1, len(diff_hist_bins) - 1),
                    dtype=np.uint64,
                )
                norm_diffs = np.abs(tensor - base_tensors[dataset_name]) / np.abs(
                    base_tensors[dataset_name]
                )
                norm_diffs = np.where(
                    np.isinf(norm_diffs),
                    1e9,
                    np.where(np.isnan(norm_diffs), 0, norm_diffs),
                )
                for i in range(len(diff_hist_bins) - 1):
                    weight_diff_histograms[:, i] = np.histogram(
                        tensor[
                            (norm_diffs >= diff_hist_bins[i])
                            & (norm_diffs < diff_hist_bins[i + 1])
                        ],
                        bins=hist_bins[dataset_name],
                    )[0]
                nodes[dataset_name]["data"][model_id] = {
                    "sum": np.abs(tensor).astype(np.float64).sum(),
                    "square_sum": (tensor**2).astype(np.float64).sum(),
                    "shape": shape,
                    "num_activations": len(tensor),
                    "histogram": {
                        "bins": hist_bins[dataset_name],
                        "values": np.histogram(tensor, bins=hist_bins[dataset_name])[0],
                    },
                    "difference_sum": np.abs(tensor - base_tensors[dataset_name]).sum(),
                    "difference_histogram": {
                        "bins": [hist_bins[dataset_name], diff_hist_bins],
                        "values": weight_diff_histograms,
                    },
                }
            file.close()

        base_file.close()

    # Add missing parents (but no aggregation)
    tree_nodes = {**nodes}
    for module_name in nodes:
        curr = module_name
        while curr:
            # Add to tree nodes with final data comparison
            parent_id = tree_nodes[curr]["parent"]
            print(curr, parent_id)
            if not parent_id:
                break
            if parent_id not in tree_nodes:
                tree_nodes[parent_id] = {
                    "parent": ".".join(parent_id.split(".")[:-1])
                    if "." in parent_id
                    else "",
                    "children": [],
                    "data": {},
                    "has_activations": False,
                }
            if curr not in tree_nodes[parent_id]["children"]:
                tree_nodes[parent_id]["children"].append(curr)

            curr = tree_nodes[curr]["parent"]

    output.complete(
        {
            "nodes": utils.standardize_json(tree_nodes),
            "type": "activations",
            "comparison_models": model_ids,
            "base_model": base_model_id,
        }
    )


class LayerDetailComponent(traitlets.HasTraits, SocketRequestable):
    """
    Class that shows details about a small number of models (as many
    as can be loaded in CPU).
    """

    model_list = traitlets.List([]).tag(sync=True)
    base_model = traitlets.Unicode("").tag(sync=True)
    comparison_models = traitlets.List([]).tag(sync=True)
    has_task_runner = traitlets.Bool(False).tag(sync=True)

    tensor_type = traitlets.Unicode("sparsity_mask").tag(sync=True)
    tensor_tree = traitlets.Dict({}).tag(sync=True)
    error_message = traitlets.Unicode("").tag(sync=True)
    loading_message = traitlets.Unicode("").tag(sync=True)
    loading_progress = traitlets.Float(0.0).tag(sync=True)

    def __init__(self, models, task_runner, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_list = [model["id"] for model in models]
        self.has_task_runner = task_runner is not None
        self.task_runner = task_runner
        self.connected = False
        self.active_task_ids = set()

    def compare_sparsity_masks(self, base_model_id, model_ids, dry_run=False):
        def receive_results(task_ids, results):
            print(
                "Receiving results",
                task_ids,
                self.active_task_ids,
                [r["status"] for r in results],
            )
            if not dry_run and any(result["status"] == "error" for result in results):
                self.error_message = "One or more tasks failed to run"
                self.active_task_ids -= set(task_ids)
                self.loading_message = ""
            elif not dry_run and not all(
                task_id in self.active_task_ids for task_id in task_ids
            ):
                # The job was stopped
                print("Canceling processing for sparsity")
                return
            elif len(results) == len(model_ids) and all(
                result["status"] == "complete" for result in results
            ):
                print("Received results", task_ids, self.active_task_ids)
                # Create the tensor tree
                self.loading_message = "Aggregating results"
                self.active_task_ids -= set(task_ids)
                self.loading_progress = 1.0

                self.tensor_tree = calculate_sparsity_changes(
                    base_model_id, model_ids, results
                )
                self.loading_message = ""
                self.comparison_models = model_ids
                self.base_model = base_model_id
                self.tensor_type = "sparsity_mask"
            elif dry_run:
                return
            else:
                running_tasks = [
                    r for r in results if r["status"] in ("running", "waiting")
                ]
                if len(running_tasks):
                    msg = running_tasks[0].get("message", "Loading weights")
                    if len(running_tasks) > 1:
                        msg += f" and {len(running_tasks) - 1} other task{'s' if len(running_tasks) - 1 > 1 else ''}"
                    self.loading_message = msg
                elif len(running_tasks) == 0:
                    self.loading_message = "Retrieving results"
                self.loading_progress = 1 - len(running_tasks) / len(model_ids)

        initial_results = self.task_runner.run_tasks(
            [{"command": "sparsity", "model_id": model_id} for model_id in model_ids],
            receive_results,
            dry_run=dry_run,
        )
        task_ids = [
            t["task_id"] if t and "task_id" in t else None for t in initial_results
        ]
        if not dry_run:
            self.active_task_ids |= set(task_ids)
        if dry_run and any(not r for r in initial_results):
            return
        receive_results(task_ids, initial_results)

    def compare_tensor_changes(
        self,
        base_model_id,
        model_ids,
        tensor_type="weight_changes",
        tensor_display_name="weights",
        post_processing_fn=calculate_weight_changes,
        dry_run=False,
    ):
        def receive_results(task_ids, intermediate_results):
            if not dry_run and any(
                result["status"] == "error" for result in intermediate_results
            ):
                self.error_message = "One or more tasks failed to run"
                self.active_task_ids -= set(task_ids)
                self.loading_message = ""
            elif not dry_run and not all(
                task_id in self.active_task_ids for task_id in task_ids
            ):
                # The job was stopped
                print("Canceling processing for", tensor_type)
                return
            elif len(intermediate_results) == len(model_ids) and all(
                result["status"] == "complete" for result in intermediate_results
            ):
                print("Received results")
                self.active_task_ids -= set(task_ids)

                def receive_processed_results(task_id, results):
                    if not dry_run and results["status"] == "error":
                        self.tensor_tree = {}
                        self.active_task_ids -= set([task_id])
                        self.error_message = results.get(
                            "message", "Post-processing failed"
                        )
                        self.loading_message = ""
                        return
                    elif not dry_run and task_id not in self.active_task_ids:
                        # The job was stopped
                        print("Canceling post-processing for", tensor_type)
                        return
                    elif results["status"] != "complete":
                        if dry_run:
                            return None

                        self.loading_message = results.get(
                            "message", "Aggregating results"
                        )
                        self.loading_progress = (
                            len(model_ids) + results.get("progress", 0.0)
                        ) / (len(model_ids) + 1)
                        return
                    print("Completed task for", tensor_type)
                    print("Results keys:", results["result"].keys())
                    self.active_task_ids -= set([task_id])
                    self.comparison_models = model_ids
                    self.base_model = base_model_id
                    self.tensor_type = tensor_type
                    self.tensor_tree = results["result"]
                    self.loading_message = ""

                initial_result = self.task_runner.run_task(
                    {
                        "command": "calculate_" + tensor_type,
                        "model_ids": model_ids,
                        "base_model_id": base_model_id,
                        "results": intermediate_results,
                    },
                    receive_processed_results,
                    target_fn=post_processing_fn,
                    dry_run=dry_run,
                )
                if not dry_run:
                    self.active_task_ids.add(initial_result["task_id"])
                    self.loading_message = "Aggregating results"
                if dry_run and not initial_result:
                    return
                receive_processed_results(initial_result["task_id"], initial_result)
            elif dry_run:
                return None
            else:
                running_tasks = [
                    r
                    for r in intermediate_results
                    if r["status"] in ("running", "waiting")
                ]
                if len(running_tasks):
                    msg = running_tasks[0].get(
                        "message", "Loading " + tensor_display_name
                    )
                    if len(running_tasks) > 1:
                        msg += f" and {len(running_tasks) - 1} other task{'s' if len(running_tasks) - 1 > 1 else ''}"
                    self.loading_message = msg
                elif len(running_tasks) == 0:
                    self.loading_message = "Retrieving results"
                self.loading_progress = 1 - len(running_tasks) / (len(model_ids) + 1)

        initial_results = self.task_runner.run_tasks(
            [
                {
                    "command": "weights"
                    if tensor_type == "weight_changes"
                    else "activations",
                    "model_id": model_id,
                }
                for model_id in model_ids
            ],
            receive_results,
            dry_run=dry_run,
        )
        task_ids = [
            t["task_id"] if t and "task_id" in t else None for t in initial_results
        ]
        if not dry_run:
            self.active_task_ids |= set(task_ids)
        if dry_run and any(not r for r in initial_results):
            return
        return receive_results(task_ids, initial_results)

    @bound_method
    @socket_route("update_tensors")
    def update_tensors(self, args):
        if self.loading_message:
            print("Already loading tensors")
            return
        comparison_models = args.get("comparison_models", [])
        base_model = args.get("base_model", "")
        tensor_type = args.get("tensor_type", "sparsity_mask")
        if not self.connected:
            print("Not connected")
            return
        if not comparison_models or not base_model:
            self.error_message = "Invalid model selection"
            return
        if not self.has_task_runner:
            self.error_message = "The server has no task runner configured."
            return
        print("Loading tensors", tensor_type)
        self.error_message = ""
        self.loading_message = "Loading"
        self.loading_progress = 0.0
        self.start_tasks(base_model, comparison_models, tensor_type)

    def start_tasks(self, base_model, comparison_models, tensor_type, dry_run=False):
        if tensor_type == "sparsity_mask":
            self.compare_sparsity_masks(base_model, comparison_models, dry_run=dry_run)
        elif tensor_type == "weight_changes":
            self.compare_tensor_changes(
                base_model,
                comparison_models,
                tensor_type="weight_changes",
                tensor_display_name="weights",
                post_processing_fn=calculate_weight_changes,
                dry_run=dry_run,
            )
        elif tensor_type == "activations":
            self.compare_tensor_changes(
                base_model,
                comparison_models,
                tensor_type="activations",
                tensor_display_name="activations",
                post_processing_fn=calculate_activations,
                dry_run=dry_run,
            )
        else:
            self.error_message = f"Unknown tensor type '{tensor_type}'"
            self.loading_message = ""

    @traitlets.observe("comparison_models", "base_model", "tensor_type")
    def changed_tensor_arguments(self, change):
        # Without actually fetching the weights, check whether the weight comparisons
        # have already been computed
        model_ids = (
            change.new if change.name == "comparison_models" else self.comparison_models
        )
        base_model = change.new if change.name == "base_model" else self.base_model
        tensor_type = change.new if change.name == "tensor_type" else self.tensor_type
        if (
            self.tensor_tree is not None
            and set(model_ids) == set(self.tensor_tree.get("comparison_models", []))
            and base_model == self.tensor_tree.get("base_model", None)
            and tensor_type == self.tensor_tree.get("type", None)
        ):
            # No changes, just return
            return
        if base_model not in model_ids:
            print("Base model must be in the model ID list")
            return
        model_ids.remove(base_model)
        model_ids.insert(0, base_model)
        print("Starting tasks for", model_ids, base_model, tensor_type)
        self.tensor_tree = {}
        self.start_tasks(base_model, model_ids, tensor_type, dry_run=True)

    @bound_method
    @socket_route("stop_updating")
    def stop_updating_tensors(self, _):
        if not self.loading_message:
            print("Not loading tensors")
            return
        for task_id in self.active_task_ids:
            try:
                self.task_runner.stop_task(task_id)
            except JobLookupError:
                print(f"No job with task ID {task_id}")
        self.active_task_ids = set()
        self.loading_message = ""
        self.loading_message = ""
        self.loading_progress = 1.0
        self.error_message = ""

    def connect(self):
        print("Connected")
        self.connected = True
