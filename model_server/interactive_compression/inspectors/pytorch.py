"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

from .base import ModelInspectorBase


import h5py
import torch
import os
import numpy as np
from .pytorch_activations import TensorOutputTracker, make_simple_quantizer
from interactive_compression.utils import empty_device_cache
import re


def _get_modules(model):
    prunable_modules = []
    module_names = []
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            if module.weight.requires_grad:
                prunable_modules.append((module, "weight"))
                module_names.append(name)
    return prunable_modules, module_names


class PyTorchModelInspector(ModelInspectorBase):
    def __init__(self):
        super().__init__()

    def get_model(self, model_id, for_inference=False):
        """
        Loads a model with the given ID.

        :param model_id: String ID to load
        :param for_inference: True if the model will be used to run
            inference (and should be transferred to a cuda device if
            available), False if it is simply being loaded to read
            weights
        :returns: A PyTorch Module object representing the model
        """
        raise NotImplementedError(
            "get_model must be implemented before using this model inspector"
        )

    def get_eval_dataloader(self):
        """
        Creates or retrieves a dataloader for running evaluations on models.

        :returns: A PyTorch DataLoader or iterable of data batches
            that can be run through a model.
        """
        raise NotImplementedError(
            "get_eval_dataloader must be implemented before using this model inspector"
        )

    def iter_modules(self, model):
        """
        Creates a generator that yields pairs (module_name, (layer, parameter_name)),
        where module_name is a string name defining the module hierarchy separated by
        '.', layer is a PyTorch layer object, and parameter_name is the name of the
        parameter to display.
        """
        layers_to_prune, module_names = _get_modules(model)
        return zip(module_names, layers_to_prune)

    # A list of patterns for module names to exclude from activation calculations
    ACTIVATION_EXCLUDE_PATTERNS = []

    def iter_activation_modules(self, model):
        """
        Creates a generator that yields pairs (module_name, layer),
        where module_name is a string name defining the module hierarchy separated by
        '.', and layer is a PyTorch layer object for which activations should be
        collected. The default behavior yields from the model's named_modules.
        """
        for name, module in model.named_modules():
            if any(
                re.search(pattern, name, flags=re.I) is not None
                for pattern in self.ACTIVATION_EXCLUDE_PATTERNS
            ):
                continue
            yield name, module

    def sparsity(self, model_id):
        model = self.get_model(model_id)

        tensor_filepath = os.path.join(self.output_path, "sparsity_masks.h5")
        tensor_file = h5py.File(tensor_filepath, "w")

        results = {}
        for name, (layer, param_name) in self.iter_modules(model):
            num_zeros = 0
            num_parameters = torch.numel(getattr(layer, param_name))
            sparsity_mask = torch.ones(*getattr(layer, param_name).shape)
            if hasattr(layer, param_name + "_mask"):
                num_zeros = (
                    (1.0 - getattr(layer, param_name + "_mask").float()).sum().item()
                )
                num_parameters = torch.numel(getattr(layer, param_name + "_mask"))
                sparsity_mask = getattr(layer, param_name + "_mask")
            else:
                num_zeros = (getattr(layer, param_name) == 0).sum().item()
                sparsity_mask = getattr(layer, param_name) != 0
            tensor_name = f"{name}.{param_name}"
            tensor_file.create_dataset(
                tensor_name, data=sparsity_mask.cpu().numpy().astype(np.uint8)
            )

            results[tensor_name] = {
                "module_name": name,
                "parameter_name": param_name,
                "num_zeros": num_zeros,
                "num_parameters": num_parameters,
                "sparsity_mask": {
                    "path": tensor_filepath,
                    "format": "hdf5",
                    "dataset_name": tensor_name,
                },
            }

        return results

    def weights(self, model_id):
        model = self.get_model(model_id)

        tensor_filepath = os.path.join(self.output_path, "weights.h5")
        tensor_file = h5py.File(tensor_filepath, "w")

        results = {}
        with torch.no_grad():
            for name, (layer, param_name) in self.iter_modules(model):
                tensor_name = f"{name}.{param_name}"
                if hasattr(layer, param_name + "_mask") and hasattr(
                    layer, param_name + "_orig"
                ):
                    weights = getattr(layer, param_name + "_mask") * getattr(
                        layer, param_name + "_orig"
                    )
                else:
                    weights = getattr(layer, param_name)
                tensor_file.create_dataset(
                    tensor_name, data=weights.cpu().numpy().astype(np.float16)
                )

                results[tensor_name] = {
                    "module_name": name,
                    "parameter_name": param_name,
                    "weights": {
                        "path": tensor_filepath,
                        "format": "hdf5",
                        "dataset_name": tensor_name,
                    },
                }

        return results

    def get_activations_dataloader(self):
        """
        Creates or retrieves a dataloader for collecting activations. The
        default behavior is to use the first batch of the eval dataloader.

        :returns: A PyTorch DataLoader or iterable of data batches
            that can be run through a model.
        """
        return [next(iter(self.get_eval_dataloader()))]

    def activations(self, model_id):
        model = self.get_model(model_id, for_inference=True)
        loader = self.get_activations_dataloader()

        TensorOutputTracker.set_output_dir(self.output_path)
        TensorOutputTracker.clear_all()
        TensorOutputTracker.output_prefix = model_id

        def empty_cache_hook(m, i, o):
            print("Emptying device cache")
            empty_device_cache()

        module_names = []
        TensorOutputTracker.quantization_function = make_simple_quantizer(
            np.float16
        )  # make_linear_bin_quantizer(hist_bins)

        last_name = None

        for name, module in self.iter_activation_modules(model):
            if not name:
                continue
            TensorOutputTracker.create_output_hook(name, module)
            module_names.append(name)
            if last_name is None or last_name.split(".")[0] != name.split(".")[0]:
                module.register_forward_hook(empty_cache_hook)
            last_name = name

        with torch.no_grad():
            for i, batch in enumerate(loader):
                _ = self.inference_batch(model, batch, i)
                empty_device_cache()

        # List out the files written
        summary = {}
        for name in module_names:
            path = TensorOutputTracker.get_path_for_module(name)
            if os.path.exists(path):
                summary[name] = {
                    "activations": {
                        "path": path,
                        "format": "hdf5",
                    }
                }
            else:
                print("Missing activations file:", path)

        return summary

    def inference_batch(self, model, batch, batch_index):
        """
        Runs an inference pass on a single batch of data, and returns a list of
        Prediction objects as well as a dictionary of 2D tensors representing
        a vector for each instance in the batch.

        :param model: A PyTorch Module object representing a model to
            run.
        :param batch: A batch of data (one item yielded from the dataloader
            returned by `get_eval_dataloader`).
        :param batch_index: The number of the current batch as returned from the
            dataloader. The implementation can use this to assign IDs to each
            prediction, though it is not required. If IDs based on indexes from the
            dataloader are used, the user is responsible for making sure the same
            instances are returned for these IDs in the `instances()` method.
        :returns: A tuple consisting of (1) a list of Prediction objects, one
            for each data item in the batch; and (2) a dictionary of 1-2D numpy arrays.
            Each key in the dictionary must be returned consistently with every
            call to inference_batch. The values should be 1 or 2D arrays, where the
            number of rows is the number of instances in the batch and the number
            of columns can be arbitrary. The keys here will correspond to the dataset
            names returned by `comparison_metrics`.
        """
        raise NotImplementedError(
            "inference_batch must be implemented to get predictions"
        )

    def comparison_metrics(self):
        """
        Defines what comparison metrics will be displayed in the interface. This
        method must return at least one comparison metric.

        :returns: A dictionary of comparison metrics, where each key is a user-
            defined name for the metric, and the value is a tuple (comparison_type,
            dataset_name) where dataset_name is the name of a tensor set returned by
            `inference_batch` and comparison_type can be one of the following:
            * "difference"
            * "abs_difference"
            * "difference_argmax"
            * "abs_difference_argmax"
            * "ratio"
            * "ratio_argmax"
            * "mse"
            * "kl_divergence"
        """
        raise NotImplementedError(
            "comparison_metrics must be implemented to get predictions"
        )

    def predictions(self, model_id):
        model = self.get_model(model_id, for_inference=True)
        dataloader = self.get_eval_dataloader()

        prediction_outputs = []
        tensor_results = {}
        with torch.no_grad():
            for batch_idx, inputs in enumerate(dataloader):
                self.progress("Computing predictions", batch_idx / len(dataloader))

                batch_outputs, tensors = self.inference_batch(model, inputs, batch_idx)

                assert (
                    len(tensors) > 0
                ), "At least one output tensor must be provided for comparison"

                prediction_outputs += [p.to_dict() for p in batch_outputs]
                for tensor_name, tensor_data in tensors.items():
                    assert (
                        batch_idx == 0 or tensor_name in tensor_results
                    ), f"Tensor result '{tensor_name}' not returned with all data batches"
                    tensor_results.setdefault(tensor_name, []).append(tensor_data)

        tensor_filepath = os.path.join(self.output_path, "tensors.h5")
        tensor_file = h5py.File(tensor_filepath, "w")

        for tensor_name, tensor_data in tensor_results.items():
            tensor_data = np.concatenate(tensor_data, axis=0)
            tensor_file.create_dataset(tensor_name, data=tensor_data)

        comparisons = self.comparison_metrics()
        return {
            "predictions": prediction_outputs,
            "comparisons": {
                comparison_name: {
                    "comparison": comp_info[0],
                    "vectors": {
                        "format": "hdf5",
                        "path": tensor_filepath,
                        "dataset_name": comp_info[1],
                    },
                    "options": comp_info[2] if len(comp_info) > 2 else {},
                }
                for comparison_name, comp_info in comparisons.items()
            },
        }
