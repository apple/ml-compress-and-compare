"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import gc
import os
import torch
import pickle
import numpy as np
import h5py


def unpack_tensors(output, base_name):
    if isinstance(output, dict):
        for k, v in output.items():
            yield from unpack_tensors(v, f"{base_name}.{k}")
    elif isinstance(output, (list, tuple)):
        for i, v in enumerate(output):
            yield from unpack_tensors(
                v, f"{base_name}.{i}" if len(output) > 1 else base_name
            )
    elif isinstance(output, np.ndarray):
        yield base_name, output
    elif output is None:
        return
    else:
        raise ValueError(f"Unexpected output value: {type(output)}")


class TensorOutputTracker:
    quantization_function = None
    output_prefix = ""
    output_hooks = {}
    clear_hook = None
    tensor_converters = {}
    tensor_output_dir = ""

    @staticmethod
    def set_output_dir(output_dir):
        TensorOutputTracker.tensor_output_dir = output_dir

    @staticmethod
    def register_output_converter(output_type, fn):
        TensorOutputTracker.tensor_converters[output_type] = fn

    @staticmethod
    def get_path_for_module(module_name, output_prefix=None):
        prefix = output_prefix or TensorOutputTracker.output_prefix
        path = os.path.join(
            TensorOutputTracker.tensor_output_dir,
            f"output_{prefix}_{module_name}.h5",
        )
        return path

    @staticmethod
    def convert_tensors(output, name):
        if isinstance(output, dict):
            return {
                k: TensorOutputTracker.convert_tensors(v, name)
                for k, v in output.items()
            }
        elif isinstance(output, list):
            return [TensorOutputTracker.convert_tensors(v, name) for v in output]
        elif isinstance(output, tuple):
            return tuple(TensorOutputTracker.convert_tensors(v, name) for v in output)
        elif output is None or isinstance(output, (bool, int, float)):
            return None
        elif isinstance(output, torch.Tensor):
            result = output.detach().cpu().numpy()
            if TensorOutputTracker.quantization_function is not None:
                result = TensorOutputTracker.quantization_function(result)
            return result
        elif type(output) in TensorOutputTracker.tensor_converters:
            return TensorOutputTracker.convert_tensors(
                TensorOutputTracker.tensor_converters[type(output)](output), name
            )
        raise ValueError(f"Unknown output type for module '{name}': {type(output)}")

    @staticmethod
    def create_output_hook(name, module):
        def hook(module, input, output):
            print("forward", name)
            result = TensorOutputTracker.convert_tensors(output, name)
            path = TensorOutputTracker.get_path_for_module(name)
            if os.path.exists(path):
                h5file = h5py.File(path, "a")
                append = True
            else:
                h5file = h5py.File(path, "w")
                append = False

            for tensor_name, tensor in unpack_tensors(result, name):
                if append:
                    assert (
                        tensor_name in h5file
                    ), f"Expected tensor {tensor_name} in existing activations output file"
                    h5file[tensor_name].resize(
                        (h5file[tensor_name].shape[0] + tensor.shape[0]), axis=0
                    )
                    h5file[tensor_name][-tensor.shape[0] :] = tensor
                else:
                    h5file.create_dataset(tensor_name, data=tensor, maxshape=(None, *tensor.shape[1:]))
            h5file.close()

        remover = module.register_forward_hook(hook)
        TensorOutputTracker.output_hooks[name] = remover
        return remover

    @staticmethod
    def remove_output_hooks():
        for remover in TensorOutputTracker.output_hooks.values():
            remover.remove()
        TensorOutputTracker.output_hooks = {}

    @staticmethod
    def get_module_outputs(name, output_prefix=None):
        path = TensorOutputTracker.get_path_for_module(name, output_prefix)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        results = {}
        file = h5py.File(path, "r")
        for dataset in file.keys():
            results[dataset] = file[dataset][:]
        file.close()
        return results

    @staticmethod
    def clear_all():
        for path in os.listdir(TensorOutputTracker.tensor_output_dir):
            os.remove(os.path.join(TensorOutputTracker.tensor_output_dir, path))

    @staticmethod
    def clear():
        gc.collect()

    @staticmethod
    def create_clear_hook(module, fn=None):
        """Creates a forward hook on the given module that runs a given function, then clears the cache."""

        def hook(m, input, output):
            if fn is not None:
                fn()
            TensorOutputTracker.clear()

        remover = module.register_forward_hook(hook)
        TensorOutputTracker.clear_hook = remover

    @staticmethod
    def remove_clear_hook(module):
        TensorOutputTracker.clear_hook.remove()
        TensorOutputTracker.clear_hook = None


# We could think about having larger bin widths at larger values, since there is less data density there
def make_linear_bin_quantizer(bins):
    bins = np.array(bins)
    steps = bins[1:] - bins[:-1]
    assert all(abs(steps[i] - steps[0]) <= 1e-5 for i in range(len(steps)))
    minimum = bins.min()
    maximum = bins.max()
    step = bins[1] - bins[0]
    count = len(bins)

    def quantizer(tensor):
        num_below_minimum = (tensor < minimum).sum()
        if num_below_minimum > 1:
            print(
                f"{num_below_minimum} below minimum value: ranging {tensor[tensor < minimum].min()} - {tensor[tensor < minimum].max()}"
            )
        return (np.clip(np.floor((tensor - minimum) / step), -1, count - 1) + 1).astype(
            np.uint8
        )

    return quantizer


def make_simple_quantizer(dtype):
    def quantizer(tensor):
        return tensor.astype(dtype)

    return quantizer
