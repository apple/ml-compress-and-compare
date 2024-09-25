"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

Compression performance metrics.
"""

from sklearn.metrics import top_k_accuracy_score
import subprocess
import os
import numpy as np
import torch
from image_compression_experiments import compression_utils


class Metric:
    """Metric parent class for compression metrics."""

    def __init__(self, name, min=None, max=None, format=None, unit=None):
        """Initialze metric.

        Args:
            name (str): Metric name.
            min (number or None): Minimum value for the metric.
            max (number or None): Maximum value for the metric.
            format (str): String format for displaying the metric.
            unit (str): Units used for the metric.
        """
        self.name = name
        self.min = min
        self.max = max
        self.format = format
        self.unit = unit

    def representation(self):
        """Return the metric representation used for Compress and Compare."""
        output = {
            "name": self.name,
            "min": self.min,
            "max": self.max,
            "primary": True,
        }
        if self.format is not None:
            output["format"] = self.format
        if self.unit is not None:
            output["unit"] = self.unit
        return output


class Top1Accuracy(Metric):
    def __init__(self):
        super().__init__("Top-1 Accuracy", 0.0, 1.0, format=".2%")

    def compute(self, model, dataloader, labels, outputs, checkpoint, device):
        return top_k_accuracy_score(
            labels, outputs, k=1, labels=list(range(outputs.shape[1]))
        )


class Top5Accuracy(Metric):
    def __init__(self):
        super().__init__("Top-5 Accuracy", 0.0, 1.0, format=".2%")

    def compute(self, model, dataloader, labels, outputs, checkpoint, device):
        return top_k_accuracy_score(
            labels, outputs, k=5, labels=list(range(outputs.shape[1]))
        )


class Latency(Metric):
    def __init__(self):
        super().__init__("Latency", 0.0, None, unit="ms")

    def compute(self, model, dataloader, labels, outputs, checkpoint, device):
        batch = next(iter(dataloader))[0]
        return self._time(model, batch, device, 10)

    def _time(self, model, input, device, repetitions):
        model = model.to(device)
        if isinstance(model, torch.nn.DataParallel):
            model = model.module.to(device)
        input = input.to(device)

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        timings = np.zeros((repetitions, 1))

        # GPU warm up
        for _ in range(10):
            _ = model(input)

        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

        mean_syn = np.sum(timings) / repetitions
        return mean_syn


class DiskSize(Metric):
    def __init__(self):
        super().__init__("Model Size", 0.0, None, format=".2~s")

    def _covert_bytes(bytes):
        threshold = 1000
        if bytes < threshold:
            return f"{bytes} B"

        units = ["kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
        u = -1
        r = 10
        while bytes >= threshold and u < len(units) - 1:
            bytes /= threshold
            u += 1
        if bytes >= threshold:
            raise ValueError(f"File sizes larger than {units[-1]} not supported")

        return f"{bytes} {units[u]}"

    def compute(self, model, dataloader, labels, outputs, checkpoint, device):
        torch.save(model.state_dict(), "/tmp/model.h5")
        subprocess.run(["gzip", "-qf", "/tmp/model.h5"])
        bytes = subprocess.run(
            ["du", "-b", "/tmp/model.h5.gz"], capture_output=True, text=True
        ).stdout.split("\t")[0]
        if os.path.isfile("/tmp/model.h5.gz"):
            os.remove("/tmp/model.h5.gz")
        if os.path.isfile("/tmp/model.h5"):
            os.remove("/tmp/model.h5")
        return int(bytes)


class Sparsity(Metric):
    def __init__(self):
        super().__init__("Sparsity", 0.0, 1.0, format=".2%")

    def compute(self, model, dataloader, labels, outputs, checkpoint, device):
        try:
            return compression_utils.compute_model_sparsity(model)
        except:  # quantized models can't compute sparsity
            return None


class Epochs(Metric):
    def __init__(self):
        super().__init__("Epochs", 0.0, None)

    def compute(self, model, dataloader, labels, outputs, checkpoint, device):
        return checkpoint["epoch"]


def get_metric_fns():
    """Returns list of all metric classes."""
    return [Top1Accuracy(), Top5Accuracy(), Latency(), DiskSize(), Sparsity(), Epochs()]
