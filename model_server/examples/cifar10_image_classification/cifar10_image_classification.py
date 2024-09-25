"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

Cifar10 image classification analysis for Compress and Compare.

To run with all loaded models and outputs, the cifar10_image_classification directory
should contain a models/ directory with PyTorch checkpoints for each model defined in
cifar10_image_classification_models.json.
"""

import argparse
import json
import sys
import os
import torch
import numpy as np
import torch.nn.functional as F

from interactive_compression.server import start_flask_server
from interactive_compression.runners import LocalRunner
from interactive_compression.inspectors.pytorch import PyTorchModelInspector
from interactive_compression.inspectors.base import Prediction
from interactive_compression.monitors import LocalModelMonitor

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from image_compression_experiments import (
    cifar10,
    compression,
    compression_utils,
    utils,
    resnet,
)

torch.backends.quantized.engine = "qnnpack"

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--port", type=int, help="Server port number", default=5001
)
args = parser.parse_args()

model_filename = "cifar10_image_classification_models.json"
monitor = LocalModelMonitor(
    os.path.dirname(__file__), metadata_filename=model_filename
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset")


class CIFARModelInspector(PyTorchModelInspector):
    def get_model(self, model_id, for_inference=False):
        self.device = "cpu"
        if for_inference and torch.cuda.is_available():
            self.device = "cuda"

        checkpoint = monitor.load_model_checkpoint(model_id, map_location=self.device)

        if "quantize" in model_id:
            if not hasattr(self, "test_dataloader"):
                self.get_eval_dataloader()
            model = resnet.resnet20(num_classes=10, quantized=True).to(self.device)
            model = compression.quantize(
                model,
                amount="int8",
                data=self.test_dataloader,
                copy_model=False,
                backend="qnnpack",
            )
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model = resnet.resnet20(num_classes=10).to(self.device)
            try:
                model.load_state_dict(checkpoint["state_dict"])
            except RuntimeError:
                model = compression.randompruning(model, 0.0, copy_model=False)
                model.load_state_dict(checkpoint["state_dict"])
        return model

    def get_eval_dataloader(self):
        if not hasattr(self, "test_dataloader"):
            _, self.test_dataloader = cifar10.load_split(
                cifar10.test_transform(), 128, DATA_PATH, train=False
            )
        return self.test_dataloader

    def iter_modules(self, model):
        module_names, layers_to_prune = compression_utils.get_prunable_parameters(model)
        return zip(module_names, layers_to_prune)

    def inference_batch(self, model, batch, batch_index):
        inputs, labels = batch
        outputs = model(inputs.to(self.device))
        predictions = torch.argmax(outputs, dim=1)

        prediction_outputs = []
        scores = []
        logits = []
        true_probs = []
        for i, prediction in enumerate(predictions):
            overall_idx = batch_index * self.test_dataloader.batch_size + i

            prediction_outputs.append(
                Prediction(
                    overall_idx,
                    cifar10.CIFAR_LABELS[labels[i].item()],
                    cifar10.CIFAR_LABELS[prediction.item()],
                    [cifar10.CIFAR_LABELS[labels[i].item()]],
                )
            )

            true_probs.append(float(F.softmax(outputs[i], 0)[labels[i]].item()))
            scores.append(labels[i] == torch.argmax(outputs[i]).item())
            logits.append(outputs[i])

        logits = torch.stack(logits, 0)
        return prediction_outputs, {
            "scores": np.array(scores).astype(np.uint8),
            "logits": logits.cpu().numpy().astype(np.float16),
            "probs": F.softmax(logits, 1).cpu().numpy().astype(np.float16),
            "true_probs": np.array(true_probs).astype(np.float16),
        }

    def comparison_metrics(self):
        return {
            "Correctness": (
                "difference",
                "scores",
                {"relative": True},
            ),
            "Confidence on True Label": ("difference", "true_probs"),
            "Logits MSE": ("mse", "logits"),
            "KL Divergence": ("kl_divergence", "probs"),
        }

    def instances(self, ids):
        test_dataloader = self.get_eval_dataloader()
        return {
            "type": "image",
            "instances": {
                id: utils.encode_image(
                    test_dataloader.dataset[int(id)][0],
                    cifar10.visualization_transform(image=True),
                )
                for id in ids
            },
        }


if __name__ == "__main__":
    start_flask_server(
        monitor,
        LocalRunner(
            CIFARModelInspector,
            os.path.join(os.path.dirname(__file__), "task_outputs"),
            max_workers=1,
        ),
        port=args.port,
        debug=False,
    )
