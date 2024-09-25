"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

CelebA image classification analysis for Compress and Compare.

To run with all models and outputs loaded, the celeba_blonde_classification
directory should contain a models/ directory with PyTorch checkpoints for each
model in celeba_models.json, and a dataset/ directory containing the
test_attributes.npy file.
"""

import argparse
import sys
import os
import h5py
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
    celeba,
    compression,
    compression_utils,
    models,
    utils,
)

torch.backends.quantized.engine = "qnnpack"

DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--port", type=int, help="Server port number", default=5001
)
args = parser.parse_args()

monitor = LocalModelMonitor(
    os.path.dirname(__file__),
    metadata_filename="celeba_models.json",
    outputs_type="numpy",
)


class CelebAModelInspector(PyTorchModelInspector):
    def get_model(self, model_id, for_inference=False):
        self.device = "cpu"
        if for_inference and torch.cuda.is_available():
            self.device = "cuda"

        checkpoint = monitor.load_model_checkpoint(model_id, map_location=self.device)
        model = models.load_model(name="resnet18", pretrained=False)[0]
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except RuntimeError:  # pruned models must be loaded into to pruned models
            model = compression.randompruning(model, 0.0, copy_model=False)
            model.load_state_dict(checkpoint["state_dict"])

        return model

    def get_eval_dataloader(self):
        if not hasattr(self, "test_dataloader"):
            _, self.test_dataloader = celeba.load_split(
                celeba.test_transform(), 256, "test", "Blond_Hair", False, DATA_PATH
            )
        return self.test_dataloader

    def iter_modules(self, model):
        module_names, layers_to_prune = compression_utils.get_prunable_parameters(model)
        return zip(module_names, layers_to_prune)

    def predictions(self, model_id):
        logits = monitor.get_model_outputs(model_id)
        outputs = torch.from_numpy(logits)
        attributes = np.load(os.path.join(DATA_PATH, "test_attributes.npy"))
        label_name = "Blond_Hair"
        label_attribute_index = celeba.CELEBA_ATTR.index(label_name)
        labels = attributes[:, label_attribute_index]

        sensitive_attributes_names = [label_name, "Young", "Male"]
        sensitive_attributes_names = ["Young", "Male"]
        sensitive_attribute_inds = [
            celeba.CELEBA_ATTR.index(a) for a in sensitive_attributes_names
        ]
        sensitive_attributes = attributes[:, sensitive_attribute_inds]

        predictions = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        probs = F.softmax(outputs, 1).detach().cpu().numpy()
        true_probs = probs[np.arange(0, len(probs)), labels]

        scores = np.array(labels == predictions).astype(np.uint8)
        tensor_results = {
            "scores": scores,
            "errors": np.logical_not(scores).astype(np.uint8),
            "logits": logits.astype(np.float16),
            "probs": probs.astype(np.float16),
            "true_probs": np.array(true_probs).astype(np.float16),
        }

        tested_attribute_classes = [f"Not {label_name}", label_name]
        sensitive_attribute_classes = [
            [f"Not {name}", name] for name in sensitive_attributes_names
        ]

        prediction_outputs = []
        for i in range(len(outputs)):
            prediction_outputs.append(
                Prediction(
                    i,
                    tested_attribute_classes[labels[i]],
                    tested_attribute_classes[predictions[i]],
                    [
                        sensitive_attribute_classes[attr_i][attr_value]
                        for attr_i, attr_value in enumerate(sensitive_attributes[i])
                    ],
                ).to_dict()
            )

        tensor_filepath = os.path.join(self.output_path, "tensors.h5")
        tensor_file = h5py.File(tensor_filepath, "w")

        for tensor_name, tensor_data in tensor_results.items():
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

    def comparison_metrics(self):
        return {
            "Correctness": ("difference", "scores", {"relative": True}),
            "Error Rate": ("difference", "errors", {"relative": True}),
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
                    celeba.visualization_transform(image=True),
                )
                for id in ids
            },
        }


if __name__ == "__main__":
    start_flask_server(
        monitor,
        LocalRunner(
            CelebAModelInspector,
            os.path.join(os.path.dirname(__file__), "task_outputs"),
            max_workers=1,
        ),
        port=args.port,
        debug=False,
    )
