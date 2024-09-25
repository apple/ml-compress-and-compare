"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

ImageNet image classification analysis for Compress and Compare.

To run with all models and outputs loaded, the celeba_blonde_classification
directory should contain a models/ directory with PyTorch checkpoints for each
model in the imagenet_models.json file.
"""

import sys
import os
import argparse
import torch
import h5py
import numpy as np
import torch.nn.functional as F

from interactive_compression.server import start_flask_server
from interactive_compression.runners import LocalRunner
from interactive_compression.inspectors.pytorch import PyTorchModelInspector
from interactive_compression.inspectors.base import Prediction
from interactive_compression.monitors import LocalModelMonitor

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from image_compression_experiments import (
    imagenet,
    compression,
    compression_utils,
    models,
    utils,
)

torch.backends.quantized.engine = "qnnpack"

MODEL_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--models",
    type=str,
    help="Set of models to show",
    choices=["tutorial", "part1", "part2", "all"],
    default="all",
)
parser.add_argument(
    "-p", "--port", type=int, help="Server port number", default=5001
)
args = parser.parse_args()

model_filename = f"imagenet_models_userstudy_{args.models}.json"
if args.models in ["all", "part2"]:
    model_filename = "imagenet_models.json"
monitor = LocalModelMonitor(
    MODEL_PATH, metadata_filename=model_filename, outputs_type="numpy"
)


class ImageNetModelInspector(PyTorchModelInspector):
    def get_model(self, model_id, for_inference=False):
        self.device = "cpu"
        if for_inference and torch.cuda.is_available():
            self.device = "cuda"

        checkpoint = monitor.load_model_checkpoint(model_id, map_location=self.device)
        architecture = model_id.split("trained")[0]
        if architecture == "mobilenetv2":
            architecture = "mobilenet_v2"
        model = models.load_model(name=architecture, pretrained=False)[0]

        if "quantize" in model_id:
            model = compression.quantize(
                model,
                data=self.test_dataloader,
                copy_model=False,
                backend="qnnpack",
            )
            model.load_state_dict(checkpoint["state_dict"])
        else:
            try:
                model.load_state_dict(checkpoint["state_dict"])
            except RuntimeError:
                model = compression.randompruning(model, 0.0, copy_model=False)
                model.load_state_dict(checkpoint["state_dict"])
        return model

    def get_eval_dataloader(self):
        if not hasattr(self, "test_dataloader"):
            _, self.test_dataloader = imagenet.load_split(
                imagenet.test_transform(), 256, DATA_PATH, train=False
            )
        return self.test_dataloader

    def iter_modules(self, model):
        module_names, layers_to_prune = compression_utils.get_prunable_parameters(model)
        return zip(module_names, layers_to_prune)

    def predictions(self, model_id):
        logits = monitor.get_model_outputs(model_id)
        outputs = torch.from_numpy(logits)
        labels = np.load(os.path.join(DATA_PATH, "imagenet_labels.npy"))

        predictions = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        probs = F.softmax(outputs, 1).detach().cpu().numpy()
        true_probs = probs[np.arange(0, len(probs)), labels]

        tensor_results = {
            "scores": np.array(labels == predictions).astype(np.uint8),
            "logits": logits.astype(np.float16),
            "probs": probs.astype(np.float16),
            "true_probs": np.array(true_probs).astype(np.float16),
        }

        def get_imagenet_classname(class_id: int) -> str:
            # TODO: convert to human-readable class names.
            return str(class_id)

        prediction_outputs = []
        for i in range(len(outputs)):
            prediction_outputs.append(
                Prediction(
                    i,
                    get_imagenet_classname(labels[i]),
                    get_imagenet_classname(predictions[i]),
                    [get_imagenet_classname(labels[i])],
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
                    "comparison": comp_type,
                    "vectors": {
                        "format": "hdf5",
                        "path": tensor_filepath,
                        "dataset_name": tensor_name,
                    },
                }
                for comparison_name, (comp_type, tensor_name) in comparisons.items()
            },
        }

    def comparison_metrics(self):
        return {
            "Correctness": ("difference", "scores"),
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
                    imagenet.visualization_transform(image=True),
                )
                for id in ids
            },
        }


if __name__ == "__main__":
    start_flask_server(
        monitor,
        LocalRunner(
            ImageNetModelInspector,
            os.path.join(os.path.dirname(__file__), "task_outputs"),
            max_workers=1,
        ),
        port=args.port,
        debug=False,
    )
