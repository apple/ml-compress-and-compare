"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

Compute compression performance metrics on models.
"""

import os
import random
import numpy as np
import torch
import argparse
import json
from tqdm import tqdm

from image_compression_experiments import (
    compression_utils,
    celeba,
    imagenet,
    cifar10,
    resnet,
    compression,
    compression_utils,
    compression_metrics,
)
import image_compression_experiments.models as vision_models


def _compute_metrics(model, dataloader, labels, outputs, checkpoint, device):
    metric_fns = compression_metrics.get_metric_fns()
    results = {}
    for metric in metric_fns:
        results[metric.name] = metric.compute(
            model, dataloader, labels, outputs, checkpoint, device
        )

    metrics = [metric_fn.representation() for metric_fn in metric_fns]
    return results, metrics


def _load_model(model, model_file, checkpoint, dataloader):
    if "quantize" in model_file:
        model = compression.quantize(
            model, amount=None, data=dataloader, copy_model=True, backend="qnnpack"
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model, "cpu"
    if "pruning" in model_file:  # pruned models must be converted to pruned models
        model = compression.randompruning(model, 0.0)
        model.load_state_dict(checkpoint["state_dict"])
        compression_utils.make_pruning_permanent(model)
        return model, "cuda"
    else:
        model.load_state_dict(checkpoint["state_dict"])
        return model, "cuda"


def analyze_performance(model_dir, outputs_dir, dataset="imagenet"):
    print("Computing model metrics")
    # load dataset
    if dataset.lower() == "imagenet":
        test_dataloader, train_dataloader = imagenet.load_imagenet(batch_size=256)
        test_dataset = test_dataloader.dataset
        holdout_dataset = torch.utils.data.Subset(
            test_dataset, random.sample(range(len(test_dataset)), batch_size)
        )
        holdout_dataloader = torch.utils.data.DataLoader(
            holdout_dataset, batch_size=batch_size, shuffle=False
        )
        print("HOLDOUT", len(holdout_dataset))
    elif dataset.lower() == "celeba":
        test_dataloader, val_dataloader, train_dataloader = celeba.load_celeba(
            batch_size=256
        )
        val_dataset = val_dataloader.dataset
        holdout_dataset = torch.utils.data.Subset(
            val_dataset, random.sample(range(len(val_dataset)), batch_size)
        )
        holdout_dataloader = torch.utils.data.DataLoader(
            holdout_dataset, batch_size=batch_size, shuffle=False
        )
    elif dataset.lower() == "cifar10":
        batch_size = 128
        test_dataloader, train_dataloader = cifar10.load_cifar10(batch_size=batch_size)
        test_dataset = test_dataloader.dataset
        holdout_dataset = torch.utils.data.Subset(
            test_dataset, random.sample(range(len(test_dataset)), batch_size)
        )
        holdout_dataloader = torch.utils.data.DataLoader(
            holdout_dataset, batch_size=batch_size, shuffle=False
        )
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    print(f"Loaded data {len(test_dataloader)}")

    labels = np.concatenate(
        [label.detach().cpu().numpy() for _, label in tqdm(test_dataloader)]
    )

    # quatnized models must be timed on cpu
    device = "cuda"
    for model_file in os.listdir(model_dir):
        if "quantize" in model_file:
            device = "cpu"
    print(f"Computing metrics on {device}")

    models = []
    for model_file in tqdm(os.listdir(model_dir)):
        if not os.path.isfile(
            os.path.join(model_dir, model_file)
        ) or not model_file.endswith("pth"):
            continue
        print(f"Processing {model_file}")
        model_object = {}

        # check if model is already processed
        model_name, ext = os.path.splitext(model_file)
        model_metrics_dir = os.path.join(outputs_dir, "model_metrics")
        if not os.path.isdir(model_metrics_dir):
            os.mkdir(model_metrics_dir)
        model_info_file = os.path.join(model_metrics_dir, f"{model_name}.json")
        if os.path.isfile(model_info_file):
            print(f"Loading {model_info_file}")
            with open(model_info_file, "r") as f:
                model_object = json.load(f)
            models.append(model_object)

        else:
            print(f"Computing {model_info_file}")
            # load model
            checkpoint = torch.load(os.path.join(model_dir, model_file))
            architecture = model_name.split("-")[0].split("train")[0]
            if architecture == "mobilenetv2":
                architecture = "mobilenet_v2"
            try:
                model = vision_models.load_model(architecture, pretrained=False)[0]
            except:
                model = eval(f"resnet.{architecture}")()
            model, device = _load_model(
                model, model_file, checkpoint, holdout_dataloader
            )
            model = model.to(device).eval()
            if device == "cuda":
                model = torch.nn.DataParallel(model)

            # compute outputs
            model_outputs_dir = os.path.join(outputs_dir, "model_outputs")
            if not os.path.isdir(model_outputs_dir):
                os.mkdir(model_outputs_dir)
            outputs_file = os.path.join(model_outputs_dir, f"{model_name}.npy")
            if not os.path.isfile(outputs_file):
                outputs = []
                for image, _ in test_dataloader:
                    output = model(image.to(device))
                    outputs.append(output.detach().cpu().numpy())
                outputs = np.vstack(outputs)
                np.save(outputs_file, outputs)
            outputs = np.load(outputs_file)
            print(f"Computed outputs {outputs.shape}")

            # compute metrics
            results, metrics = _compute_metrics(
                model, test_dataloader, labels, outputs, checkpoint, device
            )
            print(f"Computed {len(metrics)} metrics")

            # store model info
            model_object["id"] = model_name
            if "parent" in checkpoint:
                model_object["base"] = checkpoint["parent"]
            model_object["operation"] = checkpoint["operation"]
            model_object["tag"] = ""
            if len(model_object["operation"]) > 0:
                operation_str = model_object["operation"]["name"].title()
                operation_parameters_str = ", ".join(
                    [
                        f"{k}: {v}"
                        for k, v in model_object["operation"]["parameters"].items()
                    ]
                )
                model_object["tag"] = f"{operation_str} ({operation_parameters_str})"
            model_object["metrics"] = results
            with open(model_info_file, "w") as f:
                json.dump(model_object, f, indent=4)
            models.append(model_object)

    # get metrics summary
    metrics = [
        metric_fn.representation() for metric_fn in compression_metrics.get_metric_fns()
    ]

    # update sparsity for quantized models
    for model in models:
        if model["metrics"]["Sparsity"] is None:
            parent_sparsity = [
                m["metrics"]["Sparsity"] for m in models if m["id"] == model["base"]
            ]
            assert len(parent_sparsity) == 1
            model["metrics"]["Sparsity"] = parent_sparsity[0]

    # create operation summary
    operations = {}
    for model in models:
        if len(model["operation"]) > 0:
            operation_object = {
                "name": model["operation"]["name"],
                "parameters": [k for k in model["operation"]["parameters"]],
            }
            operations[operation_object["name"]] = operation_object
    operations = list(operations.values())

    # write out spec
    output = {"metrics": metrics, "operations": operations, "models": models}
    with open(os.path.join(outputs_dir, f"{dataset}_models.json"), "w") as f:
        json.dump(output, f, indent=4)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, default="./results/celeba_blonde_classification/"
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="./results/celeba_blonde_classification/outputs/",
    )
    parser.add_argument("--dataset", type=str, default="celeba")
    args = parser.parse_args()
    analyze_performance(args.model_dir, args.outputs_dir, dataset=args.dataset)
