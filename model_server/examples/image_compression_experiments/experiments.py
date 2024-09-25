"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

Experiment code for interconnected compression experiments.
"""

import os
import random
import torch
import numpy as np
from sklearn.metrics import accuracy_score

import celeba
import cifar10
import compression
import compression_utils
import imagenet
import training


def run_compression_experiment(
    dataset_name,
    model_fn,
    output_dir,
    checkpoint_name=None,
    compression_fns=[],
    compression_amounts=[],
    train_num_epochs=0,
    retrain_num_epochs=[],
    recalibrate=False,
    device="cuda",
    model_fn_kwargs={},
    verbose=True,
    iterative_compression=False,
    allow_weight_updates=False,
):
    """
    Runs compression experiment performing the following steps as specified:
        1. Initializes the model from model_fn and model_fn_kwargs. Initializes
           an optimizer and scheduler based on ImageNet training procedures.
        2. If a checkpoint path is provided it loads a model, optimizer, and
           scheduler from the checkpoint.
        3. If train, trained the model for num_epochs.
        4. For each compression_fn:
            4a. It applies compression by the specified compression_amount.
            4b. If recalibration it recalibrates the batchnorm layers using the
                training data.
            4c. If retraining, it retraines the model for retrain_num_epochs.
        5. Saves the model checkpoint.

    Args:
        dataset_name (str): One of 'celeba', 'imagenet', or 'cifar10'. Selects
            which dataset to use.
        model_fn (function): Function that creates a model. If dataset_name
            is not 'cifar10' it should also create a criterion, optimizer,
            scheduler, batch_size, and starting_epoch.
        output_dir (str): Directory to save compressed models.
        checkpoint_name (str): Name of checkpoint model to load otherwise None.
        compression_fns (list of function): List of compression functions to
            apply. Functions are applied in order. If only one compression
            function is passed it will be ran len(compression_amounts) times.
        compression_amounts (list of numbers): List of compression amounts. Can
            be floating point percentages [0, 1] or integer number of
            parameters.
        train_num_epochs (int): Number of epochs to train for.
        retrain_num_epochs (list of int): Number of epochs to retrain for after
            each compression function is applied.
        recalibrate (bool): Whether to recalibrate the batch normalization
            layers after compression.
        device (str): Device to run inference on. 'cuda' or 'cpu'. Quantization
            must be run on cpu.
        model_fn_kwargs (dict): Keyword arguments to model_fn.
        verbose (bool): Whether to print info during compression experiment.
        iterative_compression (bool): If True, only saves the final compressed
            model and does not save intermediate compression steps.
        allow_weight_updates (bool): If True, allows previously pruned weights
            to update in the next stage of compression and retraining.

    Returns: the final compressed model. Saves model checkpoints to disk during
        compression.
    """
    fn_name = globals()[f"run_{dataset_name}_experiment"]
    model = fn_name(
        model_fn,
        output_dir,
        checkpoint_name,
        compression_fns,
        compression_amounts,
        train_num_epochs,
        retrain_num_epochs,
        recalibrate,
        device,
        model_fn_kwargs,
        verbose,
        iterative_compression,
        allow_weight_updates,
    )
    return model


def run_celeba_experiment(
    model_fn,
    output_dir,
    checkpoint_name,
    compression_fns,
    compression_amounts,
    train_num_epochs,
    retrain_num_epochs,
    recalibrate,
    device,
    model_fn_kwargs,
    verbose,
    iterative_compression,
    allow_weight_updates,
):
    """Runs CelebA compression experiment."""
    print("Running CelebA compression experiment")

    # load ImageNet dataset
    model, criterion, optimizer, scheduler, batch_size, starting_epoch = model_fn(
        **model_fn_kwargs
    )
    model = torch.nn.DataParallel(model).to(device)

    model_name = model_fn_kwargs["name"]

    test_dataloader, val_dataloader, train_dataloader = celeba.load_celeba()
    val_dataset = val_dataloader.dataset
    holdout_dataset = torch.utils.data.Subset(
        val_dataset, random.sample(range(len(val_dataset)), batch_size)
    )
    holdout_dataloader = torch.utils.data.DataLoader(
        holdout_dataset, batch_size=batch_size, shuffle=False
    )

    model = _run_experiment(
        test_dataloader,
        holdout_dataloader,
        train_dataloader,
        model,
        criterion,
        optimizer,
        scheduler,
        output_dir,
        model_name,
        checkpoint_name,
        compression_fns,
        compression_amounts,
        train_num_epochs,
        retrain_num_epochs,
        recalibrate,
        device,
        verbose,
        starting_epoch,
        iterative_compression,
        allow_weight_updates,
    )
    return model


def run_imagenet_experiment(
    model_fn,
    output_dir,
    checkpoint_name,
    compression_fns,
    compression_amounts,
    train_num_epochs,
    retrain_num_epochs,
    recalibrate,
    device,
    model_fn_kwargs,
    verbose,
    iterative_compression,
    allow_weight_updates,
):
    """Runs ImageNet compression experiment."""
    print("Running ImageNet compression experiment")

    # load ImageNet dataset
    model, criterion, optimizer, scheduler, batch_size, starting_epoch = model_fn(
        **model_fn_kwargs
    )
    model = torch.nn.DataParallel(model).to(device)

    model_name = model_fn_kwargs["name"]
    if "pretrained" in model_fn_kwargs and model_fn_kwargs["pretrained"]:
        model_name += "trained"

    test_dataloader, train_dataloader = imagenet.load_imagenet()

    # hold out one batch to run gradient compression on
    test_dataset = test_dataloader.dataset
    holdout_dataset = torch.utils.data.Subset(
        test_dataset, random.sample(range(len(test_dataset)), batch_size)
    )
    holdout_dataloader = torch.utils.data.DataLoader(
        holdout_dataset, batch_size=batch_size, shuffle=False
    )

    # set schedule to final epoch to simulate pretraining (if any)
    for _ in range(starting_epoch):
        for _ in range(len(train_dataloader)):
            optimizer.step()
        scheduler.step()

    model = _run_experiment(
        test_dataloader,
        holdout_dataloader,
        train_dataloader,
        model,
        criterion,
        optimizer,
        scheduler,
        output_dir,
        model_name,
        checkpoint_name,
        compression_fns,
        compression_amounts,
        train_num_epochs,
        retrain_num_epochs,
        recalibrate,
        device,
        verbose,
        starting_epoch,
        iterative_compression,
        allow_weight_updates,
    )
    return model


def run_cifar10_experiment(
    model_fn,
    output_dir,
    checkpoint_name,
    compression_fns,
    compression_amounts,
    train_num_epochs,
    retrain_num_epochs,
    recalibrate,
    device,
    model_fn_kwargs,
    verbose,
    iterative_compression,
    allow_weight_updates,
):
    """Runs CIFAR10 compression experiment."""
    print("Running CIFAR-10 compression experiment")

    # load CIFAR dataset
    batch_size = 128
    test_dataloader, train_dataloader = cifar10.load_cifar10(batch_size=batch_size)

    # hold out one batch to run gradient compression on
    test_dataset = test_dataloader.dataset
    holdout_dataset = torch.utils.data.Subset(
        test_dataset, random.sample(range(len(test_dataset)), batch_size)
    )
    holdout_dataloader = torch.utils.data.DataLoader(
        holdout_dataset, batch_size=batch_size, shuffle=False
    )

    # set up CIFAR training procedure
    model = model_fn(**model_fn_kwargs).to(device)
    model_name = model_fn.__name__
    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer_kwargs = {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4}
    optimizer = torch.optim.SGD(model.parameters(), **optimizer_kwargs)

    scheduler_kwargs = {"gamma": 0.2, "milestones": [60, 120, 160]}
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_kwargs)

    checkpoint_path = _model_path(output_dir, checkpoint_name)
    starting_epoch = 0

    model = _run_experiment(
        test_dataloader,
        holdout_dataloader,
        train_dataloader,
        model,
        criterion,
        optimizer,
        scheduler,
        output_dir,
        model_name,
        checkpoint_name,
        compression_fns,
        compression_amounts,
        train_num_epochs,
        retrain_num_epochs,
        recalibrate,
        device,
        verbose,
        starting_epoch,
        iterative_compression,
        allow_weight_updates,
    )
    return model


def _run_experiment(
    test_dataloader,
    holdout_dataloader,
    train_dataloader,
    model,
    criterion,
    optimizer,
    scheduler,
    output_dir,
    model_name,
    checkpoint_name=None,
    compression_fns=[],
    compression_amounts=[],
    train_num_epochs=0,
    retrain_num_epochs=[],
    recalibrate=False,
    device="cuda",
    verbose=True,
    starting_epoch=0,
    iterative_compression=False,
    allow_weight_updates=False,
):
    # load new model and training procedures
    print("Initializing")
    model = model.to(device)
    epoch = starting_epoch
    compression_techniques = [model_name.replace("_", "")]
    compression_steps = []
    parent_name = None
    name = _get_name(compression_techniques, compression_steps)
    if not _model_exists(output_dir, name):
        _save_checkpoint(model, epoch, {}, parent_name, output_dir, name)
    parent_name = name

    # load model checkpoint
    checkpoint_path = os.path.join(output_dir, f"{checkpoint_name}.pth")
    if checkpoint_name is not None and os.path.isfile(checkpoint_path):
        print(f"Loading from checkpoint: {checkpoint_path}")
        model, epoch = _load_model(checkpoint_path, model)
        base_model = os.path.splitext(os.path.basename(checkpoint_path))[0].split("_")
        compression_techniques = [base_model[0]]
        if len(base_model) > 1:
            compression_steps = [base_model[1]]
        parent_name = _get_name(compression_techniques, compression_steps)

    # train the model
    if train_num_epochs > 0:
        print(f"Training for {train_num_epochs} epochs")
        compression_techniques[-1] += "trained"
        name = _get_name(compression_techniques, compression_steps)
        training.train(
            model,
            train_dataloader,
            test_dataloader,
            optimizer,
            scheduler,
            criterion,
            num_epochs=train_num_epochs,
            starting_epoch=epoch,
            device=device,
        )
        epoch += train_num_epochs
        operation = {"name": "train", "parameters": {"epochs": train_num_epochs}}
        _save_checkpoint(model, epoch, operation, parent_name, output_dir, name)
        parent_name = name

    if verbose:
        print(
            f"Original model accuracy = {training.test(model, test_dataloader)[0]:.4%}"
        )

    # compress the model
    if not isinstance(compression_fns, list):
        compression_fns = [compression_fns for _ in range(len(compression_amounts))]

    for iteration, amount in enumerate(compression_amounts):
        if (
            isinstance(amount, int) and amount > 1
        ):  # passing number of parameters opposed to percentage
            prunable_modules = compression_utils.get_prunable_parameters(model)[1]
            parameters = sum([float(m[0].weight.nelement()) for m in prunable_modules])
            zero_parameters = sum(
                [float(torch.sum(m[0].weight == 0)) for m in prunable_modules]
            )
            remaining_parameters = parameters - zero_parameters
            amount = amount / remaining_parameters
        print(f"Compressing: {compression_fns[iteration]} {amount}")

        # initial compression
        compression_fn = compression_fns[iteration]
        if not iterative_compression or iteration == 0:
            total_amount = amount
            if iterative_compression:
                num_parameters = sum(
                    [
                        float(m[0].weight.nelement())
                        for m in compression_utils.get_prunable_parameters(model)[1]
                    ]
                )
                total_amount = np.sum(compression_amounts) / num_parameters
            if isinstance(total_amount, float):
                total_amount *= 100
            if isinstance(total_amount, float) or isinstance(total_amount, int):
                total_amount = int(total_amount)
            compression_techniques.append(f"{compression_fn.__name__}{total_amount}")
            print("Compression Technique", compression_techniques)
        compression_steps = ["compressed"]
        if compression_techniques[-1] == "quantizeint8":
            compression_name = f"{parent_name}_quantizeint8"
        else:
            compression_name = _get_name(compression_techniques, compression_steps)

        make_compression_permanent = False
        if allow_weight_updates and iteration < len(compression_amounts) - 1:
            make_compression_permanent = True
        model.to(device)
        model = compression_fn(
            model,
            amount,
            holdout_dataloader,
            copy_model=False,
            permanent=make_compression_permanent,
        ).to(device)

        operation = {
            "name": "quantize"
            if "quantize" in compression_fn.__name__
            else f"{compression_fn.__name__.split('pruning')[0]} prune",
            "parameters": {"amount": amount},
        }
        retrain = (
            len(retrain_num_epochs) > iteration and retrain_num_epochs[iteration] > 0
        )
        if not iterative_compression or (
            iteration == len(compression_amounts) - 1
            and not recalibrate
            and not retrain
        ):
            _save_checkpoint(
                model, epoch, operation, parent_name, output_dir, compression_name
            )
        parent_name = compression_name

        if verbose:
            print(
                f"Compressed model accuracy: {training.test(model, test_dataloader, device)[0]:.4%}"
            )
            try:
                print(
                    f"Compressed model sparsity: {compression_utils.compute_model_sparsity(model):%}"
                )
            except:
                pass

        # recalibration
        if recalibrate:
            print("Recalibrating")
            compression_steps = ["compressed", "recalibrated"]
            name = _get_name(compression_techniques, compression_steps)
            compression_utils.recalibrate(model, train_dataloader, device=device)
            operation = {"name": "calibrate", "parameters": {"data": "train batch"}}
            if not iterative_compression or iteration == len(compression_amounts) - 1:
                _save_checkpoint(model, epoch, operation, parent_name, output_dir, name)
            parent_name = name

            if verbose:
                print(
                    f"Recalibrated model accuracy: {training.test(model, test_dataloader, device)[0]:.4%}"
                )

        # retraining
        if retrain:
            print(f"Retraining for {retrain_num_epochs[iteration]} epochs")
            compression_steps = ["compressed", "retrained"]
            name = _get_name(compression_techniques, compression_steps)
            parent_name = compression_name
            if len(retrain_num_epochs) != len(compression_amounts):
                raise ValueError(
                    "Retraining iterations do not match compression iterations"
                )
            training.train(
                model,
                train_dataloader,
                test_dataloader,
                optimizer,
                scheduler,
                criterion,
                num_epochs=retrain_num_epochs[iteration],
                starting_epoch=epoch,
                device=device,
            )
            epoch += retrain_num_epochs[iteration]
            operation = {
                "name": "train",
                "parameters": {"epochs": retrain_num_epochs[iteration]},
            }
            if not iterative_compression or iteration == len(compression_amounts) - 1:
                _save_checkpoint(model, epoch, operation, parent_name, output_dir, name)
            parent_name = name

            if verbose:
                print(
                    f"Retrained model accuracy: {training.test(model, test_dataloader, device)[0]:.4%}"
                )

    return model


def _save_checkpoint(model, epoch, operation, parent_name, output_dir, name):
    """Saves a model checkpoint with epoch, state_dict, optimizer, and scheduler."""
    filepath = _model_path(output_dir, name)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    checkpoint = {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "operation": operation,
        "parent": parent_name,
    }
    torch.save(checkpoint, os.path.join(output_dir, f"{name}.pth"))


def _get_name(compression_techniques, compression_steps):
    """
    Creates name for the experiment in the format:
        compression-techniques_compression-steps
    """
    name = "-".join(compression_techniques)
    if len(compression_steps) > 0:
        name = f"{name}_{'-'.join(compression_steps)}"
    return name


def _model_path(output_dir, name):
    """Gets full path to model."""
    return os.path.join(os.path.join(output_dir, f"{name}.pth"))


def _model_exists(output_dir, name):
    """Checks if model exists."""
    return os.path.isfile(_model_path(output_dir, name))


def _load_state_dict(model, state_dict):
    """Loads state_dict into the model. Removes DataParallel (if any) from the
    model and state_dict. Returns the loaded model."""
    model_data_parallel = isinstance(model, torch.nn.DataParallel)
    state_dict_data_parallel = list(state_dict.keys())[0].startswith("module")
    if model_data_parallel and not state_dict_data_parallel:
        model.module.load_state_dict(state_dict)
    elif not model_data_parallel and state_dict_data_parallel:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
        model = model.module
    else:
        model.load_state_dict(state_dict)
    return model


def _load_model(filepath, model):
    """Loads checkpoint at filepath into the model"""
    checkpoint = torch.load(filepath)
    print(f"Loading {filepath}")
    try:  # load uncompressed model
        model = _load_state_dict(model, checkpoint["state_dict"])
    except:  # covert to a pruned model (set sparsity to 0)
        model = compression.randompruning(model, 0.0)
        model = _load_state_dict(model, checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    return model, epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--dataset_name", type=str, options=["celeba", "imagenet", "cifar10"]
    )
    parser.add_argument("-m", "--model_fn", type=str, default="models.load_model")
    parser.add_argument("-o", "--output_dir", type=str, default="./results/")
    parser.add_argument("-c", "--checkpoint_name", type=str)
    parser.add_argument(
        "-f", "--compression_fns", type=lambda s: [str(item) for item in s.split(",")]
    )
    parser.add_argument(
        "-a",
        "--compression_amounts",
        type=lambda s: [float(item) for item in s.split(",")],
    )
    parser.add_argument("-e", "--train_num_epochs", type=int, default=0)
    parser.add_argument(
        "-r",
        "--retrain_num_epochs",
        type=lambda s: [int(item) for item in s.split(",")],
    )
    parser.add_argument("-c", "--recalibrate", action="store_true")
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument(
        "-k",
        "--model_fn_kwargs",
        type=json.loads,
        default="{'name': 'resnet18', 'pretrained': False}",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-i", "--iterative_compression", action="store_true")
    parser.add_argument("-w", "--allow_weight_updates", action="store_true")
    args = parser.parse_args()

    run_compression_experiment(
        dataset_name=args.dataset_name,
        model_fn=eval(args.model_fn),
        output_dir=args.output_dir,
        checkpoint_name=args.checkpoint_name,
        compression_fns=[eval(fn) for fn in args.compression_fns],
        compression_amounts=args.compression_amounts,
        train_num_epochs=args.train_num_epochs,
        retrain_num_epochs=args.retrain_num_epochs,
        recalibrate=args.recalibrate,
        device=args.device,
        model_fn_kwargs=args.model_fn_kwargs,
        verbose=args.verbose,
        iterative_compression=args.iterative_compression,
        allow_weight_updates=args.allow_weight_updates,
    )
