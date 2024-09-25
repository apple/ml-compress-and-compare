"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import subprocess
import os
import numpy as np
import torch
import time


def compute_latency(model, input, run_fn=None, device="cpu", repetitions=10):
    """
    Computes the latency of running a single batch on the given device
    by running it `repetitions` times. If run_fn is provided, it should be
    a function that takes a model and a batch and returns a function that
    takes no arguments and runs the model forward.
    """

    model = model.to(device)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module.to(device)
    input = input.to(device)

    runner = run_fn(model, input) if run_fn is not None else model

    print("inputs transferred")

    if torch.cuda.is_available():
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
    timings = np.zeros((repetitions, 1))

    with torch.no_grad():
        # GPU-WARM-UP
        print("Warming up")
        for rep in range(10):
            _ = runner()

        print("Measuring")
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                print(rep)
                start_time = time.time()
                if torch.cuda.is_available():
                    starter.record()
                    _ = runner()
                    ender.record()
                    # WAIT FOR GPU SYNC
                    torch.cuda.synchronize()
                    curr_time = starter.elapsed_time(ender)
                else:
                    _ = runner()
                    curr_time = time.time() - start_time
                timings[rep] = curr_time

    print("Done measuring")
    mean_syn = np.sum(timings) / repetitions
    model = model.to("cpu")
    return mean_syn


def compute_model_size(model):
    torch.save(model.state_dict(), "/tmp/model.h5")
    subprocess.run(["gzip", "-qf", "/tmp/model.h5"])
    bytes = subprocess.run(
        ["du", "-B 512", "/tmp/model.h5.gz"], capture_output=True, text=True
    ).stdout
    bytes = bytes.split("\t")[0]
    if os.path.isfile("/tmp/model.h5.gz"):
        os.remove("/tmp/model.h5.gz")
    if os.path.isfile("/tmp/model.h5"):
        os.remove("/tmp/model.h5")
    return int(bytes) * 512
