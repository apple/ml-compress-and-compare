"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import torch
import gc
import base64
import numpy as np
import pyarrow as pa
from io import BytesIO
import gzip


def empty_device_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        from torch import mps

        mps.empty_cache()
    gc.collect()


def convert_arrow_batch_to_string(batch, gzipped=True, base64_encoded=True):
    sink = pa.BufferOutputStream()

    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)

    file_like = BytesIO(sink.getvalue().to_pybytes())

    if gzipped:
        new_file = BytesIO()
        with gzip.GzipFile(fileobj=new_file, mode="w") as gzipfile:
            gzipfile.write(file_like.getvalue())
        file_like = new_file

    if base64_encoded:
        encoded = base64.b64encode(file_like.getvalue()).decode("ascii")
    else:
        encoded = file_like.getvalue().decode("ascii")

    print("encoded arrow batch has length", len(encoded))
    return encoded


def standardize_json(o, round_digits=8):
    """
    Produces a JSON-compliant object by replacing numpy types with system types
    and rounding floats to save space.
    """
    if isinstance(o, (float, np.float16, np.float32, np.float64)):
        if np.isnan(o) or np.isinf(o):
            print(
                "WARNING: data to be JSON-serialized contains nans or infinity values. They will be converted to zeros to prevent crashing the client."
            )
            return 0
        return round(float(o), round_digits)
    if isinstance(o, (np.int64, np.int32, np.uint8, np.int8)):
        return int(o)
    if isinstance(o, np.ndarray):
        return standardize_json(o.tolist())
    if isinstance(o, dict):
        return {
            standardize_json(k, round_digits): standardize_json(v, round_digits)
            for k, v in o.items()
        }
    if isinstance(o, (list, tuple)):
        return [standardize_json(x, round_digits) for x in o]
    return o


image_encoding_warning = False


def standardize_image_format(image):
    """
    Converts the given image array to RGBA uint8 format.

    Args:
        image: An image matrix.
    """

    assert (
        len(image.shape) == 3
    ), "Images must be 3-dimensional arrays (include a color channel)"

    image = image.astype(np.uint8)
    result = np.zeros((*image.shape[:2], 4), dtype=np.uint8)
    if image.shape[2] == 1:
        for channel in range(3):
            result[:, :, channel] = image[:, :, 0]
        result[:, :, 3] = 255
    elif image.shape[2] == 3:
        result[:, :, :3] = image
        result[:, :, 3] = 255
    elif image.shape[2] == 4:
        result = image
    else:
        raise ValueError(
            "Incorrect image shape, expected 3rd dimension to be 1, 3 or 4; got {} instead".format(
                image.shape[2]
            )
        )

    if np.max(image[:, :, :3]) < 2 and not image_encoding_warning:
        print(
            "Warning: Your image pixel values are very small. This code expects integer pixel values from 0-255."
        )
        image_encoding_warning = True
    return result


def choose_histogram_bins(min_val, max_val, n_bins=None):
    """
    Automatically generates a nice set of histogram bins for the given
    data bounds.
    """
    if not np.isfinite(min_val): min_val = 0
    if not np.isfinite(max_val): max_val = 1
    if max_val == min_val: max_val = min_val + 1
    print(min_val, max_val)

    data_range = max_val - min_val
    bin_scale = np.floor(np.log10(data_range))
    if data_range / (10**bin_scale) < 2.5:
        bin_scale -= 1  # Make sure there aren't only 2-3 bins
    step = 10**bin_scale
    if n_bins is not None:
        # Adjust the step so that data_range / step is greater than n_bins
        step *= 0.1
        while data_range / step > n_bins:
            if data_range / step > n_bins * 5 - 1:
                step *= 5
            else:
                step *= 2
        # Start at the middle of the data range and go outward
        bin_start = min_val + data_range * 0.5 - (n_bins // 2) * step
        bin_end = min_val + data_range * 0.5 + (n_bins - n_bins // 2) * step
        assert (
            bin_start <= min_val and bin_end >= max_val
        ), f"New bins don't cover data range: {bin_start} - {bin_end} vs {min_val} - {max_val}"
        hist_bins = np.arange(
            np.floor(bin_start / step) * step,
            np.ceil(bin_end / step) * step,
            step,
        )
    else:
        upper_tol = (
            2
            if (np.ceil(max_val / (10**bin_scale))) * (10**bin_scale) == max_val
            else 1
        )
        if data_range / step < 5:
            step /= 5
        elif data_range / step < 12:
            step /= 2
        hist_bins = np.arange(
            np.floor(min_val / step) * step,
            (np.ceil(max_val / step) + upper_tol) * step,
            step,
        )
    return hist_bins
