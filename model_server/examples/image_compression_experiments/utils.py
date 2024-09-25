"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.

Utility functions for compress and compare examples.
"""

import base64
from io import BytesIO


def encode_image(tensor, visualization_transform):
    """Encode image tensor to byte string."""
    image = visualization_transform(tensor)
    with BytesIO() as output:
        image.save(output, format="png")
        image_string = base64.b64encode(output.getvalue()).decode("ascii")
    image_format = "image/png"
    return {"image": image_string, "image_format": image_format}
