"""
Code is adapted from:
https://stackoverflow.com/questions/3490727/what-are-some-methods-to-analyze-image-brightness-using-python
https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/tdpf/1/1/art00005#:~:text=Y

Ouput: #2, #4 gave almost identical results. Method #1 closely followed #3 (with a few exceptions).
Speed: Method #2, #3, and #4 were about evenly fast (taking ~140ms to process 20 images of varying sizes),
Method #1 was only slightly slower (~180ms)

"""
import math
from typing import Callable

import numpy as np
from PIL import Image, ImageStat

from src.emotion.features.detections import Detections


# Convert image to greyscale, return average pixel brightness.
def average_brightness(img_arr: np.ndarray, detections: Detections) -> Detections:
    im = Image.fromarray(np.uint8(img_arr))
    im_gray = im.convert("L")
    stat = ImageStat.Stat(im_gray)
    brightness = stat.mean[0]
    detections.brightness = brightness
    return detections


# Convert image to greyscale, return RMS pixel brightness.
def rms_brightness(img_arr: np.ndarray, detections: Detections) -> Detections:
    im = Image.fromarray(np.uint8(img_arr))
    im_gray = im.convert("L")
    stat = ImageStat.Stat(im_gray)
    brightness = stat.mean[0]
    detections.brightness = brightness
    return detections


# Average pixels, then transform to "perceived brightness".
def perceived_brightness_average(
    img_arr: np.ndarray, detections: Detections
) -> Detections:
    im = Image.fromarray(np.uint8(img_arr))
    stat = ImageStat.Stat(im)
    r, g, b = stat.mean
    brightness = math.sqrt(0.299 * (r**2) + 0.587 * (g**2) + 0.114 * (b**2))
    detections.brightness = brightness
    return detections


# RMS of pixels, then transform to "perceived brightness".
def perceived_brightness_rms(img_arr: np.ndarray, detections: Detections) -> Detections:
    im = Image.fromarray(np.uint8(img_arr))
    stat = ImageStat.Stat(im)
    r, g, b = stat.rms
    brightness = math.sqrt(0.299 * (r**2) + 0.587 * (g**2) + 0.114 * (b**2))
    detections.brightness = brightness
    return detections


def brightness_factory(
    method_name: str,
) -> Callable[[np.ndarray, Detections], Detections]:
    if method_name == "average_brightness":
        return average_brightness
    elif method_name == "rms_brightness":
        return rms_brightness
    elif method_name == "perceived_brightness_average":
        return perceived_brightness_average
    elif method_name == "perceived_brightness_rms":
        return perceived_brightness_rms
    else:
        raise ValueError(f"Unsupported method name: {method_name}")
