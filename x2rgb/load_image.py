import os

import cv2
import torch

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np


def convert_rgb_2_XYZ(rgb):
    # Reference: https://web.archive.org/web/20191027010220/http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    # rgb: (h, w, 3)
    # XYZ: (h, w, 3)
    XYZ = torch.ones_like(rgb)
    XYZ[:, :, 0] = (
        0.4124564 * rgb[:, :, 0] + 0.3575761 * rgb[:, :, 1] + 0.1804375 * rgb[:, :, 2]
    )
    XYZ[:, :, 1] = (
        0.2126729 * rgb[:, :, 0] + 0.7151522 * rgb[:, :, 1] + 0.0721750 * rgb[:, :, 2]
    )
    XYZ[:, :, 2] = (
        0.0193339 * rgb[:, :, 0] + 0.1191920 * rgb[:, :, 1] + 0.9503041 * rgb[:, :, 2]
    )
    return XYZ


def convert_XYZ_2_Yxy(XYZ):
    # XYZ: (h, w, 3)
    # Yxy: (h, w, 3)
    Yxy = torch.ones_like(XYZ)
    Yxy[:, :, 0] = XYZ[:, :, 1]
    sum = torch.sum(XYZ, dim=2)
    inv_sum = 1.0 / torch.clamp(sum, min=1e-4)
    Yxy[:, :, 1] = XYZ[:, :, 0] * inv_sum
    Yxy[:, :, 2] = XYZ[:, :, 1] * inv_sum
    return Yxy


def convert_rgb_2_Yxy(rgb):
    # rgb: (h, w, 3)
    # Yxy: (h, w, 3)
    return convert_XYZ_2_Yxy(convert_rgb_2_XYZ(rgb))


def convert_XYZ_2_rgb(XYZ):
    # XYZ: (h, w, 3)
    # rgb: (h, w, 3)
    rgb = torch.ones_like(XYZ)
    rgb[:, :, 0] = (
        3.2404542 * XYZ[:, :, 0] - 1.5371385 * XYZ[:, :, 1] - 0.4985314 * XYZ[:, :, 2]
    )
    rgb[:, :, 1] = (
        -0.9692660 * XYZ[:, :, 0] + 1.8760108 * XYZ[:, :, 1] + 0.0415560 * XYZ[:, :, 2]
    )
    rgb[:, :, 2] = (
        0.0556434 * XYZ[:, :, 0] - 0.2040259 * XYZ[:, :, 1] + 1.0572252 * XYZ[:, :, 2]
    )
    return rgb


def convert_Yxy_2_XYZ(Yxy):
    # Yxy: (h, w, 3)
    # XYZ: (h, w, 3)
    XYZ = torch.ones_like(Yxy)
    XYZ[:, :, 0] = Yxy[:, :, 1] / torch.clamp(Yxy[:, :, 2], min=1e-6) * Yxy[:, :, 0]
    XYZ[:, :, 1] = Yxy[:, :, 0]
    XYZ[:, :, 2] = (
        (1.0 - Yxy[:, :, 1] - Yxy[:, :, 2])
        / torch.clamp(Yxy[:, :, 2], min=1e-4)
        * Yxy[:, :, 0]
    )
    return XYZ


def convert_Yxy_2_rgb(Yxy):
    # Yxy: (h, w, 3)
    # rgb: (h, w, 3)
    return convert_XYZ_2_rgb(convert_Yxy_2_XYZ(Yxy))


def load_ldr_image(image_path, from_srgb=False, clamp=False, normalize=False):
    # Load png or jpg image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image.astype(np.float32) / 255.0)  # (h, w, c)
    image[~torch.isfinite(image)] = 0
    if from_srgb:
        # Convert from sRGB to linear RGB
        image = image**2.2
    if clamp:
        image = torch.clamp(image, min=0.0, max=1.0)
    if normalize:
        # Normalize to [-1, 1]
        image = image * 2.0 - 1.0
        image = torch.nn.functional.normalize(image, dim=-1, eps=1e-6)
    return image.permute(2, 0, 1)  # returns (c, h, w)


def load_exr_image(image_path, tonemaping=False, clamp=False, normalize=False):
    image = cv2.cvtColor(cv2.imread(image_path, -1), cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image.astype("float32"))  # (h, w, c)
    image[~torch.isfinite(image)] = 0
    if tonemaping:
        # Exposure adjuestment
        image_Yxy = convert_rgb_2_Yxy(image)
        lum = (
            image[:, :, 0:1] * 0.2125
            + image[:, :, 1:2] * 0.7154
            + image[:, :, 2:3] * 0.0721
        )
        lum = torch.log(torch.clamp(lum, min=1e-6))
        lum_mean = torch.exp(torch.mean(lum))
        lp = image_Yxy[:, :, 0:1] * 0.18 / torch.clamp(lum_mean, min=1e-6)
        image_Yxy[:, :, 0:1] = lp
        image = convert_Yxy_2_rgb(image_Yxy)
    if clamp:
        image = torch.clamp(image, min=0.0, max=1.0)
    if normalize:
        image = torch.nn.functional.normalize(image, dim=-1, eps=1e-6)
    return image.permute(2, 0, 1)  # returns (c, h, w)
