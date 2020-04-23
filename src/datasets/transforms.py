import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance

import numpy as np
import numbers
import types
import collections
import warnings

import torchvision.transforms as transforms
from torchvision.transforms import functional as TF

__all__ = ["AlignedRandomAffine", "AlignedRandomFlip"]

class AlignedRandomAffine(object):
    """ Simultaneously random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to desactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """
    def __init__(self, degrees=[-180,180], scale_ranges=(0.8,1.25), shears=(-10.,10.)):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        translate, img_size = None, None
        self.rand_affine_params = transforms.RandomAffine.get_params(degrees, translate, scale_ranges, shears, img_size)  # (degrees=[-180,180], translate=((-5,5), (-5,5)), scale=(0.8,1.25), shear=(-10.,10.), img_size=img_size)

    def __call__(self, img):
        return TF.affine(img, *self.rand_affine_params, resample=Image.BICUBIC, fillcolor=0)


class AlignedRandomFlip(object):
    """ Simultaneously horizontally and vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p
        self.rand_horizontal = random.random()  # reproducible, very np.random cannot?
        self.rand_vertical = random.random()

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if self.rand_horizontal < self.p:
            img = TF.hflip(img)
        if self.rand_vertical < self.p:
            img = TF.vflip(img)
        return img


class RandomAdjustColor(object):
    """ NOTICE: this implementation will keep the same random param for each instance. 
    
    Randomly apply color tuning to the input images.
    Remember to assign seed in worker_init_fn()

    Args for input image:
        brightness_range=(0.5, 2.) - nonnegative number: 0 gives a black image, 1 gives the original image while 2 increases the brightness by a factor of 2.
        contrast_range=(0.5, 2.) - nonnegative number: 0 gives a solid gray image, 1 gives the original image while 2 increases the contrast by a factor of 2.
        gamma_range=(0.5, 2.) - nonnegative number: gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.
    """

    def __init__(self, brightness_range, contrast_range, gamma_range):
        # self.brightness_range = brightness_range
        # self.contrast_range = contrast_range
        # self.gamma_range = gamma_range
        self.rand_brightness = np.random.uniform(*brightness_range)    # sclar
        self.rand_contrast = np.random.uniform(*contrast_range)    # sclar
        self.rand_gamma = np.random.uniform(*gamma_range)    # sclar

    def __call__(self, img):
        if self.rand_brightness != 1:
            img = TF.adjust_brightness(img, brightness_factor = self.rand_brightness)
        if self.rand_contrast != 1:
            img = TF.adjust_contrast(img, contrast_factor = self.rand_contrast)
        if self.rand_gamma != 1:
            img = TF.adjust_gamma(img, gamma = self.rand_gamma, gain=1)

        return img

