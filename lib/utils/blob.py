# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


def prep_im_for_blob(im, pixel_means, pixels_stds, channel_order,
                     max_pixel_range, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    pixel_means = np.asarray(pixel_means).reshape((1, 1, 3)).astype(np.float32)
    pixels_stds = np.asarray(pixels_stds).reshape((1, 1, 3)).astype(np.float32)
    im = im.astype(np.float32, copy=False)
    if max_pixel_range == 1:
        im /= 255.0
    im -= pixel_means
    im /= pixels_stds
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    if channel_order == 'RGB':
        im = im[:, :, ::-1]

    return im, im_scale
