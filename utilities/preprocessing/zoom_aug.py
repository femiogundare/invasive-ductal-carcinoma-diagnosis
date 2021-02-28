# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 01:53:05 2021

@author: femiogundare
"""


import numpy as np
import cv2


def zoom_aug(img, zoom_var=1.5, seed=None):
    """Performs a random spatial zoom of a Numpy image array.
    # Arguments
        img: Numpy image array.
        zoom_var: zoom range multiplier for width and height.
        seed: Random seed.
    # Returns
        Zoomed Numpy image array.
    """
    scale = np.random.RandomState(seed).uniform(low=1 / zoom_var, high=zoom_var)
    resized_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    return resized_img