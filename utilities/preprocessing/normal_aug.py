# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 23:30:50 2021

@author: femiogundare
"""


import albumentations as A




normal_aug = A.Compose([
        A.RandomRotate90(p=0.7),
        A.OneOf([
            A.HorizontalFlip(p=1), 
            A.VerticalFlip(p=1)]
        )
        ])