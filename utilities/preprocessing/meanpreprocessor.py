# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 22:30:36 2020

@author: femiogundare
"""

import cv2

class MeanPreprocessor:
    def __init__(self, rMean, gMean, bMean):
        #store the R, G, B means across the training set
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean
        
    def preprocess(self, image):
        #split the image into its resppective Red, Blue and Green channels
        B, G, R = cv2.split(image.astype("float32"))
        
        #subtract the means for each channels
        B -= self.bMean
        G -= self.gMean
        R -= self.rMean
        
        #merge the channels back and return the image
        return cv2.merge([B, G, R])