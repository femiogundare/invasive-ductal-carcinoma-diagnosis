# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 10:40:06 2020

@author: femiogundare
"""

import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical


class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, preprocessors=None, aug=None, binarize=True, n_classes=2):
        #store the variables
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.n_classes = n_classes
        
        #open the HDF5 database and ccheck for the total number of images in the database
        self.db = h5py.File(name=dbPath, mode='r')
        self.numImages = self.db['labels'].shape[0]
        
    def generator(self, passes=np.inf):
        
        epochs = 0
        
        #loop infinitely; the model will stop when the desired epoch is reached
        while epochs < passes:
            #loop over and generate images in batches
            for i in np.arange(0, self.numImages, self.batchSize):
                images = self.db['images'][i : i + self.batchSize]
                labels = self.db['labels'][i : i + self.batchSize]
                
                #check whether or not the labels should be binarized
                if self.binarize:
                    labels = to_categorical(labels, self.n_classes)
                
                #check whether or not any preprocessing should be done to the images
                if self.preprocessors is not None:
                    #initialize a list of processed images
                    procImages = []
                    
                    #loop over the images
                    for image in images:
                        #loop over the preprocessors and apply each to the image
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                            
                        #update the list of the processed image
                        procImages.append(image)
                        
                    #convert the processed images to array
                    images = np.array(procImages)
                    
                #if data augmentation is to be applied
                if self.aug is not None:
                    images = np.stack([self.aug(image=image)['image'] for image in images], axis=0)
                    
                yield images, np.array(labels)
                
            epochs += 1
            
    def close(self):
        #close the database
        self.db.close()