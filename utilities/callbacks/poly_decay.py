# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 20:03:24 2021

@author: femiogundare
"""

from config import idc_config as config

configs = config.Config()
configs_dict = configs.get_config()

NUM_EPOCHS = configs_dict['n_epochs']
INIT_LR = configs_dict['max_lr']

def poly_decay(epoch):
    """Polynomial Learning Rate Decay"""
    # Initialize the maximum number of epochs, base learning rate, and power of the polynomial
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 2.0
    
    # Compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    
    return alpha