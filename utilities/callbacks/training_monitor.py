# -*- coding: utf-8 -*-
"""
Created on Fri May 15 02:10:06 2020

@author: femiogundare
"""

from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class TrainingMonitor(BaseLogger):
    def __init__(self, fig_path, json_path, start_at=0):
        
        super(TrainingMonitor, self).__init__()
        self.fig_path = fig_path
        self.json_path = json_path
        self.start_at = start_at
        
    def on_train_begin(self, logs={}):
        # Initialize the history dictionary
        self.H = {}
        
        # If the JSON history path exists, load the training history
        if self.json_path is not None:
            if os.path.exists(self.json_path):
                self.H = json.loads(open(self.json_path).read())
                
                # Check to see if a starting epoch was supplied
                if self.start_at > 0:
                    # Loop over the entries in the history log and trim any entries that are past the starting epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.start_at]
        
    def on_epoch_end(self, epoch, logs={}):
        # Loop over the logs and update the loss, accuracy, etc. for the entire training process
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(np.float64(v))
            self.H[k] = l
            
        # Check to see if the training history should be serialized to file
        if self.json_path is not None:
            f = open(self.json_path, "w")
            f.write(json.dumps(self.H))
            f.close()
        
        # Ensure at least two epochs have passed before plotting
        if len(self.H["loss"]) > 1:
            # Plot the training loss and auc
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["auc"], label="train_auc")
            plt.plot(N, self.H["val_auc"], label="val_auc")
            plt.title("Training Loss and AUC [Epoch {}]".format(
                len(self.H["loss"])))
            plt.xlabel("Epoch")
            plt.ylabel("Loss/AUC")
            #plt.ylabel("Loss")
            plt.legend()
            
            # save the figure
            plt.savefig(self.fig_path)
            plt.close()