# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 22:17:08 2021

@author: femiogundare
"""

# Import the required packages
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt

import os
import h5py
import progressbar
import json
import numpy as np
import pandas as pd
from imutils import paths
import efficientnet.tfkeras as efn
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from utilities.preprocessing.stain_normalization import StainNormalization
from utilities.preprocessing.meanpreprocessor import MeanPreprocessor
from utilities.preprocessing.hematoxylin_eosin_aug import hematoxylin_eosin_aug
from utilities.preprocessing.zoom_aug import zoom_aug
from utilities.preprocessing.normal_aug import normal_aug
from utilities.io.hdf5datasetgenerator import HDF5DatasetGenerator
from utilities.metrics.metrics_for_scoring import *
from utilities.others import plot_confusion_matrix
from config import idc_config as config


# Initialize the Config class in the config script
configs = config.Config()
# Put all the arguments of the argparse in a dictionary by calling the get_config method of the Config class
configs_dict = configs.get_config()
TRAIN_HDF5_PATH = configs_dict['training_hdf5_path']
TEST_HDF5_PATH = configs_dict['test_hdf5_path']
NUM_CLASSES = configs_dict['num_classes']

IMAGE_HEIGHT = configs_dict['image_height']
IMAGE_WIDTH = configs_dict['image_width']
#CROP_IMAGE_HEIGHT = configs_dict['crop_image_height']
#CROP_IMAGE_WIDTH = configs_dict['crop_image_width']
N_CHANNELS = configs_dict['n_channels']
TTA_STEPS = configs_dict['tta_steps']

BATCH_SIZE = configs_dict['batch_size']
NETWORK_NAME = configs_dict['network_name']
AUGMENTATION_TYPE = configs_dict['augmentation_type']

OUTPUT_DIR = configs_dict['output_dir']
DATASET_MEAN_PATH = OUTPUT_DIR + '/idc_dataset_mean.json'
WEIGHTS_PATH = OUTPUT_DIR + '/weights/' + NETWORK_NAME + '.hdf5' 

RESULT_DIR = configs_dict['result_dir']
PREDICTIONS_DIR = configs_dict['predictions_dir']


classes_names = ['Non-IDC', 'IDC']

# Get the test labels
print('Obtaining the test labels...')
testLabels = h5py.File(TEST_HDF5_PATH, mode='r')['labels']
testLabels = np.array(testLabels)

# Initialize the preprocessors
print('Initializing the preprocessors...')
#sn = StainNormalization()
means = json.loads(open(DATASET_MEAN_PATH, 'r').read())
mp = MeanPreprocessor(rMean=means['R'], gMean=means['G'], bMean=means['B'])

# Load the pretrained network
print('Loading the model...')
model = load_model(WEIGHTS_PATH, compile=False)

print('Name of model: {}'.format(NETWORK_NAME))

# Select augmentation type to be performed during test-time augmentation
if AUGMENTATION_TYPE=='hematoxylin_eosin':
    aug = hematoxylin_eosin_aug
elif AUGMENTATION_TYPE=='zoom_aug':
    aug = zoom_aug
elif AUGMENTATION_TYPE=='normal_aug':
    aug = normal_aug

# Initialize the test generator (and allow for test time augmentation to be performed)
print('Initializing the test generator...')
testGen = HDF5DatasetGenerator(
        TEST_HDF5_PATH, BATCH_SIZE, preprocessors=[mp], aug=aug, n_classes=NUM_CLASSES,
        )

"""
# Predict on the test data
print('Predicting on the test data...')
predictions = model.predict_generator(
        testGen.generator(), steps=(testGen.numImages//BATCH_SIZE),
        max_queue_size=BATCH_SIZE*2
        )
"""

# Perform predictions on the test data using test-time augmentation
print(f'Predicting on the test data with TTA of {TTA_STEPS} steps...')
predictions_with_tta = []

for i in range(TTA_STEPS):
    print('TTA step {}'.format(i+1))
    predictions = model.predict_generator(
        testGen.generator(), steps=(testGen.numImages//BATCH_SIZE),
        max_queue_size=BATCH_SIZE*2
    )
    predictions_with_tta.append(predictions)
    
predictions = (np.array(predictions_with_tta).sum(axis=0)) / TTA_STEPS


# Check the model performance
print('Checking the model performance on the test data...')
conf_matrix = confusion_matrix(testLabels, predictions.argmax(axis=1))
tn, fn, tp, fp = conf_matrix[0][0], conf_matrix[1][0], conf_matrix[1][1], conf_matrix[0][1]
auc = roc_auc_score(testLabels, predictions[:, 1])
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
ppv = tp/(tp+fp)
npv = tn/(tn+fn)
J = (sensitivity + specificity - 1)
print('AUC: {:.4f}'.format(auc))
print('Sensitivity: {:.4f}'.format(sensitivity))
print('Specificity: {:.4f}'.format(specificity))
print('Positive Predictive Value: {:.4f}'.format(ppv))
print('Negative Predictive Value: {:.4f}'.format(npv))
print("Youden's J Statistic: {:.4f}".format(J))
print('Confusion Matrix: \n{}'.format(conf_matrix))


# Store the predictions to a csv file
print('Storing the predictions to csv...')
names_of_images_in_dataset = OUTPUT_DIR + '/names_of_images.json'
names = json.loads(open(names_of_images_in_dataset).read())
names_of_test_images = names['test_names']

df = pd.DataFrame(
    dict(
        name=names_of_test_images, 
        label=testLabels, 
        prediction=predictions[:, 1]
        )
    )

df.to_csv(PREDICTIONS_DIR+'/'+NETWORK_NAME+'_predictions.csv', index=False)


# Plot the confusion matrix
plt.figure()
plot_confusion_matrix(conf_matrix, classes=classes_names,
                      title='Confusion matrix')
#plt.show()
plt.savefig(RESULT_DIR + '/'+NETWORK_NAME + '/confusion_matrix.png')


# Plot the ROC Curve and save to png
plt.figure()
fpr, tpr, _ = roc_curve(testLabels, predictions[:, 1], pos_label=1)
plt.style.use('seaborn')
plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.4f)' % auc)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.title('Receiving Operating Characteristic Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
#plt.show()
plt.savefig(RESULT_DIR + '/'+NETWORK_NAME+ '/roc_curve.png')