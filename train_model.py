# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 18:34:39 2021

@author: femiogundare
"""


# Import the required libraries and packages
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import h5py
import json
import numpy as np
#import albumentations as A
#from albumentations import Compose, RandomRotate90, Transpose, Flip, OneOf, CLAHE, IAASharpen, IAAEmboss, RandomBrightnessContrast, JpegCompression, Blur, GaussNoise, HueSaturationValue, ShiftScaleRotate, Normalize
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.metrics import AUC
from tensorflow.keras.losses import BinaryCrossentropy 
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, EarlyStopping, ReduceLROnPlateau
from utilities.preprocessing.simplepreprocessor import SimplePreprocessor
from utilities.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from utilities.preprocessing.meanpreprocessor import MeanPreprocessor
from utilities.preprocessing.stain_normalization import StainNormalization
from utilities.preprocessing.hematoxylin_eosin_aug import hematoxylin_eosin_aug
from utilities.preprocessing.zoom_aug import zoom_aug
from utilities.preprocessing.normal_aug import normal_aug
from utilities.io.hdf5datasetgenerator import HDF5DatasetGenerator
from utilities.nn.neural_network import NeuralNetwork
from utilities.callbacks.poly_decay import poly_decay
from utilities.callbacks.cyclical_learning_rate import CyclicLR
from utilities.callbacks.training_monitor import TrainingMonitor
from utilities.metrics.metrics_for_compiling import sensitivity, specificity
from config import idc_config as config


# Initialize the Config class in the config script
configs = config.Config()
# Put all the arguments of the argparse in a dictionary by calling the get_config method of the Config class
configs_dict = configs.get_config()
# Load the supplied arguments from the config file
print('Loading the supplied arguments from the config file...')
TRAIN_HDF5_PATH = configs_dict['training_hdf5_path']
VAL_HDF5_PATH = configs_dict['validation_hdf5_path']
NUM_CLASSES = configs_dict['num_classes']

IMAGE_HEIGHT = configs_dict['image_height']
IMAGE_WIDTH = configs_dict['image_width']
N_CHANNELS = configs_dict['n_channels']

MIN_LR = configs_dict['min_lr']
MAX_LR = configs_dict['max_lr']
BATCH_SIZE = configs_dict['batch_size']
STEP_SIZE = configs_dict['step_size']
CLR_METHOD = configs_dict['clr_method']  #default is 'Traingular'
NUM_EPOCHS = configs_dict['n_epochs']
FACTOR = configs_dict['factor']
PATIENCE = configs_dict['patience']

NETWORK_NAME = configs_dict['network_name']
AUGMENTATION_TYPE = configs_dict['augmentation_type']

OPTIMIZER = configs_dict['optimizer']
if OPTIMIZER=='Adam':
    OPTIMIZER = Adam(learning_rate=MAX_LR)
elif OPTIMIZER=='SGD':
    OPTIMIZER = SGD(learning_rate=MAX_LR, momentum=0.9, nesterov=True)
elif OPTIMIZER=='RMSprop':
    OPTIMIZER = RMSprop(learning_rate=MAX_LR)
else:
    print('Specified optimizer not allowed for this task')
    sys.exit(-1)
    
    
OUTPUT_DIR = configs_dict['output_dir']
DATASET_MEAN_PATH = OUTPUT_DIR + '/idc_dataset_mean.json'
CLASS_WEIGHTS_PATH = OUTPUT_DIR + '/idc_dataset_class_weight.json'
WEIGHTS_PATH = OUTPUT_DIR + '/weights/' + NETWORK_NAME + '.hdf5' 
PLOT_PATH = OUTPUT_DIR + '/plots/' + NETWORK_NAME + '.png'
MODEL_PATH = OUTPUT_DIR + '/models/' + NETWORK_NAME + '.hdf5'

MONITOR_DIR = OUTPUT_DIR + '/monitor/' + NETWORK_NAME
MONITOR_PLOTS_PATH = MONITOR_DIR + '/fig_path'
MONITOR_JSON_PATH = MONITOR_DIR + '/json_path'


# Initialize the preprocessors
sp = SimplePreprocessor(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
itap = ImageToArrayPreprocessor()
sn = StainNormalization()

means = json.loads(open(DATASET_MEAN_PATH, 'r').read())
mp = MeanPreprocessor(rMean=means['R'], gMean=means['G'], bMean=means['B'])

#calculate the frequencies of the idc and non idc classes in the training labels
#the calculated frequencies will serve as the class weights in the generator function
print('Computing the frequencies and weights of each class...')
trainLabels = h5py.File(TRAIN_HDF5_PATH, mode='r')['labels']
trainLabels = np.array(trainLabels)
train_idc_freq = trainLabels.sum(axis=0)/trainLabels.shape[0]
train_non_idc_freq = 1-train_idc_freq
train_idc_weight, train_non_idc_weight = train_non_idc_freq, train_idc_freq
print(f'The IDC class and the Non-IDC class have weights of {train_idc_weight} and {train_non_idc_weight} respectively in the training set')
# Serialize the weights to a json file
print('Serializing the weights to json...')
dic = {'train_idc_weight' : train_idc_weight, 'train_non_idc_weight' : train_non_idc_weight}
f = open(CLASS_WEIGHTS_PATH, 'w')
f.write(json.dumps(dic))
f.close()

# Select augmentation type 
if AUGMENTATION_TYPE=='hematoxylin_eosin':
    aug = hematoxylin_eosin_aug
elif AUGMENTATION_TYPE=='zoom_aug':
    aug = zoom_aug
elif AUGMENTATION_TYPE=='normal_aug':
    aug = normal_aug
    

#initialize the training and validation dataset generators
print('Initializing the training and validation generators...')
trainGen = HDF5DatasetGenerator(
        dbPath=TRAIN_HDF5_PATH, batchSize=BATCH_SIZE, 
        preprocessors=[sp, mp, itap], aug=aug, n_classes=NUM_CLASSES,
        )

valGen = HDF5DatasetGenerator(
        dbPath=VAL_HDF5_PATH, batchSize=BATCH_SIZE, 
        preprocessors=[sp, mp, itap], n_classes=NUM_CLASSES,
        )


print('Model: {}'.format(NETWORK_NAME))
# Initialize and compile the model
print('Compiling the model...')
metrics = [sensitivity, specificity, AUC()]
model = NeuralNetwork.build(name=NETWORK_NAME, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, depth=N_CHANNELS, 
                      n_classes=NUM_CLASSES
                      )
print(model.summary())
model.compile(loss=BinaryCrossentropy(label_smoothing=0.1), 
              optimizer=OPTIMIZER, 
              metrics=metrics
              )

# Initialize the list of callbacks
print('Initializing the list of callbacks...')
print('1. Learning Rate')
print(f'Learning rate to be reduced by a factor of {FACTOR} if loss does not decrease in {PATIENCE} epochs')
lr_schedule = ReduceLROnPlateau(monitor='val_loss',
                                    factor=FACTOR,
                                    patience=PATIENCE,
                                    verbose=1,
                                    mode='auto',
                                    min_lr=MIN_LR,)

"""
if NETWORK_NAME in ['EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5']:
    # Set learning rate schudule to Cyclic Learning Rate
    print(f'Using {CLR_METHOD} with a minimum learning rate of {MIN_LR}, maximum learning rate of {MAX_LR} and step size of {STEP_SIZE}')
    lr_schedule = CyclicLR(
        mode=CLR_METHOD, 
        base_lr=MIN_LR, 
        max_lr=MAX_LR, 
        step_size= STEP_SIZE * (trainGen.numImages //  BATCH_SIZE)
    )
    
else:
    # Reduce learning rate on plateau
    print(f'Learning rate to be reduced by a factor of {FACTOR} if loss does not decrease in {PATIENCE} epochs')
    lr_schedule = ReduceLROnPlateau(monitor='val_loss',
                                    factor=FACTOR,
                                    patience=PATIENCE,
                                    verbose=1,
                                    mode='auto',
                                    min_lr=MIN_LR,)
"""

print('2. Model Checkpoint')
model_checkpoint = ModelCheckpoint(WEIGHTS_PATH, monitor='val_loss', mode='min', save_best_only=True, verbose=1)

print('3. Training Monitor')
print("[INFO process ID: {}]".format(os.getpid()))
figPath = os.path.sep.join([MONITOR_PLOTS_PATH, "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([MONITOR_JSON_PATH, "{}.json".format(os.getpid())])
training_monitor = TrainingMonitor(fig_path=figPath, json_path=jsonPath)

print('4. Terminate on NaN')
terminate_on_nan = TerminateOnNaN()

print('5. Early Stopping')
early_stopping = EarlyStopping(monitor='val_loss', patience=4, mode='min', verbose=1, restore_best_weights=True)


callbacks = [lr_schedule, training_monitor, terminate_on_nan, early_stopping, model_checkpoint]

# Check to see if a GPU is available for training or not
print('GPU is', 'Available' if tf.test.is_gpu_available() else 'Not Available')

# Train the model
print('Training the model...')
H = model.fit_generator(generator=trainGen.generator(), 
                        steps_per_epoch=trainGen.numImages//BATCH_SIZE, 
                        validation_data=valGen.generator(),
                        validation_steps=valGen.numImages//BATCH_SIZE,
                        epochs=NUM_EPOCHS, callbacks=callbacks, verbose=1, 
                        class_weight={0:train_non_idc_weight, 1:train_idc_weight}
                        )

print('Serializing the model...')
model.save(MODEL_PATH, overwrite=True)

# Close the HDF5 datasets
trainGen.close()
valGen.close()


# Loss and AUC Curves for the trained model
plt.figure()
plt.plot(np.arange(0, len(H.history['auc'])), H.history['auc'], '-o', label='Train AUC', color='#ff7f0e')
plt.plot(np.arange(0, len(H.history['val_auc'])), H.history['val_auc'], '-o', label='Val AUC', color='#1f77b4')
x = np.argmax( H.history['val_auc'] ); y = np.max( H.history['val_auc'] )
xdist = plt.xlim()[1] - plt.xlim()[0]; ydist = plt.ylim()[1] - plt.ylim()[0]
plt.scatter(x,y,s=200,color='#1f77b4'); plt.text(x-0.03*xdist,y-0.13*ydist,'max auc\n%.2f'%y,size=14)
plt.ylabel('AUC',size=14); plt.xlabel('Epoch',size=14)
plt.legend(loc=2)
plt2 = plt.gca().twinx()
plt2.plot(np.arange(0, len(H.history['loss'])), H.history['loss'], '-o', label='Train Loss', color='#2ca02c')
plt2.plot(np.arange(0, len(H.history['val_loss'])), H.history['val_loss'], '-o', label='Val Loss', color='#d62728')
x = np.argmin( H.history['val_loss'] ); y = np.min( H.history['val_loss'] )
ydist = plt.ylim()[1] - plt.ylim()[0]
plt.scatter(x,y,s=200,color='#d62728'); plt.text(x-0.03*xdist,y+0.05*ydist,'min loss',size=14)
plt.ylabel('Loss',size=14)
plt.title(f'Training AUC and Loss Curves ({NETWORK_NAME})',size=18)
plt.legend(loc=3)
plt.show()
plt.savefig(PLOT_PATH)