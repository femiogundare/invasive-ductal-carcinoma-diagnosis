# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 20:29:51 2021

@author: femiogundare
"""

# Import the required libraries
import os
from os import listdir
import json
import cv2
import pickle
import progressbar
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utilities.io.hdf5datasetwriter import HDF5DatasetWriter
from config import idc_config as config
from utilities.build.build_dataset import extract_coords
from utilities.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor


# Initialize the Config class in the config script
configs = config.Config()
# Put all the arguments of the argparse in a dictionary by calling the get_config method of the Config class
configs_dict = configs.get_config()
# Load the supplied arguments from the config file
DATA_PATH = configs_dict['base_path'] 
IMAGES_PATH = configs_dict['images_dir']
TRAIN_HDF5_PATH = configs_dict['training_hdf5_path']
VAL_HDF5_PATH = configs_dict['validation_hdf5_path']
TEST_HDF5_PATH = configs_dict['test_hdf5_path']
SEED = configs_dict['random_seed']

IMAGE_HEIGHT = configs_dict['image_height']
IMAGE_WIDTH = configs_dict['image_width']
N_CHANNELS = configs_dict['n_channels']

OUTPUT_DIR = configs_dict['output_dir']
DATASET_MEAN_PATH = OUTPUT_DIR + '/idc_dataset_mean.json'
LABEL_ENCODER_PATH = OUTPUT_DIR + '/label_encoder.cpickle'
NAMES_OF_IMAGES_IN_DATASET = OUTPUT_DIR + '/names_of_images.json'




FOLDER = listdir(IMAGES_PATH)
TOTAL_IMAGES = 277524

# Create a dataframe containing the IDs of the patients, Path to each images, the Target value and the Image name
data = pd.DataFrame(index=np.arange(0, TOTAL_IMAGES), columns=["path", "target", "patient_id", "image_name"])

k = 0
for n in range(len(FOLDER)):
    patient_id = FOLDER[n]
    patient_path = IMAGES_PATH + '/' + patient_id 
    for c in [0,1]:
        class_path = patient_path + "/" + str(c) + "/"
        subfiles = listdir(class_path)
        for m in range(len(subfiles)):
            image_path = subfiles[m]
            data.iloc[k]["path"] = class_path + image_path
            data.iloc[k]["target"] = c
            data.iloc[k]["patient_id"] = patient_id
            data.iloc[k]["image_name"] = image_path
            k += 1 

print(data.shape)
print(f'There are {data.shape[0]} images in the dataset.')
          
# Ensure the target variable is in integer format
data.target = data.target.astype(np.int)

# Encode the target variable
print('Encoding the target variable...')
le = LabelEncoder()
data.target = le.fit_transform(data.target)

# Get the unique patient ids in the dataset, and split them into training, validation, and test ids
patient_ids = data.patient_id.unique()
split_size = round(len(patient_ids)/10)   # split ratio is 10%

train_ids, test_ids = train_test_split(patient_ids, test_size=split_size, random_state=SEED)

train_ids, val_ids = train_test_split(train_ids, test_size=split_size, random_state=SEED)


# Get the training, validation, and test dataframes based on the patient ids
training_df = data.loc[data.patient_id.isin(train_ids), :].copy()
validation_df = data.loc[data.patient_id.isin(val_ids), :].copy()
test_df = data.loc[data.patient_id.isin(test_ids), :].copy()

idc_class_freq = training_df['target'].sum()/training_df.shape[0]
non_idc_class_freq = 1 - idc_class_freq

num_train = 18000
num_val = 3000
num_test = 3000

training_df_idc = training_df[training_df['target']==1].sample(int(round(idc_class_freq*num_train)))
training_df_non_idc = training_df[training_df['target']==0].sample(int(round(non_idc_class_freq*num_train)))
training_df = pd.concat([training_df_idc, training_df_non_idc], axis=0).sample(num_train)

validation_df_idc = validation_df[validation_df['target']==1].sample(int(round(idc_class_freq*num_val)))
validation_df_non_idc = validation_df[validation_df['target']==0].sample(int(round(non_idc_class_freq*num_val)))
validation_df = pd.concat([validation_df_idc, validation_df_non_idc], axis=0).sample(num_val)

test_df_idc = test_df[test_df['target']==1].sample(int(round(idc_class_freq*num_test)))
test_df_non_idc = test_df[test_df['target']==0].sample(int(round(non_idc_class_freq*num_test)))
test_df = pd.concat([test_df_idc, test_df_non_idc], axis=0).sample(num_test)

print(f'There are {training_df.shape[0]} images in the training set.')
print(f'There are {validation_df.shape[0]} images in the validation set.')
print(f'There are {test_df.shape[0]} images in the test set.')
print(training_df.isnull().sum())
print(validation_df.isnull().sum())
print(test_df.isnull().sum())

"""
# Add the coordinates (x, y) where each patch is found in the whole mount sample to the dataframe
#training_df = extract_coords(training_df)
#validation_df = extract_coords(validation_df)
#test_df = extract_coords(test_df)
"""

# Construct a list pairing the images paths, images labels and the output hdf5 files of the training,
# validation and test sets
print('Pairing the images paths, images labels and the output hdf5 files of the training, validation and test sets...')
datasets = [
        ("train", training_df['path'], training_df['target'], TRAIN_HDF5_PATH),
        ("val", validation_df['path'], validation_df['target'], VAL_HDF5_PATH),
        ("test", test_df['path'], test_df['target'], TEST_HDF5_PATH)
        ]

# Initialize the image preprocessor and the RGB channels mean
aap = AspectAwarePreprocessor(width=IMAGE_HEIGHT, height=IMAGE_WIDTH, inter=cv2.INTER_AREA)
R, G, B = [], [], []

#loop over the datasets tuples
for dType, images, labels, outputPath in datasets:
    #create the HDF5 writer
    print("Building {}...".format(outputPath))
    writer = HDF5DatasetWriter(
            dims=(len(images), IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS), outputPath=outputPath, dataKey="images", buffSize=1000
            )
    
    pbar = ["Building Dataset: ", progressbar.Percentage(), " ", 
            progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(images),
                                   widgets=pbar).start()
    
    #loop over the image paths
    for i, (image, label) in enumerate(zip(images, labels)):
        #load and preprocess the image
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = aap.preprocess(image)
        
        #compute the mean of each channel in the training set
        if dType=="train":
            r, g, b = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        
        #add the processed images to the HDF5 writer
        writer.add(rows=[image], labels=[label])
        pbar.update(i)
        
    pbar.finish()
    writer.close()
    
    
# Serialize the means to a json file
print('Serializing the means...')
dic = {'R' : np.mean(R), 'G' : np.mean(G), 'B' : np.mean(B)}
f = open(DATASET_MEAN_PATH, 'w')
f.write(json.dumps(dic))
f.close()

# Serialize the label encoder to a json file
print('Serializing the label encoder...')
f = open(LABEL_ENCODER_PATH, 'wb')
f.write(pickle.dumps(le))
f.close()

# Serialize the names of images in the training, validation, and test sets to json
print('Serializing the names of the images...')
train_names = training_df['image_name']
val_names = validation_df['image_name']
test_names = test_df['image_name']
dic = {'train_names' : list(train_names), 'val_names' : list(val_names), 'test_names' : list(test_names)}
f = open(NAMES_OF_IMAGES_IN_DATASET, 'w')
f.write(json.dumps(dic))
f.close()