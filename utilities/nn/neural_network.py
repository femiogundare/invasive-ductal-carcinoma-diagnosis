# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 21:03:04 2021

@author: femiogundare
"""

import efficientnet.tfkeras as efn
from tensorflow.keras import backend as K
from tensorflow.keras.applications import ResNet50, DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Concatenate



class NeuralNetwork:
    """
    Convolutional Neural Network Architecture to train the histopathology images on.
    """
    
    @staticmethod
    def build(name, width, height, depth, n_classes, reg=0.8):
        """
        Args:
            name: name of the network
            width: width of the images
            height: height of the images
            depth: number of channels of the images
            reg: regularization value
        """
        
        # If Keras backend is TensorFlow
        inputShape = (height, width, depth)
        chanDim = -1
        
        # If Keras backend is Theano
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
            
        # Define the base model architecture
        if name=='EfficientNetB0':
            base_model = efn.EfficientNetB0(weights='imagenet', include_top=False, input_shape=inputShape)
        elif name=='EfficientNetB1':
            base_model = efn.EfficientNetB1(weights='imagenet', include_top=False, input_shape=inputShape)
        elif name=='EfficientNetB2':
            base_model = efn.EfficientNetB2(weights='imagenet', include_top=False, input_shape=inputShape)
        elif name=='EfficientNetB3':
            base_model = efn.EfficientNetB3(weights='imagenet', include_top=False, input_shape=inputShape)
        elif name=='EfficientNetB4':
            base_model = efn.EfficientNetB4(weights='imagenet', include_top=False, input_shape=inputShape)
        elif name=='EfficientNetB5':
            base_model = efn.EfficientNetB5(weights='imagenet', include_top=False, input_shape=inputShape)
        elif name=='EfficientNetB6':
            base_model = efn.EfficientNetB6(weights='imagenet', include_top=False, input_shape=inputShape)
        elif name=='ResNet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=inputShape)
        elif name=='DenseNet121':
            base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=inputShape)
            
        #x1 = GlobalMaxPooling2D()(base_model.output)    # Compute the max pooling of the base model output
        #x2 = GlobalAveragePooling2D()(base_model.output)    # Compute the average pooling of the base model output  
        #x3 = Flatten()(base_model.output)    # Flatten the base model output
        
        #x = Concatenate(axis=-1)([x1, x2, x3])
        
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dropout(0.5)(x)
        """
        # First Dense => Relu => BN => DO
        fc_layer_1 = Dense(512, kernel_regularizer=l2(reg))(x)
        activation_1 = Activation('relu')(fc_layer_1)
        batch_norm_1 = BatchNormalization(axis=-1)(activation_1)
        dropout_1 = Dropout(0.5)(batch_norm_1)
        
        # First Dense => Relu => BN => DO
        fc_layer_2 = Dense(256, kernel_regularizer=l2(reg))(dropout_1)
        activation_2 = Activation('relu')(fc_layer_2)
        batch_norm_2 = BatchNormalization(axis=-1)(activation_2)
        dropout_2 = Dropout(0.5)(batch_norm_2)
        
        # Add the output layer
        output = Dense(n_classes, kernel_regularizer=l2(reg), activation='softmax')(dropout_2)
        """
        output = Dense(n_classes, kernel_regularizer=l2(reg), activation='softmax')(x)
        
        
        # Create the model
        model = Model(inputs=base_model.inputs, outputs=output)
        
        return model
