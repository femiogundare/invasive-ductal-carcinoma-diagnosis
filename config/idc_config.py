# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 20:06:04 2021

@author: femiogundare
"""

import sys
import argparse


class Config:
    """Config
    Attributes:
        parser: to read all config
        args: argument from argument parser
        config: save config in pairs like key:value
    """
    def __init__(self):
        """Load common and customized settings
        """
        super(Config, self).__init__()
        self.parser = argparse.ArgumentParser(description='Skin Cancer Classification')
        self.config = {}

        # add setting via parser
        self._add_common_setting()
        self._add_customized_setting()
        # get argument parser
        self.args = self.parser.parse_args()
        # load them into config
        self._load_common_setting()
        self._load_customized_setting()

    def _add_common_setting(self):
        # Need be defined each time
        
        # define the data directory --- BASEPATH
        self.parser.add_argument(
            '--base_path', default='C:\\Users\\Dell\\Desktop\\CV Projects\\Invasive Ductal Carcinoma Diagnosis/data', type=str, help='data directory'
            )
        
        # define the path to the images
        self.parser.add_argument(
            '--images_dir', default='C:\\Users\\Dell\\Desktop\\CV Projects\\Invasive Ductal Carcinoma Diagnosis\\data/IDC_regular_ps50_idx5', type=str, help='path to the images'
            )
        
        # define the path to the training hdf5 dataset
        self.parser.add_argument(
            '--training_hdf5_path', default='C:\\Users\\Dell\\Desktop\\CV Projects\\Invasive Ductal Carcinoma Diagnosis\\data\\hdf5/train.hdf5', type=str, 
            help='path to the training hdf5 dataset'
            )
        
        # define the path to the validation hdf5 dataset
        self.parser.add_argument(
            '--validation_hdf5_path', default='C:\\Users\\Dell\\Desktop\\CV Projects\\Invasive Ductal Carcinoma Diagnosis\\data\\hdf5/val.hdf5', type=str, 
            help='path to the validation hdf5 dataset'
            )
        
        # define the path to the test hdf5 dataset
        self.parser.add_argument(
            '--test_hdf5_path', default='C:\\Users\\Dell\\Desktop\\CV Projects\\Invasive Ductal Carcinoma Diagnosis\\data\\hdf5/test.hdf5', type=str, 
            help='path to the test hdf5 dataset'
            )
        
        # define the path to the output directory
        self.parser.add_argument(
            '--output_dir', default='C:\\Users\\Dell\\Desktop\\CV Projects\\Invasive Ductal Carcinoma Diagnosis/output', type=str, help='path to the outputs'
            )
        
        # define the path to the result directory
        self.parser.add_argument(
            '--result_dir', default='C:\\Users\\Dell\\Desktop\\CV Projects\\Invasive Ductal Carcinoma Diagnosis/result', type=str, help='path to the results'
            )
        
        # define the path to the predictions directory
        self.parser.add_argument(
            '--predictions_dir', default='C:\\Users\\Dell\\Desktop\\CV Projects\\Invasive Ductal Carcinoma Diagnosis/predictions', type=str, help='path to the predictions'
            )
        
        
        # Hyper parameters
        self.parser.add_argument('--lr_type', default='Cyclical', type=str,
                                 help="learning rate schedule", choices=['Cyclical', 'Fixed', 'Decayed'])
        
        self.parser.add_argument('--min_lr', default=0.000001, type=float,
                                 help="minimum learning rate")
        
        self.parser.add_argument('--max_lr', default=0.0006, type=float,
                                 help="maximum learning rate")
        
        self.parser.add_argument('--clr_method', default='triangular', type=str,
                                 choices=['triangular', 'triangular2', 'exp_range'],
                                 help="cyclic learning rate method(traingular, triangular2, exp_range)"
                                 )
    
        self.parser.add_argument('--step_size', default=8, type=int,
                                 choices=[i for i in range(2, 9)],
                                 help="step size for cyclic learning rate (2-8)"
                                 )
        
        self.parser.add_argument("--batch_size", default=8, type=int,
                                 help="batch size per epoch")
        
        self.parser.add_argument("--n_epochs", default=25, type=int,
                                 help="#epochs to train the network on")

        self.parser.add_argument('--random_seed', default=47, type=int,
                                 help='desired radom state for numpy and other packages')

        self.parser.add_argument('--optimizer', default='Adam', type=str,
                                 choices=['Adam', 'SGD', 'RMSprop'],
                                 help="optimizer to used (use 'Adam', 'SGD' or 'RMSprop')"
                                 )
         
        self.parser.add_argument('--network_name', default='ResNet50', type=str,
                                 choices=[
                                          'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5',
                                          'ResNet50'
                                          ],
                                 help="name of neural network to be used"
                                 )
        
        self.parser.add_argument("--factor", default=0.25, type=float,
                                 help='factor to reduce learning rate by')
        
        self.parser.add_argument("--patience", default=3, type=int,
                                 help='patience')
        
        # Input images related
        self.parser.add_argument("--image_height", default=50, type=int,
                                   help="image height")
        
        self.parser.add_argument("--image_width", default=50, type=int,
                                   help="image width")
        
        #self.parser.add_argument("--crop_image_height", default=45, type=int,
        #                           help="crop image height")
        
        #self.parser.add_argument("--crop_image_width", default=45, type=int,
        #                           help="crop image width")
        
        self.parser.add_argument("--n_channels", default=3, type=int,
                                   help="number of channels of the images")
        
        self.parser.add_argument("--augmentation_type", default='normal_aug', type=str,
                                 choices=['hematoxylin_eosin', 'zoom', 'normal_aug'],
                                   help="type of augmentation")
        
        self.parser.add_argument("--tta_steps", default=25, type=int,
                                   help="number of test time augmentation steps")
        
        
    def _add_customized_setting(self):
        """Add customized setting
        """
        # define the number of classes to be trained on
        self.parser.add_argument(
            '--num_classes', default=2, type=int, 
            help='#classes to train on'
            )
        
    def _load_common_setting(self):
        """Load default setting from Parser
        """
        
        # Directories and network types
        self.config['base_path'] = self.args.base_path
        self.config['images_dir'] = self.args.images_dir
        self.config['output_dir'] = self.args.output_dir
        self.config['result_dir'] = self.args.result_dir
        self.config['predictions_dir'] = self.args.predictions_dir
        self.config['training_hdf5_path'] = self.args.training_hdf5_path
        self.config['validation_hdf5_path'] = self.args.validation_hdf5_path
        self.config['test_hdf5_path'] = self.args.test_hdf5_path
        # Hyperparameters
        self.config['lr_type'] = self.args.lr_type
        self.config['min_lr'] = self.args.min_lr
        self.config['max_lr'] = self.args.max_lr
        self.config['clr_method'] = self.args.clr_method
        self.config['step_size'] = self.args.step_size
        self.config['batch_size'] = self.args.batch_size
        self.config['n_epochs'] = self.args.n_epochs
        self.config['random_seed'] = self.args.random_seed
        self.config['optimizer'] = self.args.optimizer
        self.config['network_name'] = self.args.network_name
        self.config['factor'] = self.args.factor
        self.config['patience'] = self.args.patience
        # Input images related
        self.config['image_height'] = self.args.image_height
        self.config['image_width'] = self.args.image_width
        #self.config['crop_image_height'] = self.args.crop_image_height
        #self.config['crop_image_width'] = self.args.crop_image_width
        self.config['n_channels'] = self.args.n_channels
        self.config['augmentation_type'] = self.args.augmentation_type
        self.config['tta_steps'] = self.args.tta_steps

    def _load_customized_setting(self):
        """Load sepcial setting
        """
        self.config['num_classes'] = self.args.num_classes


    def get_config(self):
        """return config
        """
        return self.config
