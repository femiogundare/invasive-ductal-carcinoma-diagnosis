# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 00:27:33 2021

@author: femiogundare
"""


import os
import json
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from utilities.others import plot_confusion_matrix
from config import idc_config as config


# Initialize the Config class in the config script
configs = config.Config()
# Put all the arguments of the argparse in a dictionary by calling the get_config method of the Config class
configs_dict = configs.get_config()

OUTPUT_DIR = configs_dict['output_dir']
RESULT_DIR = configs_dict['result_dir']
PREDICTIONS_DIR = configs_dict['predictions_dir']


classes_names = ['Non-IDC', 'IDC']

# Check the files in the predictions directory
print(os.listdir(PREDICTIONS_DIR))

network_names = ['EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'ResNet50']

# Read the csv file of the predictions made by the neural networks and put them in a list
combined_predictions = [
    pd.read_csv(PREDICTIONS_DIR + '/'+ pred_file) for pred_file in os.listdir(PREDICTIONS_DIR)
]


x = np.zeros((len(combined_predictions[0]), len(os.listdir(PREDICTIONS_DIR))))

for k in range(len(os.listdir(PREDICTIONS_DIR))):
    x[:, k] = combined_predictions[k].prediction.values

target = combined_predictions[0].label.values


### ENSEMBLE==========
# Compute the average of the predictions of the networks
avg_preds = (x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3])/4

# Compute the AUC of the ensemble
ensemble_auc_score = roc_auc_score(target, avg_preds)
print('AUC: {:.4f}'.format(ensemble_auc_score))

# Compute the Sensitivity, Specificity, PPV, NPV, and J statistic of the ensemble
pred_labels = [1 if pred>=0.5 else 0 for pred in avg_preds]
cnf_matrix = confusion_matrix(target, pred_labels)
tn, fn, tp, fp = cnf_matrix[0][0], cnf_matrix[1][0], cnf_matrix[1][1], cnf_matrix[0][1]
ensemble_sensitivity_score = tp/(tp+fn)
ensemble_specificity_score = tn/(tn+fp)
ensemble_ppv_score = tp/(tp+fp)
ensemble_npv_score = tn/(tn+fn)
ensemble_J_score = (ensemble_sensitivity_score + ensemble_specificity_score - 1)
print('Sensitivity: {:.4f}'.format(ensemble_sensitivity_score))
print('Specificity: {:.4f}'.format(ensemble_specificity_score))
print('Positive Predictive Value: {:.4f}'.format(ensemble_ppv_score))
print('Negative Predictive Value: {:.4f}'.format(ensemble_npv_score))
print("Youden's J statistic: {:.4f}".format(ensemble_J_score))
print('Confusion Matrix: \n{}'.format(cnf_matrix))

# Store the ensemble predictions to a csv file
print('Storing the predictions to csv...')
names_of_images_in_dataset = OUTPUT_DIR + '/names_of_images.json'
names = json.loads(open(names_of_images_in_dataset).read())
names_of_test_images = names['test_names']

df = pd.DataFrame(
    dict(
        name=names_of_test_images, 
        label=target, 
        prediction=avg_preds
        )
    )

df.to_csv(PREDICTIONS_DIR+ '/ensemble_avg_predictions.csv', index=False)



### COMPOSE A DATAFRAME FOR THE SCORES OF THE NEURAL NETWORKS AND ENSEMBLE
auc_scores = []
sensitivity_scores = []
specificity_scores = []
ppv_scores = []
npv_scores = []
youden_indices = []

for k in range(x.shape[1]):
    print('Computing scores for {}...'.format(network_names[k]))
    predictions = x[:, k]
    prediction_labels = [1 if pred>=0.5 else 0 for pred in predictions]
    pred_labels = [1 if pred>=0.5 else 0 for pred in predictions]
    cnf_matrix = confusion_matrix(target, prediction_labels)
    tn, fn, tp, fp = cnf_matrix[0][0], cnf_matrix[1][0], cnf_matrix[1][1], cnf_matrix[0][1]
    sensitivity, specificity = tp/(tp+fn), tn/(tn+fp)
    sensitivity_scores.append(round(sensitivity, 4))
    specificity_scores.append(round(specificity, 4))
    auc_scores.append(round(roc_auc_score(target, predictions), 4))
    ppv_scores.append(round(tp/(tp+fp), 4))
    npv_scores.append(round(tn/(tn+fn), 4))
    youden_indices.append(round(sensitivity+specificity-1, 4))
    #auc = roc_auc_score(target, x[:, k])
    
for k in range(x.shape[1]):
    print('{}: Sensitivity = {}, Specificty = {}, AUC = {}, PPV={}, NPV={}, Youden J Index = {}'.format(
    network_names[k], round(sensitivity_scores[k], 4), 
    round(specificity_scores[k], 4), round(auc_scores[k], 4),
    round(ppv_scores[k], 4), round(npv_scores[k], 4), round(youden_indices[k], 4)
         ))
    
# Store the results in a csv file
results = pd.DataFrame({
    'Neural Network' : ['EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'ResNet50', 'Ensemble (Average)'],
    'AUC (%)' : [auc_scores[0], auc_scores[1], auc_scores[2], auc_scores[3], round(ensemble_auc_score, 4)],
    'Sensitivity (%)' : [sensitivity_scores[0], sensitivity_scores[1], sensitivity_scores[2], 
                     sensitivity_scores[3], round(ensemble_sensitivity_score, 4)
                    ],
    'Specificity (%)' : [specificity_scores[0], specificity_scores[1], specificity_scores[2], 
                     specificity_scores[3], round(ensemble_specificity_score, 4)
                    ],
    'PPV (%)' : [ppv_scores[0], ppv_scores[1], ppv_scores[2], 
                     ppv_scores[3], round(ensemble_ppv_score, 4)
                    ],
    'NPV (%)' : [npv_scores[0], npv_scores[1], npv_scores[2], 
                     npv_scores[3], round(ensemble_npv_score, 4)
                    ],
    'J Statistic (%)' : [youden_indices[0], youden_indices[1], youden_indices[2], 
                     youden_indices[3], round(ensemble_J_score, 4)
                    ]
    
})


results.set_index('Neural Network', drop=True, inplace=True)
results = 100*results
results.to_csv(RESULT_DIR+'/scores.csv')

print(results)



# Plot the confusion matrix of the ensemble
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=classes_names,
                      title='Confusion matrix')
#plt.show()
plt.savefig(RESULT_DIR + '/ensemble/confusion_matrix.png')


# Plot the ROC Curve of the ensemble and save to png
plt.figure()
fpr, tpr, _ = roc_curve(target, avg_preds, pos_label=1)
plt.style.use('seaborn')
plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.4f)' % ensemble_auc_score)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.title('Receiving Operating Characteristic Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
#plt.show()
plt.savefig(RESULT_DIR + '/ensemble/roc_curve.png')