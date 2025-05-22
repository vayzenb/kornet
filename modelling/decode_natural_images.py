'''
Decode images used for training (using a cross-validated procedure)
'''



project_name = 'kornet'
import os
#get current working directory
cwd = os.getcwd()
git_dir = cwd.split(project_name)[0] + project_name
import sys
sys.path.append(git_dir)

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

import pdb
import scipy.stats as stats
from glob import glob as glob


#load args
model_arch = sys.argv[1] #which architecture to use
k_folds = int(sys.argv[2]) #how many folds to use for cross validation
cond = sys.argv[3] #which condition to test on
print(model_arch)

conds = ['Outline', 'Pert', 'IC']

classifiers = ['NB', 'KNN', 'logistic', 'NC','SVM', 'Ridge']



act_dir = f'{git_dir}/modelling/acts'
results_dir = f'{git_dir}/results'
test_label = np.asanyarray([0, 1])



#load classes from csv
class_list = pd.read_csv(f'{git_dir}/stim/kornet_classes.csv')
#determine categories (animate, inanimate, etc)
categories = class_list['category'].unique()

conditions = ['Outline', 'Pert', 'IC']

def classify(classifier, train_data, train_labels, test_data, test_labels):
    
    if classifier != 'prototype':
        #train and score SVM, Ridge, naive bayes, KNN, and logistic regressionclassifiers, nearest centroid 
        if classifier == 'SVM':
            clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        elif classifier == 'Ridge':
            clf = make_pipeline(StandardScaler(), RidgeClassifierCV())
        elif classifier == 'NB':
            clf = make_pipeline(StandardScaler(), GaussianNB())
        elif classifier == 'KNN':
            clf = make_pipeline(StandardScaler(), KNeighborsClassifier())
        elif classifier == 'logistic':
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        elif classifier == 'NC':
            clf = make_pipeline(StandardScaler(), NearestCentroid())
        
        clf.fit(train_data, train_labels)

        score = clf.score(test_data, test_labels)

    return score


summary_df = pd.DataFrame(columns = ['model','classifier','condition','acc'])

#loop through classes and concatenate acts
for cat_n, cat_name in enumerate(class_list['object']):
    acts = np.load(f'{act_dir}/{model_arch}_{cat_name}.npy')
    if cat_name == class_list['object'][0]:
        all_acts = acts
    else:
        all_acts = np.vstack((all_acts, acts))


    #create labels for all classes
    if cat_name == class_list['object'][0]:
        labels = np.zeros((1,acts.shape[0]))
    else:
        labels = np.hstack((labels, np.zeros((1,acts.shape[0]))+cat_n))
        
                    
        
labels = labels.flatten()
print("Starting classification")

for classifier in classifiers:
    #implement stratified shuffle split
    sss = StratifiedShuffleSplit(n_splits=k_folds, test_size=0.2, random_state=0)
    fold_acc = []
    n = 0
    for train_index, test_index in sss.split(all_acts, labels):
        train_acts, test_acts = all_acts[train_index], all_acts[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        #train and score classifier
        score = classify(classifier, train_acts, train_labels, test_acts, test_labels)
        
        #add score to summary_df
        curr_data = pd.Series([model_arch, classifier, cond, score], index = summary_df.columns)


        summary_df = pd.concat([summary_df, curr_data.to_frame().T],ignore_index=True)
        
        n += 1

    summary_df.to_csv(f'{results_dir}/natural_image_decoding/{model_arch}_decoding.csv', index = False)
    #print acc
    



            






        
