project_name = 'kornet'
import os
#get current working directory
cwd = os.getcwd()
git_dir = cwd.split(project_name)[0] + project_name
import sys
sys.path.append(git_dir)

import pandas as pd
import numpy as np
from sklearnex import patch_sklearn 
patch_sklearn()

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.feature_selection import VarianceThreshold

#import pca
from sklearn.decomposition import PCA
from sklearn import cluster

import pdb
import scipy.stats as stats
from glob import glob as glob
import time
import resource
from joblib import Parallel, delayed



#load args
model_arch = sys.argv[1] #which architecture to use
train_n = int(sys.argv[2]) #how many images to use t otrain the classifier
classifier = sys.argv[3] #which classifier to use
k_folds = int(sys.argv[4]) #how many folds to use for cross validation
cond = sys.argv[5] #which condition to test on
print(model_arch)



act_dir = f'{git_dir}/modelling/acts'
results_dir = f'{git_dir}/results'
test_label = np.asanyarray([0, 1])

#load layers file
model_layers = pd.read_csv(f'{git_dir}/modelling/all_model_layers.csv')

#extract layers for current model
layers = model_layers.loc[(model_layers['model'] == model_arch) & (model_layers['use'] == 1)]['layers'].values


#load classes from csv
class_list = pd.read_csv(f'{git_dir}/stim/kornet_classes.csv')
#determine categories (animate, inanimate, etc)
categories = class_list['category'].unique()

conditions = ['Outline', 'Pert', 'IC']

sel = VarianceThreshold()

def classify(classifier, train_data, train_labels, test_data, test_labels):
    

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
        clf = make_pipeline(StandardScaler(), LogisticRegression())
    elif classifier == 'NC':
        clf = make_pipeline(StandardScaler(), NearestCentroid())
    
    clf.fit(train_data, train_labels)

    score = clf.score(test_data, test_labels)

    del clf

    return score

def prep_for_classification(train_acts1, train_acts2, test_ims, train_n,classifier, test_label):

    train_acts1_shuffled = np.copy(train_acts1)
    train_acts2_shuffled = np.copy(train_acts2)

    #shuffle training acts
    np.random.shuffle(train_acts1_shuffled)
    np.random.shuffle(train_acts2_shuffled)


    train_acts = np.vstack((train_acts1_shuffled[0:train_n,:], train_acts2_shuffled[0:train_n,:]))                        

    
    #clf = RidgeClassifierCV()
    
    #create empty train array for labels
    label_list = np.zeros((1,train_n))
    label_list = np.hstack((label_list, np.zeros((1,train_n))+1))
    label_list = label_list.flatten()
    
                                  

    #evaluate fold
    score = classify(classifier, train_acts, label_list, test_ims, test_label)



    #clear memory
    del train_acts
    del label_list

    return score


summary_df = pd.DataFrame(columns = ['model','layer','classifier','train_ims','condition', 'animacy','obj1', 'obj2','acc'])
#load acts

layers = ['model.decoder.avgpool']
for layer in layers:
    test_acts = np.load(f'{act_dir}/{model_arch}_{layer}_{cond}.npy')

    #loop through superordinate category (animate, inanimate, etc)
    for category in categories:
        print(model_arch, layer, cond, category, flush=True)
        class_list_cat = class_list[class_list['category'] == category]
        class_list_cat = class_list_cat.reset_index(drop=True)
        
        
        #load first training set
        for cat1n, cat_name1 in enumerate(class_list_cat['object']):
            
            #load training acts
            train_acts1 = np.load(f'{act_dir}/{model_arch}_{layer}_{cat_name1}.npy')
            

            #create empty test array
            test_ims = np.zeros((2, test_acts.shape[1]))
            
            #determine image num for test object
            img_num1 = class_list_cat[class_list_cat['object']==cat_name1].index[0]

            

            
            
            #timeit
            
            #load second training set
            for cat_name2 in class_list_cat['object'][cat1n+1:]:
                start = time.time()
                if cat_name1 == cat_name2:
                    continue
                else:
                    
                    #load second training acts
                    train_acts2 = np.load(f'{act_dir}/{model_arch}_{layer}_{cat_name2}.npy')

                    
                    #determine image num for test object
                    img_num2 = class_list_cat[class_list_cat['object']==cat_name2].index[0]

                    #read test act for that num
                    test_ims[0,:] = test_acts[img_num1,:]
                    test_ims[1,:] = test_acts[img_num2,:]

                    fold_acc = []
                    fold_train_acc = []
                    
                    #run prep_for_classification in parallel and append to fold_acc
                    fold_acc = Parallel(n_jobs=k_folds)(delayed(prep_for_classification)(train_acts1, train_acts2, test_ims, train_n,classifier, test_label) for i in range(0,k_folds))
                        


                        

                        
                    #print memory usage in gb
                    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024, flush=True)
                    
                    end = time.time()
                    print(f'{cat_name1} vs {cat_name2} took {end - start} seconds', flush=True)

                    #add results to summary
                    avg_test = np.mean(fold_acc)
                    # append 'model','classifier','train_ims','test condition', 'animacy category','obj1', 'obj2','acc'
                    curr_data = pd.Series([model_arch, layer, classifier,train_n, cond, category, cat_name1, cat_name2, avg_test], index = summary_df.columns)
                    summary_df = pd.concat([summary_df, curr_data.to_frame().T],ignore_index=True)

    summary_df.to_csv(f'{results_dir}/models/{model_arch}_{classifier}_train{train_n}_test{cond}_layerwise.csv', index = False)

            






        
