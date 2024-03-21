import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifierCV
import pdb
import scipy.stats as stats
from glob import glob as glob


suf = '_imagenet_sketch'
curr_dir = '/user_data/vayzenbe/GitHub_Repos/kornet'
act_dir = f'{curr_dir}/modelling/acts'
results_dir = f'{curr_dir}/results'
test_label = np.asanyarray([0, 1])

#load classes from csv
class_list = pd.read_csv(f'{curr_dir}/stim/kornet_classes.csv')
#determine categories (animate, inanimate, etc)
categories = class_list['category'].unique()

conditions = ['Outline', 'Pert', 'IC']

model_archs = ['vonecornet_s','cornet_s','voneresnet', 'vit','convnext','resnet50','resnext50','alexnet','vgg19', 'ShapeNet','SayCam']
model_archs = ['vonenet_r_ecoset','vonenet_r_stylized-ecoset','vonenet_ff_ecoset','vonenet_ff_stylized-ecoset']
model_archs = ['vonenet_r_ecoset','vonenet_r_stylized-ecoset','vonenet_ff_ecoset','vonenet_ff_stylized-ecoset', 'ShapeNet','SayCam', 'convnext']

print(model_archs)
k_folds = 15
train_n = 50



for model in model_archs:
    model= model + suf

    for cond in conditions:

        summary_df = pd.DataFrame(columns = ['model','obj1', 'obj2','condition','train_acc','acc'])
        #load acts
        test_acts = np.load(f'{act_dir}/{model}_{cond}.npy')

        #loop through superordinate category (animate, inanimate, etc)
        for category in categories:
            print(model, cond, category)
            class_list_cat = class_list[class_list['category'] == category]
            
            
            #load first training set
            for cat_name1 in class_list_cat['object']:
                
                #load training acts
                train_acts1 = np.load(f'{act_dir}/{model}_{cat_name1}.npy')
                

                #create empty test array
                test_ims = np.zeros((2, test_acts.shape[1]))
                
                #determine image num for test object
                img_num = class_list_cat[class_list_cat['object']==cat_name1].index[0]

                #read test act for that num
                test_ims[0,:] = test_acts[img_num,:]

                

                #load second training set
                for cat_name2 in class_list_cat['object']:
                    if cat_name1 == cat_name2:
                        continue
                    else:
                        #load second training acts
                        train_acts2 = np.load(f'{act_dir}/{model}_{cat_name2}.npy')

                        fold_acc = []
                        fold_train_acc = []
                        for k in range(0,k_folds):
                            


                            #shuffle training acts
                            np.random.shuffle(train_acts1)
                            np.random.shuffle(train_acts2)


                            train_acts = np.vstack((train_acts1[0:train_n,:], train_acts2[0:train_n,:]))
                            clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
                            #clf = RidgeClassifierCV()
                            
                            #create empty train array for labels
                            label_list = np.zeros((1,train_n))
                            label_list = np.hstack((label_list, np.zeros((1,train_n))+1))
                            label_list = label_list.flatten()
                            
                            #determine image num for test object
                            img_num2 = class_list_cat[class_list_cat['object']==cat_name2].index[0]
                            

                            #read test act for that num
                            test_ims[1,:] = test_acts[img_num2,:]

                            

                            #train classifier
                            clf.fit(train_acts, label_list)

                            #test classifier
                            score = clf.score(test_ims, test_label)
                            train_score = clf.score(train_acts, label_list)

                            fold_acc.append(score)
                            fold_train_acc.append(train_score)

                            

                        #add results to summary
                        avg_test = np.mean(fold_acc)
                        avg_train = np.mean(fold_train_acc)
                        curr_data = pd.Series([model, cat_name1, cat_name2, cond, avg_train, avg_test], index = summary_df.columns)
                        summary_df = pd.concat([summary_df, curr_data.to_frame().T],ignore_index=True)

        summary_df.to_csv(f'{results_dir}/models/{model}_{cond}_summary.csv', index = False)
        
                    






                
