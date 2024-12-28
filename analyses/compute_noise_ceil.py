'''
Compute noise ceilings for child data
'''
#%%

project_name = 'kornet'
import os
#get current working directory
cwd = os.getcwd()
git_dir = cwd.split(project_name)[0] + project_name
import sys
sys.path.append(git_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob as glob
import pdb
import warnings
warnings.filterwarnings("ignore")


data_dir = f'{git_dir}/data'
results_dir = f'{git_dir}/results'
fig_dir = f'{git_dir}/figures'
model_dir = f'{git_dir}/modelling'

iter = 1000

sub_info = pd.read_csv(f'{data_dir}/sub_info.csv' )
#remove subjects with NaNs in code field
#sub_info = sub_info[~sub_info['code'].isna()]
#remove subjects with 1 in exclude field
sub_info = sub_info[sub_info['exclude'] != 1]


conds = ['complete', 'perturbed','deleted']
model_conds = ['Outline','Pert','IC']
durations = [.3,.25, .2,.15,.1]
speed = ['intro','slow','slow','fast','fast']

age_groups = ['3yrold','4yrold','5yrold']

stim_classes = pd.read_csv(f'{git_dir}/stim/kornet_classes.csv')

#load all sub data
sub_df = pd.read_csv(f'{results_dir}/all_sub_data.csv')

#extract duration data below .3
#sub_df = sub_df[sub_df['duration'] < .3]
conds = ['complete', 'perturbed','deleted']

def split_half():
    '''computes noise ceiling by splitting subjects into two groups and correlating their accuracies for each condition'''

    for cond in conds:
        #extract sub data for each condition
        curr_df = sub_df[sub_df['cond'] == cond]

        boot_corr = []
        for i in range(0, iter):


            #split data into two equal groups by sub

            sub_list = curr_df['sub'].unique()
            sub_list = np.random.permutation(sub_list)
            #sub_list = sub_list[~np.isnan(sub_list)]

            #create one group of subjects with first half of subjects
            sub_group1 = sub_list[:int(len(sub_list)/2)]
            #create one group of subjects with second half of subjects
            sub_group2 = sub_list[int(len(sub_list)/2):]

            #extract data for each group
            curr_df1 = curr_df[curr_df['sub'].isin(sub_group1)]
            curr_df2 = curr_df[curr_df['sub'].isin(sub_group2)]

            group_acc1 = []
            group_acc2 = []
            n = 0
            #loop through all stim pairs
            for stim_n, (stim1, cat1) in enumerate(zip(stim_classes['object'], stim_classes['category'])):
                for stim2, cat2 in zip(stim_classes['object'][stim_n+1:], stim_classes['category'][stim_n+1:]):
                    if stim1 != stim2 and cat1 == cat2:

                        #extract data for stim pair
                        stim_df1 = curr_df1[(curr_df1['target'] == stim1) & (curr_df1['distractor'] == stim2) | (curr_df1['target'] == stim2) & (curr_df1['distractor'] == stim1)]
                        stim_df2 = curr_df2[(curr_df2['target'] == stim1) & (curr_df2['distractor'] == stim2) | (curr_df2['target'] == stim2) & (curr_df2['distractor'] == stim1)]

                        #compute accuracy for each group
                        acc1 = stim_df1['acc'].mean()
                        acc2 = stim_df2['acc'].mean()

                        group_acc1.append(acc1)
                        group_acc2.append(acc2)


            #add to temp_df
            temp_df = pd.DataFrame({'group1':group_acc1,'group2':group_acc2})

            #remove rows with NaNs in group1 or group2
            temp_df = temp_df[~temp_df['group1'].isna()]
            temp_df = temp_df[~temp_df['group2'].isna()]

            #correlate the two groups
            corr = np.corrcoef(temp_df['group1'],temp_df['group2'])[0,1]

            boot_corr.append(corr)


        #compute mean and 95% CIs of bootstrapped correlations
        mean_corr = np.mean(boot_corr)
        ci = np.percentile(boot_corr,[2.5,97.5])

        print(f'{cond} mean corr: {mean_corr}, 95% CI: {ci}')
