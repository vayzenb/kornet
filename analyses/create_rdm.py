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
from plotnine import *
import seaborn as sns
from glob import glob as glob
import pdb
import warnings
warnings.filterwarnings("ignore")


data_dir = f'{git_dir}/data'
results_dir = f'{git_dir}/results'
fig_dir = f'{git_dir}/figures'
model_dir = f'{git_dir}/modelling'

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


def concat_all_subs():

    print('concatenating all subjects')
    sub_summary = pd.DataFrame(columns = ['sub','age','age_group','cond','duration','speed','target','distractor','acc'])

    for sub in sub_info['sub']:

        #check if sub has exclude value
        exclude = sub_info[sub_info['sub']==sub]['exclude'].values[0]
        sub_cond = sub_info[sub_info['sub']==sub]['cond'].values[0] -1 #subtract 1 because conditions are 1 2 3
        sub_file = glob(f'{data_dir}/{conds[sub_cond]}/{sub}_*.csv')
        #check if subfile is not empty
        if len(sub_file) != 0 and exclude != 1:
            sub_file = sub_file[0]
        
            temp_summary = pd.DataFrame(columns = sub_summary.columns)
            #glob a file from cond with sub number
            
            
            #load sub data
            sub_data = pd.read_csv(sub_file)
            
            #remove practice trials
            sub_data = sub_data[sub_data['pracTrials.ran']!=1]
            

            #add relevant cols
            
            temp_summary['duration'] = sub_data['Duration']
            temp_summary['speed'] = np.nan
            temp_summary['target'] = sub_data['objLabel']
            temp_summary['distractor'] = sub_data['objDistract']
            temp_summary['acc'] = sub_data['resp.corr']

            #add age columns
            temp_summary['sub'] = sub
            temp_summary['cond'] = conds[sub_cond]
            temp_summary['age'] = sub_info[sub_info['sub']==sub]['age'].values[0]
            temp_summary['age_group'] =np.floor(sub_info[sub_info['sub']==sub]['age'].values[0]).astype(str)[:-2] + 'yrold'

            #add speed column
            for i in range(len(durations)):
                temp_summary.loc[temp_summary['duration']==durations[i],'speed'] = speed[i]

            #append to sub_summary
            sub_summary = pd.concat([sub_summary,temp_summary], ignore_index=True)
            

    #Save
    sub_summary.to_csv(f'{results_dir}/all_sub_data.csv', index=False)


#concat_all_subs()

def compute_sub_rdm():
    print('Computing RDMs for all subs')
    

    #load sub summary
    sub_summary = pd.read_csv(f'{results_dir}/all_sub_data.csv')
    #remove .3 duration
    sub_summary = sub_summary[sub_summary['duration']<.3]

    summary_cols = ['overall','complete_overall','perturbed_overall','deleted_overall',
                    'complete_3','complete_4','complete_5','perturbed_3','perturbed_4','perturbed_5','deleted_3','deleted_4','deleted_5',
                    'complete_slow','complete_fast','perturbed_slow','perturbed_fast','deleted_slow','deleted_fast',
                    'complete_slow_3','complete_slow_4','complete_slow_5','complete_fast_3','complete_fast_4','complete_fast_5',
                    'perturbed_slow_3','perturbed_slow_4','perturbed_slow_5','perturbed_fast_3','perturbed_fast_4','perturbed_fast_5',
                    'deleted_slow_3','deleted_slow_4','deleted_slow_5','deleted_fast_3','deleted_fast_4','deleted_fast_5']

    rdm_summary = pd.DataFrame(columns = ['obj1','obj2'] + summary_cols)

    n = 0
    #loop through all stim pairs
    for stim1, cat1 in zip(stim_classes['object'], stim_classes['category']):
        for stim2, cat2 in zip(stim_classes['object'], stim_classes['category']):
            if stim1 != stim2 and cat1 == cat2:
                # set object for that row
                rdm_summary.loc[n,'obj1'] = stim1
                rdm_summary.loc[n,'obj2'] = stim2
                
                            
                #extract data for stim pair
                stim_pair = sub_summary[(sub_summary['target']==stim1) & (sub_summary['distractor']==stim2) | (sub_summary['target']==stim2) & (sub_summary['distractor']==stim1)]
                
                
                #extract each summary col
                for col in summary_cols:
                    temp_data = stim_pair
                    #extract relevant data
                    #check if col has has a dimension and then filter the data by that dimension
                    if 'overall' in col:
                        temp_data = temp_data
                    
                    if 'complete' in col:
                        temp_data = temp_data[temp_data['cond']=='complete']

                    if 'perturbed' in col:
                        temp_data = temp_data[temp_data['cond']=='perturbed']

                    if 'deleted' in col:
                        temp_data = temp_data[temp_data['cond']=='deleted']

                    if 'slow' in col:
                        temp_data = temp_data[temp_data['speed']=='slow']

                    if 'fast' in col:
                        temp_data = temp_data[temp_data['speed']=='fast']

                    if '3' in col:
                        temp_data = temp_data[temp_data['age_group']=='3yrold']

                    if '4' in col:
                        temp_data = temp_data[temp_data['age_group']=='4yrold']

                    if '5' in col:
                        temp_data = temp_data[temp_data['age_group']=='5yrold']

                    
                    #calculate mean accuracy
                    rdm_summary.loc[n,col] = temp_data['acc'].mean()

                n += 1


    #save
    rdm_summary.to_csv(f'{results_dir}/rdms/rdm_summary.csv', index=False)
                



            




def compute_model_rdm():

    '''
    Create model RDMs by calculating correlaiton between model activations for each object pair
    '''

    model_arch = ['vonenet_ff_ecoset','vonenet_ff_stylized-ecoset','vonenet_r_ecoset','vonenet_r_stylized-ecoset', 'ShapeNet','SayCam','convnext', 'vit']

    conds = ['Outline','Pert','IC']
    cond_names = ['complete','perturbed','deleted']
    #load rdm_summary
    rdm_summary = pd.read_csv(f'{results_dir}/rdms/rdm_summary.csv')

    for model in model_arch:
        #loop through objects and calculate correlation between model activations

        #load kornet objects 
        for cond in conds:
            #load objects act for that cat
            acts = np.load(f'{model_dir}/acts/{model}_{cond}.npy')
            

            
            model_corrs = []
            for obj1, obj2 in zip(rdm_summary['obj1'], rdm_summary['obj2']):
                #look up index from stim_classes for  obj1 and obj2
                obj1_idx = stim_classes[stim_classes['object']==obj1].index[0]
                acts1 = acts[obj1_idx,:]
                obj2_idx = stim_classes[stim_classes['object']==obj2].index[0]
                acts2 = acts[obj2_idx,:]


                

                #calculate correlation
                corr = np.corrcoef(acts1, acts2)[0,1]*-1
                #pdb.set_trace()

                model_corrs.append(corr)

            rdm_summary[f'{model}_{cond_names[conds.index(cond)]}'] = model_corrs

    rdm_summary.to_csv(f'{results_dir}/rdms/rdm_summary.csv', index=False)


#concat_all_subs()
#compute_sub_rdm()
compute_model_rdm()