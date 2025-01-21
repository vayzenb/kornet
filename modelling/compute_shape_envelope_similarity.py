'''Compute size similarity'''
import numpy as np
import pandas as pd
project_name = 'kornet'
import os
#get current working directory
cwd = os.getcwd()
git_dir = cwd.split(project_name)[0] + project_name
import sys
sys.path.append(git_dir)

import numpy as np
from glob import glob as glob
import pdb

#load object labels
stim_classes = pd.read_csv(f'{git_dir}/stim/kornet_classes.csv')

#load size data
size_data = pd.read_csv(f'{git_dir}/results/stim_size.csv')

size_table = pd.DataFrame(columns=['obj1','obj2','conv_hull','ratio_diff'])

#%% loop code

#loop through stim images and compute size similarity via subtraction
for i in range(len(size_data)):
    obj1 = size_data.loc[i,'object']
    convhull1 = size_data.loc[i,'conv_hull']
    ratio1 = size_data.loc[i,'axis_ratio']
    for j in range(i+1,len(size_data)):
        obj2 = size_data.loc[j,'object']
        convhull2 = size_data.loc[j,'conv_hull']
        ratio2 = size_data.loc[j,'axis_ratio']

        convhull_diff = np.abs(convhull1 - convhull2)
        ratio_diff = np.abs(ratio1 - ratio2)

        

        #add to dataframe using concat
        size_table = pd.concat([size_table, pd.DataFrame([[obj1,obj2,convhull_diff,ratio_diff]], columns = size_table.columns)])

#compute z-scores for each column and add to table
size_table['conv_hull_z'] = (size_table['conv_hull'] - size_table['conv_hull'].mean())/size_table['conv_hull'].std()
size_table['ratio_diff_z'] = (size_table['ratio_diff'] - size_table['ratio_diff'].mean())/size_table['ratio_diff'].std()

#save table
size_table.to_csv(f'{git_dir}/results/envelope_similarity.csv',index=False)

