'''
Compute curvature similarity between stimuli
'''
project_name = 'kornet'
import os
#get current working directory
cwd = os.getcwd()
git_dir = cwd.split(project_name)[0] + project_name
import sys
sys.path.append(git_dir)

import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import pickle


#noramlized curvature code directory
curv_dir = '/mnt/DataDrive3/vlad/git_repos/normalized_contour_curvature/code'

sys.path.append(curv_dir)
# import functions
from functions import *

from glob import glob as glob
import pdb
import pandas as pd

#set directory
stimuli_dir = f'{git_dir}/stim/test/completed_silhouette'

#load object labels
stim_classes = pd.read_csv(f'{git_dir}/stim/kornet_classes.csv')

#load stim files
all_images = glob(os.path.join(stimuli_dir, '*.png'))

num_images = len(all_images)
## ----- Calculate NCC For All Images ----- ##

# parameters
rho = 0.04
bg_threshold = 5
num_bins = 101

# make histogram bins
dKappa = 2/num_bins
kappaMin = -1
kappaMax = 1
binEdges = np.linspace(start=kappaMin, stop=kappaMax, num=num_bins+1)
bins = binEdges[0:-1] + dKappa/2

# prepare array to hold calculated histograms
histogram_list = np.zeros((num_images, num_bins))
img_names = []
for i in range(num_images):
    # get image
    image = processImage(all_images[i])
    img_name = os.path.split(all_images[i])[1]
    img_names.append(img_name)
    # calculate NCC
    kappa, kappaDist, bins, f = contourCurvature(image, rho, numBins=num_bins)

    # put in list
    histogram_list[i,:] = kappaDist/(dKappa * sum(kappaDist))

    if (np.mod(i, 10) == 0):
        print('Completed image %d/%d.' % (i, num_images))


#correlate histograms
correlation_matrix = np.corrcoef(histogram_list)

#convert correlation matrix to pandas dataframe with image names for index and columns
correlation_rdm = pd.DataFrame(correlation_matrix, index=img_names, columns=img_names)

#save correlation matrix
correlation_rdm.to_csv(f'{git_dir}/curvature_correlation_matrix.csv')

#melt into long format
df = correlation_rdm.reset_index().melt(id_vars='index')
df.columns = ['obj1', 'obj2', 'correlation']

#remove self-correlations
df = df[df.obj1 != df.obj2]


#zscore the correlation values
df['zscore'] = (df.correlation - df.correlation.mean()) / df.correlation.std()

#loop through object classes and add object name labels using the number
#create new columns for object names
df['obj1_name'] = np.nan
df['obj2_name'] = np.nan
for num, label in zip(stim_classes['num'], stim_classes['object']):
    #if obj1 = OBJ (num).png then obj1_name = label
    df.loc[df['obj1'] == f'OBJ ({num}).png', 'obj1_name'] = label
    #if obj2 = OBJ (num).png then obj2_name = label
    df.loc[df['obj2'] == f'OBJ ({num}).png', 'obj2_name'] = label

#save zscored correlations
df.to_csv(f'{git_dir}/curvature_correlation_table.csv')

