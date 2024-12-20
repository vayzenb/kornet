'''
Evaluate CLIP on all test images using text prompt
'''


project_name = 'kornet'
import os
#get current working directory
cwd = os.getcwd()
git_dir = cwd.split(project_name)[0] + project_name
import sys
sys.path.append(git_dir)
baby_dir = f'{cwd.split(project_name)[0]}multimodal-baby'
sys.path.insert(1, baby_dir)
import torch
import clip
from PIL import Image
import numpy as np

import pandas as pd
from multimodal.multimodal_lit import MultiModalLitModel


stim_dir = f'{git_dir}/stim/test'
results_dir = f'{git_dir}/results'
test_label = np.asanyarray([0, 1])

device = "cuda" if torch.cuda.is_available() else "cpu"
#model, preprocess = clip.load("ViT-B/32", device=device)

model, preprocess = MultiModalLitModel.load_model(model_name="cvcl")
model = model.to(device)
model.eval()

#load classes from csv
class_list = pd.read_csv(f'{git_dir}/stim/kornet_classes.csv')
#determine categories (animate, inanimate, etc)
categories = class_list['category'].unique()

conditions = ['Outline', 'Pert', 'IC']
cond_sufs = ['','_ripple','_IC']

summary_df = pd.DataFrame(columns = ['model','classifier','train_ims','condition', 'animacy','obj1', 'obj2','acc','prob1', 'prob2'])

for condition in conditions:
    cond_suf = cond_sufs[conditions.index(condition)]
    #loop through superordinate category (animate, inanimate, etc)
    for category in categories:
        print(condition, category, flush=True)
        
        class_list_cat = class_list[class_list['category'] == category]
        
        
        #load each object image and pass through clip with the label
        for target_name,obj_num in zip(class_list_cat['object'],class_list_cat['num']):
            file_num = str(obj_num).zfill(2)
            
            #load image
            image = preprocess(Image.open(f'{stim_dir}/{condition}/OBJ ({file_num}){cond_suf}.png')).unsqueeze(0).to(device)
            
            #loop through all other categories and compare to the current category
            for distract_name in class_list_cat['object']:
                if distract_name != target_name:
                    
                    #load text
                    text, texts_len = model.tokenize([target_name, distract_name])
                    text, texts_len = text.to(device), texts_len.to(device)
                    
                    #get similarity score
                    with torch.no_grad():
                        
                        
                        logits_per_image, logits_per_text = model(image, text,texts_len)
                        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                        
            
                    #check if the target category has a higher score than the distractor category
                    #if so, set acc to 1 (correct)
                    if probs[0,0] > probs[0,1]:
                        acc = 1
                    else:
                        acc = 0

                    #add results to summary dataframe using pd concat
                    summary_df = pd.concat([summary_df,pd.DataFrame({'model':'cvcl','classifier':'langauge','train_ims':0,'condition':condition, 'animacy':category,'obj1':target_name, 'obj2':distract_name,
                                                                     'acc':acc, 'prob1':probs[0,0], 'prob2': probs[0,1]},index=[0])],ignore_index=True)
                    
    #save results
    summary_df.to_csv(f'{results_dir}/models/cvcl_language_train0_test{condition}.csv',index=False)