'''
Extract acts for each model
'''
project_name = 'kornet'
import os
#get current working directory
cwd = os.getcwd()
git_dir = cwd.split(project_name)[0] + project_name
import sys
sys.path.append(git_dir)

import torch
import load_stim
from glob import glob as glob
import pdb
import numpy as np
import pandas as pd

from model_loader import load_model as load_model

print('libraries loaded...')


#stim_dir = f'{curr_dir}/stim/test'
#stim_dir = f'/user_data/vayzenbe/image_sets/kornet_images'
#stim_dir = '/mnt/DataDrive2/vlad/kornet/image_sets/kornet_images'
#train_set = 'imagenet_sketch'

#layer = ['avgpool','avgpool','ln',['decoder','avgpool']]



#check length of sys.argv
if len(sys.argv) < 3:
    print('no model architecture specified, using debug model')
    model_arch = 'SayCam'
    stim_dir = '/mnt/DataDrive3/vlad/kornet/image_sets/kornet_images/test'
else:
    model_arch = sys.argv[1]
    stim_dir = sys.argv[2]


model_name = model_arch



#load layers file
model_layers = pd.read_csv(f'{git_dir}/modelling/all_model_layers.csv')

#extract layers for current model
layers = model_layers.loc[(model_layers['model'] == model_name) & (model_layers['use'] == 1)]['layers'].values

stim_folder = glob(f'{stim_dir}/*')
#only keep folder with bicycle
#stim_folder = [x for x in stim_folder if 'bicycle' in x]

suf = ''




def extract_acts(model, image_dir, transform, layer_call):
    print('extracting features...')
    

    #set up hook to specified layer
    def _store_feats(layer, inp, output):
        """An ugly but effective way of accessing intermediate model features
        """
        #avgpool = nn.AdaptiveAvgPool2d(output_size=(1,768))
        #output = avgpool(output)
        

        output = output.cpu().numpy()
        #pdb.set_trace()
        
        
        #find which axis is same size as batch_n
        batch_axis = int(np.squeeze(np.where(np.array(output.shape) == batch_n)))



        #if batch_n is not the first axis, move it to the first axis
        if batch_axis != 0:
            output = np.moveaxis(output, batch_axis, 0)
        
        
        

        
        _model_feats.append(np.reshape(output, (len(output), -1)))

    try:
        m = model.module
    except:
        m = model
    #model_layer = getattr(getattr(m, layer), sublayer)
    model_layer = eval(layer_call)
    model_layer.register_forward_hook(_store_feats)



    #Iterate through each image and extract activations

    imNum = 0
    n=0

    

    
    test_dataset = load_stim.load_stim(image_dir, transform=transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=24, shuffle=False, num_workers = 4, pin_memory=True)
    


    with torch.no_grad():
        
        for data, label in testloader:
            #print(label)
            # move tensors to GPU if CUDA is available
            
            data= data.cuda()

            global batch_n
            batch_n = data.shape[0]
            
            
            _model_feats = []
            
            #pdb.set_trace()
            #for the last layer, just run the model
            #throws an error trying to make a hook on the last layer
            if last_layer == False: 
                if vis_lang:#if arch is clip
                    
                    model.encode_image(data)
                else:
                    model(data)
            else: 
                if vis_lang:
                    
                    _model_feats.append(model.encode_image(data).cpu().numpy())
                else:
                    _model_feats.append(model(data).cpu().numpy())
                
            #output = model(data)

            

            if len(_model_feats) > 1:
                out = np.vstack(_model_feats[-1])
            else:
                out = np.vstack(_model_feats)

            
            
            

            if n == 0:
                acts = out
                #label_list = label
            else:
                acts= np.append(acts, out,axis = 0)
                #label_list = np.append(label_list, label)
                
            
            n = n + 1

    return acts




#model = model.visual


for cat_dir in stim_folder:
    print(cat_dir)
    #reload model for each category to avoid memory issues
    model, transform, _ = load_model(model_arch)
    model = model.cuda()
    model.eval()
    


    if 'clip' in model_arch or 'cvcl' in model_arch:
        vis_lang  = True
    else:
        vis_lang = False
    
    #pdb.set_trace()
        
    

    
    cat_name = cat_dir.split('/')[-1]

    
    
    for layer in layers:
        if 'vit' in model_arch:
            model, transform, _ = load_model(model_arch)
            model = model.cuda()
            
        #for vislang models, check if layer is the last one
        if vis_lang == True and layer == layers[-1]:
            
            last_layer = True
        else:
            last_layer = False

        #convert layer into a getattribute call
        #e.g., getattr(getattr(getattr(model,'visual'), 'attnpool'),'c_proj')
        layer_split = layer.split('.')



        #construct the call for each module in the list
        for ln, l in enumerate(layer_split):
            #on first item
            if ln == 0:
                layer_call = f"getattr(model,\'{l}\')"
            else:
                layer_call = f"getattr({layer_call},\'{l}\')"



        print(model_arch, cat_name, layer)
        
        
        acts = extract_acts(model, cat_dir, transform, layer_call)
               

        
        
        np.save(f'{git_dir}/modelling/acts/{model_name}{suf}_{layer}_{cat_name}.npy', acts)
        #clear memory
        del acts
        
        #clear cache
        torch.cuda.empty_cache()


        #np.savetxt(f'{curr_dir}/modelling/acts/{model_type}_{cat_name}_labels.txt', label_list)

    
    #if model exists, delete it
    
    del model
    
    
