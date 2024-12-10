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

from model_loader import load_model as load_model

print('libraries loaded...')

torch.cuda.set_device(1)
print('gpu #', torch.cuda.current_device())
device = torch.device('cuda')
#device = torch.device('cpu')



#stim_dir = f'{curr_dir}/stim/test'
#stim_dir = f'/user_data/vayzenbe/image_sets/kornet_images'
#stim_dir = '/mnt/DataDrive2/vlad/kornet/image_sets/kornet_images'
#train_set = 'imagenet_sketch'

#layer = ['avgpool','avgpool','ln',['decoder','avgpool']]

#check length of sys.argv
if len(sys.argv) < 2:
    print('please specify model architecture')
    sys.exit()

model_arch = sys.argv[1]
model_name = model_arch

stim_dir = sys.argv[2]

'''
#specify weights file
if len(sys.argv) == 2:
    weights = None
    model_name = model_arch
elif len(sys.argv) == 3:
    weights = sys.argv[2]
    model_name = model_arch + '_' + weights
'''

    

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
            
            data= data.to(device)
            
            _model_feats = []
            model(data)
            #output = model(data)
            
            out = np.vstack(_model_feats)
            
            

            if n == 0:
                acts = out
                #label_list = label
            else:
                acts= np.append(acts, out,axis = 0)
                #label_list = np.append(label_list, label)
                
            
            n = n + 1

    return acts



model, transform, layer_call = load_model(model_arch)

model = model.to(device)

#check if model is on gpu or cpu
print(next(model.parameters()).is_cuda)

for cat_dir in stim_folder:
    print(cat_dir)
    #VIT runs out of memory quickly, so we delete and reload it after every iteration
    if 'vit' in model_arch:
        model, transform, layer_call = load_model(model_arch)
        model = model.to(device)
        

    
    cat_name = cat_dir.split('/')[-1]
    print(model_arch, cat_name)
    acts = extract_acts(model, cat_dir, transform, layer_call)

    

    
    
    np.save(f'{git_dir}/modelling/acts/{model_name}{suf}_{cat_name}.npy', acts)
    #clear memory
    del acts
    
    #clear cache
    torch.cuda.empty_cache()

    if model_arch == 'vit':
        del model
    #np.savetxt(f'{curr_dir}/modelling/acts/{model_type}_{cat_name}_labels.txt', label_list)
    
