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

import pandas as pd
import numpy as np

import torch

from glob import glob as glob
import pdb
import torchvision

#from model_loader import load_model as load_model
import modelling.two_stream.load_stim_twostream as load_stim_twostream
import modelling.two_stream.two_stream_nn as two_stream_nn

print('libraries loaded...')


#stim_dir = f'{curr_dir}/stim/test'
#stim_dir = f'/user_data/vayzenbe/image_sets/kornet_images'

#train_set = 'imagenet_sketch'

#layer = ['avgpool','avgpool','ln',['decoder','avgpool']]
suf = ''
model_arch = 'twostream_r'
model_name = model_arch

stim_dir = sys.argv[1]
weights_dir = '/mnt/DataDrive2/vlad/kornet/modelling/weights/'

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






def extract_acts(ventral_model, dorsal_model, cat_dir, transform_ventral,transform_dorsal, ventral_layer,dorsal_layer):
    print('extracting features...')
    

    #set up hook to specified layer
    def _store_feats(layer, inp, output):
        """An ugly but effective way of accessing intermediate model features
        """
        #avgpool = nn.AdaptiveAvgPool2d(output_size=(1,768))
        #output = avgpool(output)
        

        output = output.cpu().numpy()
        
        _model_feats.append(np.reshape(output, (len(output), -1)))



    #model_layer = getattr(getattr(m, layer), sublayer)
    ventral_model_layer = eval(ventral_layer)
    ventral_model_layer.register_forward_hook(_store_feats)

    dorsal_model_layer = eval(dorsal_layer)
    dorsal_model_layer.register_forward_hook(_store_feats)



    #Iterate through each image and extract activations

    imNum = 0
    n=0

    

    
    test_dataset = load_stim_twostream.load_stim(cat_dir, transform_ventral=transform_ventral, transform_dorsal=transform_dorsal)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=24, shuffle=False, num_workers = 4, pin_memory=True)
    


    with torch.no_grad():
        
        for ventral_data,dorsal_data, label in testloader:
            
            
            # move tensors to GPU if CUDA is available
            
            ventral_data,dorsal_data= ventral_data.cuda(),dorsal_data.cuda()
            
            _model_feats = []
            ventral_model(ventral_data,dorsal_data)
            #output = model(data)
            
            ventral_out = np.vstack(_model_feats)

            _model_feats = []
            dorsal_model(ventral_data,dorsal_data)
            dorsal_out = np.vstack(_model_feats)



            #concatenate ventral and dorsal activations
            out = np.hstack((ventral_out,dorsal_out))

            
            

            if n == 0:
                acts = out
                #label_list = label
            else:
                acts= np.append(acts, out,axis = 0)
                #label_list = np.append(label_list, label)
                
            
            n = n + 1

    return acts


ventral_model =  two_stream_nn.TwoStream('vonenet_r')
dorsal_model = two_stream_nn.TwoStream('vonenet_r')

ventral_model = ventral_model.cuda()
dorsal_model = dorsal_model.cuda()

checkpoint = torch.load(f'{weights_dir}/{model_name}_best_1.pth.tar')
ventral_model.load_state_dict(checkpoint['state_dict'])
dorsal_model.load_state_dict(checkpoint['state_dict'])

transform_ventral = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224,224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])])

transform_dorsal = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224,224)),
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.GaussianBlur(kernel_size=35, sigma=15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])

#Set up data loaders

ventral_layer = "getattr(getattr(getattr(ventral_model,'ventral'),'model'),'decoder')"
dorsal_layer = "getattr(getattr(dorsal_model,'dorsal'),'head')"

for cat_dir in stim_folder:
    print(cat_dir)

    
    cat_name = cat_dir.split('/')[-1]
    print(model_arch, cat_name)
    acts = extract_acts(ventral_model,dorsal_model, cat_dir, transform_ventral,transform_dorsal, ventral_layer,dorsal_layer)

    

    np.save(f'{git_dir}/modelling/acts/{model_name}{suf}_{cat_name}.npy', acts)
    #clear memory
    del acts
    
    #clear cache
    torch.cuda.empty_cache()

    #np.savetxt(f'{curr_dir}/modelling/acts/{model_type}_{cat_name}_labels.txt', label_list)
    
