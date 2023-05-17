'''
Extract acts for each model
'''
import pandas as pd
import numpy as np
import sys
vone_dir = '/user_data/vayzenbe/GitHub_Repos/vonenet'
cornet_dir = '/user_data/vayzenbe/GitHub_Repos/CORnet'

sys.path.insert(1, vone_dir)
sys.path.insert(1, cornet_dir)
import vonenet
import cornet
from torchvision.models import convnext_large, ConvNeXt_Large_Weights, vit_b_16, ViT_B_16_Weights
from torchvision.models import resnet50, ResNet50_Weights, resnext50_32x4d, ResNeXt50_32x4d_Weights
from torchvision.models import alexnet, AlexNet_Weights, vgg19, VGG19_Weights
import torch

import torch.nn as nn
import torchvision
import load_stim
from glob import glob as glob
import pdb

print('libraries loaded...')

curr_dir = '/user_data/vayzenbe/GitHub_Repos/kornet'
stim_dir = f'{curr_dir}/stim/test'
#stim_dir = f'/user_data/vayzenbe/image_sets/kornet_images'
weights_dir = '/lab_data/behrmannlab/vlad/kornet/modelling/weights'
train_set = 'imagenet_sketch'

#layer = ['avgpool','avgpool','ln',['decoder','avgpool']]

model_arch = sys.argv[1]

stim_folder = glob(f'{stim_dir}/*')
suf = ''

def load_model(model_arch):    
    """
    load model
    """
    if model_arch == 'cornets':
        model = vonenet.get_model(model_arch='cornets', pretrained=True).module
        layer_call = "getattr(getattr(getattr(getattr(model,'module'),'model'),'decoder'),'avgpool')"
        transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])])

    elif model_arch == 'cornet_s':
        model = cornet.get_model('s', pretrained=True).module
        layer_call = "getattr(getattr(getattr(model,'module'),'decoder'),'avgpool')"

        transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])

    elif model_arch == 'convnext':
        model = convnext_large(weights=ConvNeXt_Large_Weights.DEFAULT)
        transform = ConvNeXt_Large_Weights.IMAGENET1K_V1.transforms()
        layer_call = "getattr(getattr(model,'module'),'avgpool')"

    elif model_arch == 'vit':
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        transform = ViT_B_16_Weights.DEFAULT.transforms()
        layer_call = "getattr(getattr(getattr(model,'module'),'encoder'),'ln')"
        #layer_call = "getattr(getattr(getattr(getattr(getattr(getattr(model,'module'),'encoder'),'layers'),'encoder_layer_11'),'mlp'),'3')"

    elif model_arch == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        transform = ResNet50_Weights.DEFAULT.transforms()
        layer_call = "getattr(getattr(getattr(model,'module'),'decoder'),'avgpool')"
    
    elif model_arch == 'resnext50':
        model = resnext50_32x4d(weights=ResNeXt50_32x4d_Weights.DEFAULT)
        transform = ResNeXt50_32x4d_Weights.DEFAULT.transforms()
        layer_call = "getattr(getattr(getattr(model,'module'),'decoder'),'avgpool')"
    
    elif model_arch == 'alexnet':
        model = alexnet(weights=AlexNet_Weights.DEFAULT)
        transform = AlexNet_Weights.DEFAULT.transforms()

    elif model_arch == 'vgg19':
        model = vgg19(weights=VGG19_Weights.DEFAULT)
        transform = VGG19_Weights.DEFAULT.transforms()

    elif model_arch == 'ShapeNet':
        model = resnet50(weights=None)
        checkpoint = torch.load('Weights/ShapeNet_ResNet50_Weights.pth.tar')
        model.load_state_dict(checkpoint)
        transform = ResNet50_Weights.DEFAULT.transforms()
        layer_call = "getattr(getattr(getattr(model,'module'),'decoder'),'avgpool')"

    elif model_arch == 'SayCam':
        model = resnext50_32x4d(weights=None)
        #model = torch.nn.DataParallel(model)
        #model.fc = torch.nn.Linear(in_features=2048, out_features=n_out, bias=True)
        checkpoint = torch.load('Weights/SayCam_ResNext_Weights.pth.tar')
        model.load_state_dict(checkpoint)

        

    model = torch.nn.DataParallel(model).cuda()

    
    #checkpoint = torch.load(f'{weights_dir}/{model_arch}_{train_set}_best_1.pth.tar')
    #model.load_state_dict(checkpoint['state_dict'])

    return model, transform, layer_call




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
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers = 4, pin_memory=True)
    


    with torch.no_grad():
        
        for data, _ in testloader:
            # move tensors to GPU if CUDA is available
            
            data= data.cuda()
            
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

for cat_dir in stim_folder:
    cat_name = cat_dir.split('/')[-1]
    print(model_arch, cat_name)
    acts = extract_acts(model, cat_dir, transform, layer_call)

    
    
    np.save(f'{curr_dir}/modelling/acts/{model_arch}{suf}_{cat_name}.npy', acts)
    #np.savetxt(f'{curr_dir}/modelling/acts/{model_type}_{cat_name}_labels.txt', label_list)
    
