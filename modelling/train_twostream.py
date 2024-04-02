"""
Train models on ecoset with developmental constraints
"""

curr_dir = '/user_data/vayzenbe/GitHub_Repos/kornet'
import sys

import os, argparse, shutil
from collections import OrderedDict

import torch

import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision import datasets
import numpy as np
import torchvision.transforms.functional as TF
import warnings
warnings.filterwarnings("ignore")

import pdb
import model_funcs

import random
from glob import glob as glob
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from model_loader import load_model as load_model
import two_stream_dataloader
import two_stream_nn

now = datetime.now()
curr_date=now.strftime("%Y%m%d")

print('libs loaded')

#Example call:
#python modelling/train.py --data /user_data/vayzenbe/image_sets/ecoset -o /lab_data/behrmannlab/vlad/kornet/modelling/weights/ --arch cornet_s -b 128 --blur True
#python modelling/train.py --data /lab_data/behrmannlab/image_sets/stylized-ecoset -o /lab_data/behrmannlab/vlad/kornet/modelling/weights/ --arch cornet_ff --blur True
#python modelling/train_twostream.py --data /user_data/vayzenbe/image_sets/development_images -o /lab_data/behrmannlab/vlad/kornet/modelling/weights/ --epochs 10

parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('--data', required=False,
                    help='path to folder that contains train and val folders', 
                    default=None)
parser.add_argument('-o', '--output_path', default=None,
                    help='path for storing ')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--workers', default=4, type=int,
                    help='how many data loading workers to use')
parser.add_argument('--rand_seed', default=1, type=int,
                    help='Seed to use for reproducible results')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

suf='_parallel'


out_dir = '/lab_data/behrmannlab/vlad/kornet/modelling/weights'


global args, best_prec1
args = parser.parse_args()

image_type = args.data
image_type=image_type.split('/')[-1]
model_type = f'two_stream_nn{suf}'

print(image_type)

model_funcs.reproducible_results(args.rand_seed)

#These are all the default learning parameters from the vonenet paper
start_epoch = 1
lr = .1 #Starting learning rate
step_size = 10 #How often (epochs)the learning rate should decrease by a factor of 10
weight_decay = 1e-4
momentum = .9
n_epochs = args.epochs
n_save = 5 #save model every X epochs





print(model_type)
writer = SummaryWriter(f'{curr_dir}/modelling/runs/{model_type}')
best_prec1 = 0

def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar'):
    torch.save(state, f'{out_dir}/{filename}_checkpoint_{args.rand_seed}.pth.tar')
    if (epoch) == 1 or (epoch) % n_save  == 0:
        shutil.copyfile(f'{out_dir}/{filename}_checkpoint_{args.rand_seed}.pth.tar', f'{out_dir}/{filename}_{epoch}_{args.rand_seed}.pth.tar')
    if is_best:
        shutil.copyfile(f'{out_dir}/{filename}_checkpoint_{args.rand_seed}.pth.tar', f'{out_dir}/{filename}_best_{args.rand_seed}.pth.tar')

#Image directory

model = two_stream_nn.TwoStream()
#model = model.cuda()
model = torch.nn.DataParallel(model).cuda()

'''
load model
'''


optimizer = torch.optim.SGD(model.parameters(),
                                         lr,
                                         momentum=momentum,
                                         weight_decay=weight_decay)


#lr updated given some rule
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)


criterion = nn.CrossEntropyLoss()
criterion.cuda()


# optionally resume from a checkpoint
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        
        
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

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


train_dir = args.data + '/train'
val_dir = args.data + '/val'

#Set up data loaders
train_dataset = two_stream_dataloader.ImageFolderDataset(train_dir, transform_ventral=transform_ventral, transform_dorsal=transform_dorsal)
val_dataset = two_stream_dataloader.ImageFolderDataset(val_dir, transform_ventral=transform_ventral, transform_dorsal=transform_dorsal)


valloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers =  args.workers, pin_memory=True, drop_last=True)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.workers, pin_memory=True, drop_last=True)
        

print('starting training...')
valid_loss_min = np.Inf # track change in validation loss
nTrain = 1
nVal = 1


for epoch in range(start_epoch, n_epochs+1):
   
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    
    ###################
    # train the model #
    ###################
    model.train()
    n = 0
    
    for ventral_data,dorsal_data, target in trainloader:
        
        n = n + 1
        #data = TF.adjust_saturation(data, saturate)

        # move tensors to GPU if CUDA is available       
        ventral_data, dorsal_data, target = ventral_data.cuda(), dorsal_data.cuda(), target.cuda()
            
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(ventral_data, dorsal_data)
        
        
        #print(output.shape)
        # calculate the batch loss
        loss = criterion(output, target)
        #print(loss)
        writer.add_scalar("Supervised Raw Train Loss", loss, nTrain) #write to tensorboard
        writer.flush()
        nTrain = nTrain + 1
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        
        # update training loss
        train_loss += loss.item()*ventral_data.size(0)


    
        #print(train_loss)
    

    scheduler.step()
    ######################    
    # validate the model #
    ######################
    model.eval()
    accuracy = 0
    print('starting eval')
    with torch.no_grad():
        for ventral_data, dorsal_data, target in valloader:
            # move tensors to GPU if CUDA is available
            
            ventral_data,dorsal_data, target = ventral_data.cuda(), dorsal_data.cuda(), target.cuda()

            
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(ventral_data, dorsal_data)
            
            # calculate the batch loss
            loss = criterion(output, target)

            #writer.add_scalar("Supervised Raw Validation Loss", loss, nVal) #write to tensorboard
            #writer.flush()
            nVal = nVal + 1
            #print('wrote to tensorboard')
            # update average validation loss 
            valid_loss += loss.item()*ventral_data.size(0)

            topP, topClass = output.topk(1, dim=1) #get top 1 response
            equals = topClass == target.view(*topClass.shape) #check how many are right
            accuracy += torch.mean(equals.type(torch.FloatTensor)) #calculate acc; equals needed to made into a flaot first
            
            

            

    
    # calculate average losses
    train_loss = train_loss/len(trainloader.sampler)
    valid_loss = valid_loss/len(valloader.sampler)

    prec1 = accuracy/len(valloader)
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)

    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss),
        "Test Accuracy: {:.3f}".format(accuracy/len(valloader)))
    writer.add_scalar("Average Train Loss", train_loss, epoch) #write to tensorboard
    writer.add_scalar("Average Validation Loss", valid_loss, epoch) #write to tensorboard
    writer.add_scalar("Average Acc", accuracy/len(valloader), epoch) #write to tensorboard
    writer.flush()
    
    # save model if validation loss has decreased
    save_checkpoint({
                'epoch': epoch,
                'arch': model_type,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best,epoch,filename=f'{model_type}')


writer.close()
        

