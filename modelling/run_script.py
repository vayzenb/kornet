"""
iteratively creates sbatch scripts to run multiple jobs at once the
"""
project_name = 'kornet'
import os
#get current working directory
cwd = os.getcwd()
git_dir = cwd.split(project_name)[0] + project_name
import sys
sys.path.append(git_dir)
import subprocess
from glob import glob
import os
import time
import pdb
from datetime import datetime

mem = 24
run_time = "4-00:00:00"


study_dir = f'{git_dir}/modelling'

def setup_sbatch_cpu(job_name, script_name):
    sbatch_setup = f"""#!/bin/bash -l
# Job name
#SBATCH --job-name={job_name}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vayzenb@cmu.edu
# Submit job to cpu queue                
#SBATCH -p cpu
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
# Job memory request
#SBATCH --mem=24gb
# Time limit days-hrs:min:sec
#SBATCH --time {run_time}
# Exclude
# SBATCH --exclude=mind-1-23,mind-1-34,mind-1-30,mind-1-32
# Standard output and error log
#SBATCH --output={study_dir}/slurm_out/{job_name}.out


conda activate ml

{script_name}
"""
    return sbatch_setup

def setup_sbatch_gpu(job_name, script_name):
    sbatch_setup = f"""#!/bin/bash -l
# Job name
#SBATCH --job-name={job_name}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vayzenb@cmu.edu
# Submit job to cpu queue                
#SBATCH -p gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# Job memory request
#SBATCH --mem=48gb
# Time limit days-hrs:min:sec
#SBATCH --time {run_time}
# Exclude
#SBATCH --exclude=mind-1-23,mind-1-34,mind-1-30,mind-1-32
# Standard output and error log
#SBATCH --output={study_dir}/slurm_out/{job_name}.out


conda activate ml

{script_name}
"""
    return sbatch_setup


#base models
model_arch = ['vonenet_ff_ecoset','vonenet_ff_stylized-ecoset','vonenet_r_ecoset','vonenet_r_stylized-ecoset', 'SayCam', 'cvcl', 'convnext','vit','clip_vit',
              'resnet50','resnet50_21k', 'clip_resnet_15m','clip_resnet']


#model_arch = model_arch + model_arch
#create list of of len(model_arch) with imagenet_sketch in each element
#model_weights = [None] *len(model_arch) + ['imagenet_sketch']*len(model_arch)


acts_script = False

stim_dirs = [f'{git_dir}/stim/test/', '/mnt/DataDrive3/vlad/kornet/image_sets/kornet_images/']


#stim_dirs = ['/mnt/DataDrive3/vlad/kornet/image_sets/kornet_images/']


if acts_script == True:
    for model in model_arch:
        for stim_dir in stim_dirs:
        
            
            #job_name = f'extract_acts_{model}_{stim_dir.split("/")[-1]}'
            #print(job_name)
            #os.remove(f"{job_name}.sh")

            script_name = f'python {study_dir}/extract_acts_layers.py {model} {stim_dir}'
            #script_name = f'python {study_dir}/vision_lang/extract_acts_vision_lang.py {model} {stim_dir}'
            print(model, stim_dir)
            try:
                subprocess.run(script_name.split(' '),check=True, capture_output=True, text=True)
            except:
                print('error for', model, stim_dir)


'''
Whole-model decode script
'''


decode_script = False

model_arch = ['twostream_ff','vonenet_r_ecoset','vonenet_r_stylized-ecoset','vonenet_ff_ecoset','vonenet_ff_stylized-ecoset', 'ShapeNet','SayCam', 'convnext','vit']
model_arch= ['clip_vit','clip_resnet', 'cvcl']
model_arch= ['resnet50_21k']
model_arch= ['vit', 'vit_21k', 'resnet50', 'clip_vit']
model_arch= ['clip_resnet_15m', 'clip_resnet_12m']
model_arch = ['resnet_100m','resnet_1b']

#append '_imagenet_sketch' to each string in model_arch
#model_arch = model_arch+ [f'{model}_imagenet_sketch' for model in model_arch]

conds = ['Outline', 'Pert', 'IC']

classifiers = ['NB', 'KNN', 'logistic', 'NC','SVM', 'Ridge']
#classifiers = ['Ridge', 'NB', 'KNN', 'logistic', 'NC']
#classifiers = ['SVM', 'logistic']

train_ns = [5, 10, 25, 50, 100, 150, 200, 250, 300]
#train_ns = [250, 300]
#train_ns = [100, 150, 200, 250, 300]
#train_ns = [5, 10, 25, 50, 100]
fold_n = 20 

pause_time = 5 #how much time (minutes) to wait between jobs
pause_crit = 10 #how many jobs to do before pausing

if decode_script == True:
    n_job = 0
    for cond in conds:
        for train_n in train_ns:
            for classifier in classifiers:
                for model in model_arch:
            
                
                
                    job_name = f'decode_{model}{classifier}_train{train_n}_fold{fold_n}_{cond}'
                    print(job_name)
                    #os.remove(f"{job_name}.sh")

                    script_name = f'python {study_dir}/decode_images.py {model} {train_n} {classifier} {fold_n} {cond}'
                    subprocess.run(script_name.split(' '),check=True, capture_output=True, text=True)
                    #os.remove(f"{study_dir}/{job_name}.sh")
                    

'''Layer-wise decode script'''
decode_layers = True

model_arch = ['vonenet_ff_ecoset','vonenet_ff_stylized-ecoset','vonenet_r_ecoset','vonenet_r_stylized-ecoset', 'SayCam', 'cvcl', 'convnext','vit','clip_vit',
              'resnet50','resnet50_21k', 'clip_resnet_15m','clip_resnet']





conds = ['Outline', 'Pert', 'IC']

conds = ['Outline']

classifiers = ['KNN']


train_ns = [150]
#train_ns = [250, 300]
#train_ns = [100, 150, 200, 250, 300]
#train_ns = [5, 10, 25, 50, 100]
fold_n = 20 

pause_time = 5 #how much time (minutes) to wait between jobs
pause_crit = 10 #how many jobs to do before pausing

if decode_layers == True:
    n_job = 0
    for cond in conds:
        for train_n in train_ns:
            for classifier in classifiers:
                for model in model_arch:
            
                
                
                    job_name = f'decode_{model}{classifier}_train{train_n}_fold{fold_n}_{cond}'
                    print(job_name)
                    #os.remove(f"{job_name}.sh")
                    
                    
                    script_name = f'python {study_dir}/decode_images_layerwise.py {model} {train_n} {classifier} {fold_n} {cond}'
                    try:
                        proc = subprocess.run(script_name.split(' '),check=True, capture_output=True, text=True)
                    except subprocess.CalledProcessError as e:
                        print("Error:", e.stderr)
                    except:
                        print("Error:", proc.stdout)
                        
                    
                    
                    #os.remove(f"{study_dir}/{job_name}.sh")
                    

            