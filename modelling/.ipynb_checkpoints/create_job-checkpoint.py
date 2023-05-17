"""
iteratively creates sbatch scripts to run multiple jobs at once the
"""

import subprocess
from glob import glob
import os
import time
import pdb
from datetime import datetime
now = datetime.now()
curr_date=now.strftime("%Y%m%d")

mem = 36
gpu_n = 1
run_time = "5-00:00:00"

#subj info
#stim info
study_dir = f'/user_data/vayzenbe/GitHub_Repos/kornet/modelling'


stim_dir = f'/lab_data/behrmannlab/image_sets/'
stim_dir =f'/user_data/vayzenbe/image_sets/'

model_dir = f'/user_data/vayzenbe/GitHub_Repos/vonenet'

#training info
model_arch = ['cornet_ff','cornet_s']

train_types = ['imagenet-sketch']
train_types = ['ecoset']
suf = ''

def setup_sbatch(job_name, script_name):
    sbatch_setup = f"""#!/bin/bash -l


# Job name
#SBATCH --job-name={job_name}

#SBATCH --mail-type=ALL
#SBATCH --mail-user=vayzenb@cmu.edu

# Submit job to cpu queue                
#SBATCH -p gpu

#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:{gpu_n}

# Job memory request
#SBATCH --mem={mem}gb

# Time limit days-hrs:min:sec
#SBATCH --time {run_time}

# Exclude
#SBATCH --exclude=mind-1-1

# Standard output and error log
#SBATCH --output={study_dir}/slurm_out/{job_name}.out

conda activate ml



{script_name}


"""
    return sbatch_setup


def copy_data(train_type):
    try:
        #copy data
        subprocess.run(f'rsync -a {stim_dir}/{train_type} /scratch/vayzenbe/', shell=True, check=True)
        train_dir = f'/scratch/vayzenbe/{train_type}'
        print('data copied')
    except:
        print('data not copied')
        train_dir = f'{stim_dir}/{train_type}'

    return train_dir


""" 
job_name = f'extract_acts'
print(job_name)

#os.remove(f"{job_name}.sh")

f = open(f"{job_name}.sh", "a")
f.writelines(setup_sbatch(job_name))


f.close()

subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
os.remove(f"{job_name}.sh") """



for model in model_arch:
    for train_type in train_types:
        
        job_name = f'{model}_{train_type}{suf}_{curr_date}'
        print(job_name)
        
        #os.remove(f"{job_name}.sh")
        
        f = open(f"{job_name}.sh", "a")
        script_name = f'python {study_dir}/train.py --data {stim_dir}/{train_type} -o /lab_data/behrmannlab/vlad/kornet/modelling/weights/ --arch {model} --workers 4 -b 128'
        f.writelines(setup_sbatch(job_name,script_name))
        
        
        f.close()
        
        subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
        os.remove(f"{job_name}.sh")


#python {model_dir}/train.py --in_path {stim_dir}/{train_type} -o /lab_data/behrmannlab/vlad/kornet/modelling/weights/ --model_arch {model_arch} --ngpus {gpu_n}

#python finetune_models.py --data {stim_dir}/{train_type}/ -o /lab_data/behrmannlab/vlad/kornet/modelling/weights/ --arch {model} -b 64 --workers 4 --epochs 30

#python finetune_models.py --data {stim_dir}/{train_type}/ -o /lab_data/behrmannlab/vlad/kornet/modelling/weights/ --arch {model} -b 32 --workers 4 --epochs 30 --resume /lab_data/behrmannlab/vlad/kornet/modelling/weights/{model}_{train_cat}_checkpoint_1.pth.tar 








