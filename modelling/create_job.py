"""
iteratively creates sbatch scripts to run multiple jobs at once the
"""

import subprocess
from glob import glob
import os
import time
import pdb

mem = 36
gpu_n = 2
run_time = "3-00:00:00"

#subj info
#stim info
study_dir = f'/user_data/vayzenbe/GitHub_Repos/kornet/modelling'
stim_dir = f'/lab_data/tarrlab/common/datasets/ILSVRC/Data/'

stim_dir = f'/lab_data/behrmannlab/image_sets/'

model_dir = f'/user_data/vayzenbe/GitHub_Repos/vonenet'

#training info
model_arch = ['cornets_ff']

train_types = [ 'imagenet_sketch']


def setup_sbatch(model, train_cat):
    sbatch_setup = f"""#!/bin/bash -l


# Job name
#SBATCH --job-name={model}_{train_cat}

#SBATCH --mail-type=ALL
#SBATCH --mail-user=vayzenb@cmu.edu

# Submit job to cpu queue                
#SBATCH -p gpu

#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:{gpu_n}

# Job memory request
#SBATCH --mem={mem}gb

# Time limit days-hrs:min:sec
#SBATCH --time {run_time}

# Exclude
#SBATCH --exclude=mind-1-34

# Standard output and error log
#SBATCH --output={study_dir}/slurm_out/{model}_{train_cat}.out

conda activate ml_new

#rsync -a {stim_dir}/{train_type} /scratch/vayzenbe/
#echo "data copied"

echo {stim_dir}/{train_type}
#python {model_dir}/train.py --in_path {stim_dir}/{train_type} -o /lab_data/behrmannlab/vlad/kornet/modelling/weights/ --model_arch {model_arch} --ngpus {gpu_n}

python finetune_models.py --data {stim_dir}/{train_type}/ -o /lab_data/behrmannlab/vlad/kornet/modelling/weights/ --arch {model} -b 64 --workers 4 --epochs 30

#python finetune_models.py --data {stim_dir}/{train_type}/ -o /lab_data/behrmannlab/vlad/kornet/modelling/weights/ --arch {model} -b 32 --workers 4 --epochs 30 --resume /lab_data/behrmannlab/vlad/kornet/modelling/weights/{model}_{train_cat}_checkpoint_1.pth.tar 

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




for model in model_arch:
    for train_type in train_types:
        
        job_name = f'{model}_{train_type}'
        print(job_name)
        
        #os.remove(f"{job_name}.sh")
        
        f = open(f"{job_name}.sh", "a")
        f.writelines(setup_sbatch(model, train_type))
        
        
        f.close()
        
        subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
        os.remove(f"{job_name}.sh")








