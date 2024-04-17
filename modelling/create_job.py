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

mem = 48
gpu_n = 3
cpu_n = 12
run_time = "10-00:00:00"
n_jobs = 4
wait_time = 5
'''
mem = 24
gpu_n = 1
cpu_n = 4
run_time = "1-00:00:00"
n_jobs = 5
wait_time = 30
'''

#subj info
#stim info
study_dir = f'/user_data/vayzenbe/GitHub_Repos/kornet/modelling'


stim_dir = f'/lab_data/behrmannlab/image_sets/'
stim_dir =f'/user_data/vayzenbe/image_sets/'
#stim_dir =f'/lab_data/plautlab/imagesets/'


model_dir = f'/user_data/vayzenbe/GitHub_Repos/vonenet'

#training info
model_arch = ['cornet_ff','cornet_s']

train_types = ['ecoset','stylized-ecoset']


def setup_sbatch(job_name, script_name):
    sbatch_setup = f"""#!/bin/bash -l


# Job name
#SBATCH --job-name={job_name}

#SBATCH --mail-type=ALL
#SBATCH --mail-user=vayzenb@cmu.edu

# Submit job to cpu queue                
#SBATCH -p gpu

#SBATCH --cpus-per-task={cpu_n}
#SBATCH --gres=gpu:{gpu_n}

# Job memory request
#SBATCH --mem={mem}gb

# Time limit days-hrs:min:sec
#SBATCH --time {run_time}

# Exclude
#SBATCH --exclude=mind-1-11,mind-1-23

# Standard output and error log
#SBATCH --output={study_dir}/slurm_out/{job_name}.out




conda activate ml



{script_name}


"""
    return sbatch_setup


def copy_data(train_type):
    #check whether train_type exists
    if not os.path.exists(f'{stim_dir}/{train_type}'):
        print(f'{stim_dir}/{train_type} does not exist')
        return
    

    try:
        #copy data
        subprocess.run(f'rsync -a {stim_dir}/{train_type} /scratch/vayzenbe/', shell=True, check=True)
        train_dir = f'/scratch/vayzenbe/{train_type}'
        print('data copied')
    except:
        print('data not copied')
        train_dir = f'{stim_dir}/{train_type}'

    return train_dir



model_arch = ['vonecornet_s','cornet_s','voneresnet', 'vit','convnext','resnet50','resnext50','alexnet','vgg19', 'ShapeNet','SayCam']
model_arch = ['vonenet_r_ecoset','vonenet_r_stylized-ecoset','vonenet_ff_ecoset','vonenet_ff_stylized-ecoset', 'ShapeNet','SayCam', 'vit','convnext']

model_arch = ['vit']
acts_script = False
if acts_script == True:
    n_job = 0
    for model in model_arch:
        job_name = f'{model}_extract_acts_{curr_date}'
        print(job_name)

        #os.remove(f"{job_name}.sh")

        f = open(f"{job_name}.sh", "a")
        script_name = f'python {study_dir}/extract_acts.py {model} imagenet_sketch'
        f.writelines(setup_sbatch(job_name,script_name))
        f.close()

        subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
        os.remove(f"{job_name}.sh") 
        n_job += 1
        if n_job == n_jobs:
            time.sleep(wait_time*60)
            n_job = 0



#stim_dir =f'/lab_data/plautlab/imagesets/'
train_types = ['ecoset','stylized-ecoset']
suf = ''
model_arch = ['cornet_ff','cornet_s','cornet_z']
train_types = ['imagenet_sketch']
model_arch = ['vonenet_r_ecoset','vonenet_r_stylized-ecoset','vonenet_ff_ecoset','vonenet_ff_stylized-ecoset', 'SayCam','ShapeNet']
model_arch = ['vonenet_ff_ecoset','vonenet_ff_stylized-ecoset', 'SayCam','ShapeNet']

train_script = False
if train_script == True:
    n_job = 0
    for model in model_arch:
        for train_type in train_types:
            
            job_name = f'{model}_{train_type}{suf}_{curr_date}'
            print(job_name)
            
            #os.remove(f"{job_name}.sh")
            #mkdir -p /scratch/vayzenbe/
            #rsync -a {stim_dir}/{train_type} /scratch/vayzenbe/
            #echo "copied {stim_dir}/{train_type}"
            
            f = open(f"{job_name}.sh", "a")
            #script_name = f'python {study_dir}/train.py --data /scratch/vayzenbe/{train_type} -o /lab_data/behrmannlab/vlad/kornet/modelling/weights/ --arch {model} --epochs 70 --workers 8 -b 128 --rand_seed 2'
            #script_name = f'python {study_dir}/train.py --data /scratch/vayzenbe/{train_type} -o /lab_data/behrmannlab/vlad/kornet/modelling/weights/ --arch {model} --epochs 70 --workers 8 -b 128 --resume /lab_data/behrmannlab/vlad/kornet/modelling/weights/{model}_{train_type}_checkpoint_1.pth.tar'
            script_name = f'python finetune_models.py --data {stim_dir}/{train_type}/ -o /lab_data/behrmannlab/vlad/kornet/modelling/weights/ --arch {model} -b 32 --workers 4 --epochs 30'
            f.writelines(setup_sbatch(job_name,script_name))
            
            
            f.close()
            
            subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
            os.remove(f"{job_name}.sh")
            
            n_job += 1
            if n_job == n_jobs:
                time.sleep(wait_time*60*60)
                n_job = 0


    #python {model_dir}/train.py --in_path {stim_dir}/{train_type} -o /lab_data/behrmannlab/vlad/kornet/modelling/weights/ --model_arch {model_arch} --ngpus {gpu_n}

    #python finetune_models.py --data {stim_dir}/{train_type}/ -o /lab_data/behrmannlab/vlad/kornet/modelling/weights/ --arch {model} -b 64 --workers 4 --epochs 30

    #python finetune_models.py --data {stim_dir}/{train_type}/ -o /lab_data/behrmannlab/vlad/kornet/modelling/weights/ --arch {model} -b 32 --workers 4 --epochs 30 --resume /lab_data/behrmannlab/vlad/kornet/modelling/weights/{model}_{train_cat}_checkpoint_1.pth.tar 




'''
run two stream model
'''
train_type = 'ecoset'
#train_dir = copy_data(train_type)
train_dir = '/user_data/vayzenbe/image_sets/ecoset'
suf = ''
twostream_script = True
if twostream_script == True:
    job_name = f'twostream{suf}_ecoset_{curr_date}'
    print(job_name)
    
    #os.remove(f"{job_name}.sh")
    #mkdir -p /scratch/vayzenbe/
    #rsync -a {stim_dir}/{train_type} /scratch/vayzenbe/
    #echo "copied {stim_dir}/{train_type}"
    
    f = open(f"{job_name}.sh", "a")
    #script_name = f'python {study_dir}/train.py --data /scratch/vayzenbe/{train_type} -o /lab_data/behrmannlab/vlad/kornet/modelling/weights/ --arch {model} --epochs 70 --workers 8 -b 128 --rand_seed 2'
    #script_name = f'python {study_dir}/train.py --data /scratch/vayzenbe/{train_type} -o /lab_data/behrmannlab/vlad/kornet/modelling/weights/ --arch {model} --epochs 70 --workers 8 -b 128 --resume /lab_data/behrmannlab/vlad/kornet/modelling/weights/{model}_{train_type}_checkpoint_1.pth.tar'
    script_name = f'python modelling/train_twostream.py --data {train_dir} -o /lab_data/behrmannlab/vlad/kornet/modelling/weights/ --epochs 30 --workers 8 -b 128'
    f.writelines(setup_sbatch(job_name,script_name))
    
    
    f.close()
    
    subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
    os.remove(f"{job_name}.sh")



acts_script = False

stim_dir = f'{git_dir}/stim/test'
pause_time = 6 #how much time (minutes) to wait between jobs
pause_crit = 4 #how many jobs to do before pausing
if acts_script == True:
    n_job = 0
    for model in model_arch:
        
        job_name = f'extract_acts_{model}_{stim_dir.split("/")[-1]}'
        print(job_name)
        #os.remove(f"{job_name}.sh")

        script_name = f'python {study_dir}/extract_acts.py {model} {stim_dir}'
        f = open(f'{study_dir}/{job_name}.sh', 'a')
        f.writelines(setup_sbatch_gpu(job_name, script_name))
        f.close()

        subprocess.run(['sbatch', f"{study_dir}/{job_name}.sh"],check=True, capture_output=True, text=True)
        os.remove(f"{study_dir}/{job_name}.sh")
        n_job += 1

        if n_job >= pause_crit:
            #wait X minutes
            time.sleep(pause_time*60)
            n_job = 0 



decode_script = False

model_arch = ['twostream_ff','vonenet_r_ecoset','vonenet_r_stylized-ecoset','vonenet_ff_ecoset','vonenet_ff_stylized-ecoset', 'ShapeNet','SayCam', 'convnext','vit']
model_arch= ['twostream_ff']


#append '_imagenet_sketch' to each string in model_arch
#model_arch = model_arch+ [f'{model}_imagenet_sketch' for model in model_arch]

conds = ['Outline', 'Pert', 'IC']

classifiers = ['SVM', 'Ridge', 'NB', 'KNN', 'logistic', 'NC']
#classifiers = ['Ridge', 'NB', 'KNN', 'logistic', 'NC']
#classifiers = ['SVM', 'logistic']

train_ns = [5, 10, 25, 50, 100, 150, 200, 250, 300]
#train_ns = [100, 150, 200, 250, 300]
#train_ns = [5, 10, 25, 50, 100]
fold_n = 20 

pause_time = 5 #how much time (minutes) to wait between jobs
pause_crit = 10 #how many jobs to do before pausing

if decode_script == True:
    n_job = 0
    for cond in conds:
            
        for model in model_arch:
            for classifier in classifiers:
                for train_n in train_ns:
                
                    job_name = f'decode_{model}{classifier}_train{train_n}_fold{fold_n}_{cond}'
                    print(job_name)
                    #os.remove(f"{job_name}.sh")

                    script_name = f'python {study_dir}/decode_images.py {model} {train_n} {classifier} {fold_n} {cond}'

                    f = open(f'{study_dir}/{job_name}.sh', 'a')
                    f.writelines(setup_sbatch_cpu(job_name, script_name))
                    f.close()

                    subprocess.run(['sbatch', f"{study_dir}/{job_name}.sh"],check=True, capture_output=True, text=True)
                    os.remove(f"{study_dir}/{job_name}.sh")
                    n_job += 1

                    if n_job >= pause_crit:
                        #wait X minutes
                        time.sleep(pause_time*60)
                        n_job = 0 
