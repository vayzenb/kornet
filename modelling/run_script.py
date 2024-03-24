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
run_time = "2-00:00:00"
pause_time = 6 #how much time (minutes) to wait between jobs
pause_crit = 4 #how many jobs to do before pausing

study_dir = f'{git_dir}/modelling'

def setup_sbatch_cpu(job_name, script_name):
    sbatch_setup = f"""#!/bin/bash -l
# Job name
#SBATCH --job-name={job_name}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vayzenb@cmu.edu
# Submit job to cpu queue                
#SBATCH -p cpu
#SBATCH --cpus-per-task=1
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
model_arch = ['vonenet_r_ecoset','vonenet_r_stylized-ecoset','vonenet_ff_ecoset','vonenet_ff_stylized-ecoset', 'ShapeNet','SayCam', 'convnext','vit']
model_arch = model_arch + model_arch
#create list of of len(model_arch) with imagenet_sketch in each element
weights = [''] *len(model_arch) + ['imagenet_sketch']*len(model_arch)


acts_script = True
if acts_script == True:
    n_job = 0
    for model in model_arch:
        job_name = f'{model}{weights}_extract_acts'
        print(job_name)
        #os.remove(f"{job_name}.sh")

        script_name = f'python {study_dir}/extract_acts.py {model} {weights}'
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

model_arch = ['vonenet_r_ecoset','vonenet_r_stylized-ecoset','vonenet_ff_ecoset','vonenet_ff_stylized-ecoset', 'ShapeNet','SayCam', 'convnext','vit']
model_arch = model_arch + model_arch
#create list of of len(model_arch) with imagenet_sketch in each element
weights = [''] *len(model_arch) + ['imagenet_sketch']*len(model_arch)


classifiers = ['SVM', 'Ridge', 'NB', 'KNN', 'logistic', 'NC']

train_ns = [50, 100, 150, 200, 250, 300]
fold_n = 20 

if decode_script == True:
    n_job = 0
    for model in model_arch:
        for classifier in classifiers:
            for train_n in train_ns:
            
                job_name = f'{model}{weights}_{classifier}_train{train_n}_fold{fold_n}'
                print(job_name)
                #os.remove(f"{job_name}.sh")

                script_name = f'python {study_dir}/decode_images.py {model} {train_n} {classifier} {fold_n}'

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



        