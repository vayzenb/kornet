"""
iteratively creates sbatch scripts to run multiple jobs at once the
"""

import subprocess
from glob import glob
import os
import time
import pdb
from datetime import datetime


study_dir = f'/user_data/vayzenbe/GitHub_Repos/kornet/modelling'

model_arch = ['vonenet_r_ecoset','vonenet_r_stylized-ecoset','vonenet_ff_ecoset','vonenet_ff_stylized-ecoset', 'ShapeNet','SayCam', 'vit','convnext']
acts_script = True
if acts_script == True:
    n_job = 0
    for model in model_arch:
        job_name = f'{model}_extract_acts'
        print(job_name)
        #os.remove(f"{job_name}.sh")

        script_name = f'python {study_dir}/extract_acts.py {model} imagenet_sketch'
        subprocess.run(script_name, check=True)


        