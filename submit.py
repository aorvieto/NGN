# figura 10 e 43
import uuid
import itertools
import os
from argparse import ArgumentParser
import subprocess

def make_submisison_file_content(executable, arguments, output, error, log, cpus=1, gpus=0, memory=1000, disk="1G"):
    d = {
        'executable': executable,
        'arguments': arguments,
        'output': output,
        'error': error,
        'log': log,
        'request_cpus': cpus,
        'request_gpus': gpus,
        'request_memory': memory,
        'request_disk': disk
    }
    return d

def run_job(uid, bid, d):
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    job_file = os.path.join('tmp', uid)
    with open(job_file, 'w') as f:  
        for key, value in d.items():  
            f.write(f'{key} = {value}\n')
        f.write("queue")

    subprocess.run(["condor_submit_bid", str(bid), job_file]) 


if __name__ == '__main__':
    project = 'Dec_NGN_MLP_final'
    use_wandb = 'true'
    dataset = 'FMNIST'
    model = 'MLP2'
    epochs = [300]
    optimizer = ['adam']
    seed = [0,1,2]
    bs = [128]
    #lr = [3]
    #lr = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30] #, 10, 30, 100, 300, 1000
    lr = [0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01]
    #lr = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]

    lr_decay = ['false']
    wd = [0.0]
    beta1 = [0.9]
    beta2 = [0.999]
    option = ['not_interesting_option']

    #python train.py --project $1 --use_wandb $2 --uid $3 --dataset $4 --model $5 --epochs $6 --optimizer $7 --seed $8 --bs $9 --lr $10 --lr_decay $11 --wd $12 --beta1 $13 --beta2 $14 --option $15


    for run in itertools.product(*[epochs, optimizer, seed, bs, lr, lr_decay, wd, beta1, beta2, option]):
        uid = uuid.uuid4().hex[:10]
        arguments = f"{project} {use_wandb} {uid} {dataset} {model} {run[0]} {run[1]} {run[2]} {run[3]} {run[4]} {run[5]} {run[6]} {run[7]} {run[8]} {run[9]} xent"
        output = f"runs/{uid}.stdout"
        error = f"runs/{uid}.stderr"
        log = f"runs/{uid}.log"
        cpus = 8
        gpus = 1
        memory = 10000
        disk = "1G"
        executable = "run.sh"

        try:
            content = make_submisison_file_content(executable, arguments, output, error, log, cpus, gpus, memory, disk)
            run_job(uid, 15, content)
        except:
            raise ValueError("Crashed.")
    print("Done.")