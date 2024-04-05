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
    project = 'Dec_NGN_imagenet_test2'
    use_wandb = 'true'
    optimizer = ['adam']
    seed = [0]
    lr = [0.003, 0.001, 0.0003]
    lr_decay = ['true']
    wd = [0.0]
    beta1 = [0.9]

    for run in itertools.product(*[optimizer, seed, lr, lr_decay, wd, beta1]):
        uid = uuid.uuid4().hex[:10]
        arguments = f"{project} {use_wandb} {uid} {run[0]} {run[1]} {run[2]} {run[3]} {run[4]} {run[5]} xent"
        output = f"runs/imagenet{uid}.stdout"
        error = f"runs/imagenet{uid}.stderr"
        log = f"runs/imagenet{uid}.log"
        cpus = 12
        gpus = 4
        memory = 50000
        disk = "4G"
        executable = "run_imagenet.sh"
        try:
            content = make_submisison_file_content(executable, arguments, output, error, log, cpus, gpus, memory, disk)
            run_job(uid, 15, content)
        except:
            raise ValueError("Crashed.")
    print("Done.")