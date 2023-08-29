import os
import numpy as np
import itertools
import argparse
from train import run_config
from utils import *
import torch
from utils.sys_utils import get_experiment_directory, generate_filename, load_config, generate_combinations,  print_overview

def run_experiment(use_wandb, gpu, project, dataset, architecture, seed, opt):
    experiment_dir = get_experiment_directory(project, dataset, architecture, opt)
    filename = generate_filename(opt)
    results_path = os.path.join(experiment_dir, filename)
    results = np.zeros(4)#run_config(use_wandb, gpu, project, dataset, architecture, seed, opt)
    torch.save(results,results_path+".pt")
    print(f"Results saved to {results_path}")
    
#python --gpu "[2]" --config "cifar_resnet6.yml"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", type=bool, default=False, help="Whether to use wandb for logging. Default is True.")
    parser.add_argument("--gpu", type=str, default='[1]')
    parser.add_argument("--config", type=str, default="cifar_resnet.yml", help="Path to the configuration YAML file.")
    args = parser.parse_args()
    config = load_config('configs/'+args.config)
    for experiment in config['experiments']:
        datasets = experiment.get('dataset', [])
        architectures = experiment.get('arch', [])
        seeds = experiment.get('seed', [])
        opt_combinations = generate_combinations(experiment['opt'])
        for dataset, architecture, seed, opt in itertools.product(datasets, architectures, seeds, opt_combinations):
            print_overview(args.wandb, args.gpu, config['project'], dataset, architecture, seed, opt)
            run_experiment(args.wandb, args.gpu, config['project'], dataset, architecture, seed, opt)
