import os
import numpy as np
import itertools
import argparse
from train import run_config
from utils import *
from utils.sys_utils import get_experiment_directory, generate_filename, load_config, generate_combinations,  print_overview

def run_experiment(use_wandb, gpu, project, dataset, architecture, seed, opt):
    # generating folder to store experiments
    experiment_dir = get_experiment_directory(project, dataset, architecture)
    filename = generate_filename(opt)
    results_path = os.path.join(experiment_dir, filename)
    
    # running experiment
    run_config(use_wandb, gpu, project, dataset, architecture, seed, opt)
    #result = np.array([1, 2, 3])  # Dummy results, replace with actual results

    # running experiment
    #np.save(results_path, result)
    #print(f"Results saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", type=bool, default=False, help="Whether to use wandb for logging. Default is True.")
    parser.add_argument("--gpu", type=str, default='[1,2]')
    parser.add_argument("--config", type=str, default="cifar_resnet.yml", help="Path to the configuration YAML file.")
    args = parser.parse_args()
    config = load_config('configs/'+args.config)
    for experiment in config['experiments']:
        datasets = experiment.get('dataset', [])
        architectures = experiment.get('arch', [])
        seed = experiment.get('seed')
        opt_combinations = generate_combinations(experiment['opt'])
        for dataset, architecture, opt in itertools.product(datasets, architectures, opt_combinations):
            print_overview(args.wandb, args.gpu, config['project'], dataset, architecture, seed, opt)
            run_experiment(args.wandb, args.gpu, config['project'], dataset, architecture, seed, opt)
