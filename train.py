import argparse
import sys
import time
import torch
import wandb
import pandas as pd
import os
import ast
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
import wandb
import random
from utils import *
from utils.torch_utils import get_grad_norm_squared, cosine_wa_lr, step_decay_lr
from utils.sys_utils import dict_to_namespace, print_args, str2bool
from models import *
from models.cifar_resnet_vgg import ResNet18, ResNet34, vgg11_bn
from models.small_networks import CIFARCNN2, CIFARCNN3, MNISTNet1, MNISTNet2, MNISTNet3, CIFARCNN1
from models.model_svhn import svhn_model
from models.model_STL10 import stl10_model


def main():
    arguments = sys.argv
    print("Command-line arguments:", arguments)

    gpu = [0]

    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='Dec 23', type=str)
    parser.add_argument('--use_wandb', type=str2bool, nargs='?',const=True)
    parser.add_argument('--uid', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--bs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument("--lr_decay", type=str2bool, nargs='?',const=True)
    parser.add_argument('--wd', type=float)
    parser.add_argument('--beta1', default=0, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--option', type=str)
    args = parser.parse_args()

    print_args(args)

    ########### Setting Up Seed ###########
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ########### Setting Up GPU ########### 
    torch.cuda.set_device(gpu[0])
    device = 'cuda'
    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    print(f"Using GPU {gpu_index}: {gpu_name}")

    ########### Setting Up wandb ########### 
    if args.use_wandb:
        run=wandb.init(project=args.project,config=vars(args), dir="/wandb_tmp")
    print(vars(args))

    ########### Setup Data and Model ###########    
    if args.dataset=="FMNIST":
        subset = 5000

        #data
        train_dataset = datasets.FashionMNIST('./data',download=True, train= True, transform=transforms.ToTensor())
        validation_dataset = datasets.FashionMNIST('./data',download=True, train= False, transform=transforms.ToTensor())

        #Trainloader subset
        #train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=settings["bs"], shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
        subset = random.sample(range(train_dataset.data.shape[0]),subset)
        sample_ds = torch.utils.data.Subset(train_dataset, subset)
        sample_sampler = torch.utils.data.RandomSampler(sample_ds)
        train_loader = torch.utils.data.DataLoader(sample_ds, sampler=sample_sampler, batch_size=args.bs, num_workers=8, pin_memory=True, persistent_workers=True)

        #testloader
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)       

        #models
        if args.model == "MLP1": 
            model = torch.nn.DataParallel(MNISTNet1(),gpu).cuda()
        elif args.model == "MLP2":
            model = torch.nn.DataParallel(MNISTNet2(),gpu).cuda()
        elif args.model == "MLP3":
            model = torch.nn.DataParallel(MNISTNet3(),gpu).cuda()
        else: print("model not defined")
        criterion = nn.CrossEntropyLoss()

    elif args.dataset=="svhn":
        transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
        validation_dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)       
        if args.model == "CNN": 
            model = torch.nn.DataParallel(svhn_model(n_channel=32),gpu).cuda() 
        else: print("model not defined")
        criterion = nn.CrossEntropyLoss()

    elif args.dataset=="stl10":
        transform_train=transforms.Compose([
                            transforms.Pad(4),
                            transforms.RandomCrop(96),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ])
        transform_test=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ])
        train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform_train)
        validation_dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)       
        if args.model == "CNN": 
            model = torch.nn.DataParallel(stl10_model(n_channel=32),gpu).cuda() 
        else: print("model not defined")
        criterion = nn.CrossEntropyLoss()            
       

    elif args.dataset=="cifar10":

        #data transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        #data
        train_dataset = torchvision.datasets.CIFAR10(root='./data', download=True, train=True, transform=transform_train)
        validation_dataset = torchvision.datasets.CIFAR10(root='./data', download=True, train=False, transform=transform_test)
        
        #trainloader subset
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)

        #testloader
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)       

        #models
        if args.model == "CNN1": 
            model = torch.nn.DataParallel(CIFARCNN1(),gpu).cuda()
        elif args.model == "CNN2": 
            model = torch.nn.DataParallel(CIFARCNN2(),gpu).cuda()
        elif args.model == "CNN3": 
            model = torch.nn.DataParallel(CIFARCNN3(),gpu).cuda()
        elif args.model == "VGG11":
            model = torch.nn.DataParallel(vgg11_bn(),gpu).cuda()
        elif args.model == "res18":
            model = torch.nn.DataParallel(ResNet18(),gpu).cuda()
        elif args.model == "CIFAR10Res34":
            model = torch.nn.DataParallel(ResNet34(),gpu).cuda()
        else: 
            raise NotImplementedError("Model not defined")
        criterion = nn.CrossEntropyLoss() 
        
   
    ##### iteration counter
    iteration = 0
    total_steps = int(len(train_loader)*args.epochs)

    ########### Getting number of layers ###########      
    n_groups = 0
    dim_model = 0
    with torch.no_grad():
        for param in model.parameters():   
            n_groups = n_groups + 1
            dim_model = dim_model + torch.numel(param)
    print('Model dimension: ' + str(dim_model))
    print('Number of groups: ' + str(n_groups))
    print('Number of iterations: ' + str(total_steps))
    print(f'Steps per epoch: {len(train_loader)}')

    ########### Init of Optimizers ###########      
    avg_grad_1, avg_grad_2 = [], []
    grad_norm_squared_sum = 0
    for p in model.parameters():
        avg_grad_1.append(None)
        avg_grad_2.append(None) 
    if args.optimizer == 'sgdm_torch':
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum = args.beta1, weight_decay= args.wd)
        if args.lr_decay:
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)


    df = [] #stats saved here
    filename = args.model+'_'+args.dataset+'_'+args.optimizer+'_s'+str(args.seed)+'_lr'+str(args.lr)+'_decay'+str(args.lr_decay)+'_uid'+str(args.uid)+'.csv'
    print('saving in'+str(filename))

	########### Training ###########     
    for epoch in range(args.epochs):
        start_time = time.time()

        ########### Saving stats every few epochs ###########         
        model.eval()

        #computing stats: train loss
        train_loss, correct = 0, 0
        for d in train_loader:
            data, target = d[0].to(device, non_blocking=True),d[1].to(device, non_blocking=True)
            output = model(data)
            train_loss += criterion(output, target).data.item()/len(train_loader)
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
        accuracy_train = 100. * correct.to(torch.float32) / len(train_loader.dataset)

        #computing stats: test loss
        test_loss, correct, total = 0, 0, 0
        for d in validation_loader:
            data, target = d[0].to(device, non_blocking=True),d[1].to(device, non_blocking=True)
            output = model(data)
            test_loss += criterion(output, target).data.item()/len(validation_loader)
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
        accuracy_test = 100. * correct.to(torch.float32) / len(validation_loader.dataset)

        #saving to wandb
        if args.use_wandb:
            wandb.log({"train_loss":train_loss, "train_acc":accuracy_train,"test_loss":test_loss,"test_acc":accuracy_test}, commit=False)

        print('Epoch {}: Train L: {:.4f}, TrainAcc: {:.2f}, Test L: {:.4f}, TestAcc: {:.2f} \n'.format(epoch, train_loss, accuracy_train, test_loss, accuracy_test))

        ###########  Training Loop ########### 
    
        model.train()
        for _, batch in enumerate(train_loader):

            if args.optimizer == 'sgdm_torch':
                opt.zero_grad() 
            else:
                model.zero_grad() 
            ###########  Backprop  ###########
            data, target = batch[0].to(device),batch[1].to(device)

            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Getting gradient norm squared
            with torch.no_grad():
                grad_norm_squared = get_grad_norm_squared(model)
                if args.use_wandb:
                    wandb.log({"grad_norm_squared":grad_norm_squared}, commit=False)

            ###########  Optimizer update for standard methods  ########### 
            if args.optimizer == 'sgdm_torch':
                preconditioner = opt.param_groups[0]["lr"]
                if args.use_wandb:
                    wandb.log({"effective_lr":preconditioner}, commit=False)
                    df.append({'model': args.model, 'data': args.dataset, 'opt':args.optimizer, 'seed': args.seed, 'base_lr': args.lr, 'decay_lr': args.lr_decay, 'loss': loss.item(), 'epoch': epoch, 'lr': preconditioner})
                opt.step()

            elif args.optimizer == 'sgdm':
                with torch.no_grad(): 
                    #updating grad moving average 
                    norm_avg_grad_squared = 0
                    for p_idx, p in enumerate(model.parameters()):
                        if avg_grad_1[p_idx]==None:
                            avg_grad_1[p_idx] =  p.grad
                        avg_grad_1[p_idx] = args.beta1 * avg_grad_1[p_idx] + (1-args.beta1)*p.grad
                        norm_avg_grad_squared = norm_avg_grad_squared + avg_grad_1[p_idx].detach().data.norm(2).item()**2
                    #updating parameters
                    if args.lr_decay:
                        #sigma_curr = cosine_wa_lr(args.lr, iteration, total_steps, 0) 
                        sigma_curr = step_decay_lr(args.lr, epoch, [10,20], 0.1)
                    else:
                        sigma_curr = args.lr
                    preconditioner = sigma_curr
                    for p_idx, p in enumerate(model.parameters()):
                        new_val = p - preconditioner * (avg_grad_1[p_idx] - args.wd * p)
                        p.copy_(new_val)
                    if args.use_wandb:
                        wandb.log({"effective_lr":preconditioner}, commit=False)
                        df.append({'model': args.model, 'data': args.dataset, 'opt':args.optimizer, 'seed': args.seed, 'base_lr': args.lr, 'decay_lr': args.lr_decay, 'loss': loss.item(), 'epoch': epoch, 'lr': preconditioner})

            elif args.optimizer == 'ngnm':

                with torch.no_grad(): 
                    #updating grad moving average 
                    norm_avg_grad_squared = 0
                    for p_idx, p in enumerate(model.parameters()):
                        if avg_grad_1[p_idx]==None:
                            avg_grad_1[p_idx] =  p.grad
                        avg_grad_1[p_idx] = args.beta1 * avg_grad_1[p_idx] + (1-args.beta1)*p.grad
                        norm_avg_grad_squared = norm_avg_grad_squared + avg_grad_1[p_idx].detach().data.norm(2).item()**2

                    #updating parameters
                    if args.lr_decay:
                        #sigma_curr = cosine_wa_lr(args.lr, iteration, total_steps, 0) 
                        sigma_curr = step_decay_lr(args.lr, epoch, [10,20], 0.1)
                    else:
                        sigma_curr = args.lr
                    preconditioner = sigma_curr/(1+sigma_curr*norm_avg_grad_squared/(2*loss.item()))
                    for p_idx, p in enumerate(model.parameters()):
                        new_val = p - preconditioner * (avg_grad_1[p_idx] - args.wd * p)
                        p.copy_(new_val)
                    if args.use_wandb:
                        wandb.log({"effective_lr":preconditioner}, commit=False)
                        df.append({'model': args.model, 'data': args.dataset, 'opt':args.optimizer, 'seed': args.seed, 'base_lr': args.lr, 'decay_lr': args.lr_decay, 'loss': loss.item(), 'epoch': epoch, 'lr': preconditioner})

            elif args.optimizer == 'sps':

                with torch.no_grad(): 
                    #updating grad moving average 
                    norm_avg_grad_squared = 0
                    for p_idx, p in enumerate(model.parameters()):
                        if avg_grad_1[p_idx]==None:
                            avg_grad_1[p_idx] =  p.grad
                        avg_grad_1[p_idx] = args.beta1 * avg_grad_1[p_idx] + (1-args.beta1)*p.grad
                        norm_avg_grad_squared = norm_avg_grad_squared + avg_grad_1[p_idx].detach().data.norm(2).item()**2

                    #updating parameters
                    if args.lr_decay:
                        sigma_curr = step_decay_lr(args.lr, epoch, [10,20], 0.1)
                    else:
                        sigma_curr = args.lr
                    preconditioner =  np.min([sigma_curr, loss.item()/norm_avg_grad_squared])
                    for p_idx, p in enumerate(model.parameters()):
                        new_val = p - preconditioner * (avg_grad_1[p_idx] - args.wd * p)
                        p.copy_(new_val)
                    if args.use_wandb:
                        wandb.log({"effective_lr":preconditioner}, commit=False)
                        df.append({'model': args.model, 'data': args.dataset, 'opt':args.optimizer, 'seed': args.seed, 'base_lr': args.lr, 'decay_lr': args.lr_decay, 'loss': loss.item(), 'epoch': epoch, 'lr': preconditioner})

            elif args.optimizer == 'adagrad':

                with torch.no_grad(): 
                    #updating grad moving average 
                    norm_avg_grad_squared = 0
                    for p_idx, p in enumerate(model.parameters()):
                        if avg_grad_1[p_idx]==None:
                            avg_grad_1[p_idx] =  p.grad
                        avg_grad_1[p_idx] = args.beta1 * avg_grad_1[p_idx] + (1-args.beta1)*p.grad
                        norm_avg_grad_squared = norm_avg_grad_squared + avg_grad_1[p_idx].detach().data.norm(2).item()**2
                    if args.lr_decay:
                        sigma_curr = step_decay_lr(args.lr, epoch, [10,20], 0.1)
                    else:
                        sigma_curr = args.lr
                    grad_norm_squared_sum = grad_norm_squared_sum + norm_avg_grad_squared
                    preconditioner =  sigma_curr/(np.sqrt(1e-2+grad_norm_squared_sum))
                    for p_idx, p in enumerate(model.parameters()):
                        new_val = p - preconditioner * (avg_grad_1[p_idx] - args.wd * p)
                        p.copy_(new_val)
                    if args.use_wandb:
                        wandb.log({"effective_lr":preconditioner}, commit=False)
                        df.append({'model': args.model, 'data': args.dataset, 'opt':args.optimizer, 'seed': args.seed, 'base_lr': args.lr, 'decay_lr': args.lr_decay, 'loss': loss.item(), 'epoch': epoch, 'lr': preconditioner})


            elif args.optimizer == 'adam': #example of writing an optimizer
                with torch.no_grad(): # very important
                    #standard adam parameters
                    if args.lr_decay:
                        #sigma_curr = cosine_wa_lr(args.lr, iteration, total_steps, 0) 
                        sigma_curr = step_decay_lr(args.lr, epoch, [10,20], 0.1)
                    else:
                        sigma_curr = args.lr

                    # parameters have to be updated with a fpr loop (this is fast)
                    for p_idx, p in enumerate(model.parameters()):
                        grad_p = p.grad
                        square_grad_p = grad_p**2
                        if None in (avg_grad_1[p_idx], avg_grad_2[p_idx]): #init
                            avg_grad_1[p_idx] =  grad_p
                            avg_grad_2[p_idx] = square_grad_p
                        avg_grad_2[p_idx] = args.beta2 * avg_grad_2[p_idx] + (1-args.beta2)*square_grad_p
                        avg_grad_1[p_idx] = args.beta1 * avg_grad_1[p_idx] + (1-args.beta1)*grad_p

                        #parameter update
                        preconditioner = 1/(1e-8 + avg_grad_2[p_idx].sqrt())
                        new_val = p - sigma_curr * preconditioner * avg_grad_1[p_idx]
                        p.copy_(new_val)
                    if args.use_wandb:
                        #wandb.log({"effective_lr":preconditioner}, commit=False)
                        df.append({'net': args.model, 'data': args.dataset, 'opt':args.optimizer, 'seed': args.seed, 'base_lr': args.lr, 'decay_lr': args.lr_decay, 'loss': loss.item(), 'epoch': epoch})
            else:
                raise ValueError('Optimizer not defined')
            iteration = iteration +1
            
        if args.optimizer == 'sgdm_torch':
            if args.lr_decay:
                scheduler.step() 

        epoch_time = time.time()-start_time
        if args.use_wandb:
            wandb.log({"epoch_time":epoch_time})
            pd.DataFrame(df).to_csv(os.path.join('results', filename)) #backup

            
    ########### Closing Writer ###########  
    if args.use_wandb:
        run.finish()
        wandb.finish()

    return None


if __name__ == '__main__':
    main()