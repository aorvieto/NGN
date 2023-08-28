import torch
import wandb
import ast
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
import wandb
import random
from utils import *
from utils.torch_utils import get_grad_norm_squared, cosine_wa_lr
from utils.sys_utils import dict_to_namespace
from models import *
from models.cifar_resnet_vgg import ResNet18, ResNet34
from models.small_networks import CIFARCNN2, CIFARCNN3, MNISTNet1, MNISTNet2, MNISTNet3, CIFARCNN1



def run_config(use_wandb, gpu, project, dataset, architecture, seed, opt):

    ########### Setting Up Seed ###########
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ########### Setting Up GPU ########### 
    gpu_ids = ast.literal_eval(gpu)
    torch.cuda.set_device(gpu_ids[0])
    device = 'cuda'

    ########### Setting Up wandb ########### 
    config = opt
    config['architecture']= architecture
    config['dataset']= dataset
    config['seed']=seed
    if use_wandb:
        run=wandb.init(project=project,config=config, dir="/wandb_tmp")
    print(config)
    opt = dict_to_namespace(opt)

    ########### Setup Data and Model ###########    
    if dataset=="FMNIST":
        subset = 5000

        #data
        train_dataset = datasets.FashionMNIST('./data',download=True, train= True, transform=transforms.ToTensor())
        validation_dataset = datasets.FashionMNIST('./data',download=True, train= False, transform=transforms.ToTensor())

        #Trainloader subset
        #train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=settings["bs"], shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
        subset = random.sample(range(train_dataset.data.shape[0]),subset)
        sample_ds = torch.utils.data.Subset(train_dataset, subset)
        sample_sampler = torch.utils.data.RandomSampler(sample_ds)
        train_loader = torch.utils.data.DataLoader(sample_ds, sampler=sample_sampler, batch_size=opt.bs, num_workers=8, pin_memory=True, persistent_workers=True)

        #testloader
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=opt.bs, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)       

        #models
        if architecture == "MLP1": 
            model = torch.nn.DataParallel(MNISTNet1(),gpu_ids).cuda()
        elif architecture == "MLP2":
            model = torch.nn.DataParallel(MNISTNet2(),gpu_ids).cuda()
        elif architecture == "MLP3":
            model = torch.nn.DataParallel(MNISTNet3(),gpu_ids).cuda()
        else: print("model not defined")
        criterion = nn.CrossEntropyLoss()

    elif dataset=="cifar10":

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
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.bs, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)

        #testloader
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=opt.bs, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)       

        #models
        if architecture == "CNN1": 
            model = torch.nn.DataParallel(CIFARCNN1(),gpu_ids).cuda()
        elif architecture == "CNN2": 
            model = torch.nn.DataParallel(CIFARCNN2(),gpu_ids).cuda()
        elif architecture == "CNN3": 
            model = torch.nn.DataParallel(CIFARCNN3(),gpu_ids).cuda()
        elif architecture == "res18":
            model = torch.nn.DataParallel(ResNet18(),gpu_ids).cuda()
        elif architecture == "CIFAR10Res34":
            model = torch.nn.DataParallel(ResNet34(),gpu_ids).cuda()
        else: 
            raise NotImplementedError("Model not defined")
        criterion = nn.CrossEntropyLoss() 
        
    ########### Setup Writer Variables ###########  
    results = {"train_loss_it":[], "grad_norm_squared_it":[], "effective_lr_it":[],"train_loss_ep":[], "test_loss_ep":[], "test_acc_ep":[], "train_acc_ep":[], "epoch_time":[]}    
   
    ##### iteration counter
    iteration = 0
    total_steps = int(len(train_loader)*opt.ep)

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

    if opt.alg == 'sgdm':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum = opt.beta1, weight_decay= opt.wd)
        if opt.lr_decay:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)
    if opt.alg == 'ngnm':
        avg_grad_1 = []
        for p in model.parameters():
            avg_grad_1.append(None)

	########### Training ###########     
    for epoch in range(opt.ep):
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

        #saving stats
        results["train_loss_it"].append(train_loss)
        results["train_loss_ep"].append(train_loss)
        results["train_acc_ep"].append(accuracy_train)
        results["test_loss_ep"].append(test_loss)
        results["test_acc_ep"].append(accuracy_test)

        #saving to wandb
        if use_wandb:
            wandb.log({"train_loss":train_loss, "train_acc":accuracy_train,"test_loss":test_loss,"test_acc":accuracy_test}, commit=False)

        print('Epoch {}: Train L: {:.4f}, TrainAcc: {:.2f}, Test L: {:.4f}, TestAcc: {:.2f} \n'.format(epoch, train_loss, accuracy_train, test_loss, accuracy_test))

        ###########  Training Loop ########### 
        model.train()
        for _, batch in enumerate(train_loader):

            model.zero_grad() 
            ###########  Backprop  ###########
            data, target = batch[0].to(device),batch[1].to(device)

            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Getting gradient norm squared
            with torch.no_grad():
                grad_norm_squared = get_grad_norm_squared(model)
                if use_wandb:
                    wandb.log({"grad_norm_squared":grad_norm_squared}, commit=False)
                results["grad_norm_squared_it"].append(grad_norm_squared)
                results["train_loss_it"].append(loss.item())

            ###########  Optimizer update for standard methods  ########### 
            if opt.alg == 'sgdm':
                results["effective_lr_it"].append(optimizer.param_groups[0]["lr"])
                if use_wandb:
                    wandb.log({"effective_lr":optimizer.param_groups[0]["lr"]}, commit=False)
                optimizer.step()
                if opt.lr_decay:
                    scheduler.step()

            if opt.alg == 'ngnm':

                with torch.no_grad(): 
                    #updating grad moving average 
                    norm_avg_grad_squared = 0
                    for p_idx, p in enumerate(model.parameters()):
                        if avg_grad_1[p_idx]==None:
                            avg_grad_1[p_idx] =  p.grad
                        avg_grad_1[p_idx] = opt.beta1 * avg_grad_1[p_idx] + (1-opt.beta1)*p.grad
                        norm_avg_grad_squared = norm_avg_grad_squared + avg_grad_1[p_idx].detach().data.norm(2).item()**2

                    #updating parameters
                    if opt.lr_decay:
                        sigma_curr = cosine_wa_lr(opt.sigma, iteration, total_steps, 0) 
                    else:
                        sigma_curr = opt.sigma
                    preconditioner = sigma_curr/(1+sigma_curr*norm_avg_grad_squared/(2*loss.item()))
                    results["effective_lr_it"].append(preconditioner)
                    for p_idx, p in enumerate(model.parameters()):
                        new_val = p - preconditioner * (avg_grad_1[p_idx]+opt.wd*p)
                        p.copy_(new_val)
                    if use_wandb:
                        wandb.log({"effective_lr":preconditioner}, commit=False)

            iteration = iteration +1
        epoch_time = time.time()-start_time
        results["epoch_time"].append(epoch_time)
        if use_wandb:
            wandb.log({"epoch_time":epoch_time})
            
    ########### Closing Writer ###########  
    if use_wandb:
        run.finish()
        wandb.finish()

    return results
