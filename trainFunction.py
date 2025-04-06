import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn as nn
import numpy as np
from grid_search.functions.usefulFunctions3 import getData3
from grid_search.functions.usefulFunctions2 import initGenDis2, train2 
from grid_search.functions.usefulFunctions import getData
from bayes_opt import BayesianOptimization
from GANs.fElliot import test_architecture, Architecture

def function_to_optimize(batch_size, latent_dim, num_epochs, lr) :
    latent_dim = int(latent_dim)
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    dataroot = "data"
    beta1 = 0.5
    ngpu = 0

    dataloader = getData(dataroot, batch_size)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    netG, netD = initGenDis2(device, ngpu, latent_dim)

    return -train2(netG, netD, lr, beta1, num_epochs, latent_dim, dataloader, device, dataroot)


def fElliot(batch_size, latent_dim, num_epochs, lr) :
    latent_dim = int(latent_dim)
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    archi = Architecture(0.0008, [28], [45, 49, 43, 19, 36, 36, 10, 10, 13, 21], [nn.Sigmoid()], [nn.ReLU(), nn.ReLU(), nn.Tanh(), nn.Tanh(), nn.Tanh(), nn.ReLU(), nn.ReLU(), nn.Tanh(), nn.Tanh(), nn.ReLU()])
    return -test_architecture(batch_size, latent_dim, num_epochs, lr, archi, 1)

# Bounded region of parameter space
pbounds = {'batch_size': (10, 50), 'latent_dim': (1, 50), 'num_epochs': (30, 120), 'lr': (1e-5, 1e-4)}

optimizer = BayesianOptimization(
    f=fElliot,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=5,
)

print(optimizer.max)

f = open("bayesian_optimization/records_Elliot.txt", 'a')
f.write(str(optimizer.max)+'\n\n')
f.close()