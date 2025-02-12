import torch
import torch.nn.parallel
import torch.utils.data
from grid_search.usefulFunctions2 import initGenDis2, train2 
from grid_search.usefulFunctions import getData
from bayes_opt import BayesianOptimization

def function_to_optimize(batch_size, nz, num_epochs, lr) :
    nz = int(nz)
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    dataroot = "data"
    beta1 = 0.5
    ngpu = 0

    dataloader = getData(dataroot, batch_size)

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    netG, netD = initGenDis2(device, ngpu, nz)

    return train2(netG, netD, lr, beta1, num_epochs, nz, dataloader, device, dataroot)

# Bounded region of parameter space
pbounds = {'batch_size': (10, 50), 'nz': (1, 50), 'num_epochs': (30, 120), 'lr': (1e-5, 1e-4)}

optimizer = BayesianOptimization(
    f=function_to_optimize,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=3,
)

print(optimizer.max)