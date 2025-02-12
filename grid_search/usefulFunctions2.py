import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from DWasserstein4D import dWasserstein4D
import torch.nn as nn

def evalArchitecture2(nz) :
    class Generator(nn.Module):
        def __init__(self, ngpu, nz = nz):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                nn.Linear(nz, 128, bias = False),
                nn.ReLU(True),
                nn.Linear(128, 256, bias = False),
                nn.ReLU(True),
                nn.Linear(256, 128, bias = False),
                nn.ReLU(True),
                nn.Linear(128, 1, bias = False),
                nn.ReLU(True)
            )

        def forward(self, input):
            return self.main(input)
    
    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                nn.Linear(1, 128, bias = False),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 64, bias = False),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 1, bias = False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)
        
    return Generator, Discriminator

'''
hyperparamètres :
- batch_size : [2;100]
- num_epoch : [10;50]
- nz : [1;50]
- lr : [1e-4;1e-5]
- beta1 : [0;1]
'''

def initGenDis2(device, ngpu, nz) :
    Generator, Discriminator = evalArchitecture2(nz)
    netG = Generator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    netD = Discriminator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    return netG, netD

def getGrid2(lrs, beta1s, num_epochs, nzs) :
    return [[[[(lr, beta1, num_epoch, nz) for lr in lrs] for beta1 in beta1s] for num_epoch in num_epochs] for nz in nzs]

def train2(netG, netD, lr, beta1, num_epochs, nz, dataloader, device, dataroot) :
    criterion = nn.BCELoss()
    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    for _ in range(num_epochs):
        for data in dataloader:
            netD.zero_grad()

            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, nz, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            optimizerD.step()

            ############################
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()
    
    data = np.load(os.path.join(dataroot, "round1.npy"))
    N = data.shape[0]
    genData = np.empty((N, 4))
    for i in range(N):
        noise = torch.randn(b_size, nz, device=device)
        fake = netG(noise)
        genData[i,:] = fake.detach().numpy().T[0]
    
    return dWasserstein4D(data, genData, niter = 1000)