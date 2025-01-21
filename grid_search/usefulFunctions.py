import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from grid_search.constructArchitecture import evalArchitecture
from itertools import product
from DWasserstein4D import dWasserstein4D

def getData(dataroot, batch_size) :
    class Dataset():
        def __init__(self, root = dataroot):
            self.root = root
            self.dataset = self.build_dataset()
            self.length = self.dataset.shape[1]

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            step = self.dataset[:, idx]
            return step

        def build_dataset(self):
            dataset = np.load(os.path.join(self.root, "round1.npy")).T
            dataset = torch.from_numpy(dataset).float()
            dataset = torch.unsqueeze(dataset, -1)
            return dataset

    dataset = Dataset(dataroot)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True)
    return dataloader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.in_features
        y = 1.0/np.sqrt(n)
        # nn.init.normal_(m.weight.data, 0.0, y)
        # nn.init.uniform_(m.weight.data, -y, y)
        nn.init.normal_(m.weight.data, 0.0, 1e-2)
    
def initGenDis(nNeuronsGen, nNeuronsDis, device, ngpu, nz, weights_init = weights_init) :
    Generator, Discriminator = evalArchitecture(nz, nNeuronsGen, nNeuronsDis)
    netG = Generator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    netG.apply(weights_init)

    netD = Discriminator(ngpu).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
    netD.apply(weights_init)

    return netG, netD

def getGrid(listLayers, listNeurons) :
    '''
    ==output==
    list of tuples (nLayersGen, nNeuronsGen, nLayersDis, nNeuronsDis) for all combinations
    '''
    output = []
    for n in listLayers :
        temp = list(product(listNeurons, repeat = n))
        output += [x for x in temp]
    return [[[x, y] for x in output] for y in output]

def train(netG, netD, lr, beta1, num_epochs, dataloader, device, nz, dataroot) :
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