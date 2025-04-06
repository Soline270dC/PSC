import torch.nn as nn

def constructArchitecture(nNeuronsGen, nNeuronsDis, nz) :
    nLayersGen = len(nNeuronsGen)
    nLayersDis = len(nNeuronsDis)
    generator = []
    for i in range(nLayersGen) :
        generator += [nn.Linear(nNeuronsGen[i-1] if i >= 1 else nz, nNeuronsGen[i] if i < nLayersGen else 1, bias = False), nn.ReLU(True)]
    generator += [nn.Linear(nNeuronsGen[nLayersGen-1], 1, bias = False), nn.ReLU(True)]

    discriminator = []
    for i in range(nLayersDis) :
        discriminator += [nn.Linear(nNeuronsDis[i-1] if i >= 1 else 1, nNeuronsDis[i], bias = False), nn.ReLU(True)]
    discriminator += [nn.Linear(nNeuronsDis[nLayersDis-1], 1, bias = False), nn.Sigmoid()]

    return generator, discriminator

def evalArchitecture(nz, nNeuronsGen, nNeuronsDis) :
    gen, dis = constructArchitecture(nNeuronsGen, nNeuronsDis, nz)
    class Generator(nn.Module):
        def __init__(self, ngpu, nz = nz):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(*gen)

        def forward(self, input):
            return self.main(input)
    
    class Discriminator(nn.Module):
        def __init__(self, ngpu, nz = nz):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(*dis)

        def forward(self, input):
            return self.main(input)
    return Generator, Discriminator