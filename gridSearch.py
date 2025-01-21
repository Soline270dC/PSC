import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
from grid_search.usefulFunctions import *
import matplotlib.pyplot as plt
from time import time

dataroot = "data"
batch_size = 50
nz = 20
num_epochs = 20
lr = 0.0002
beta1 = 0.5
ngpu = 0
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

testing = [[1, 3, 5], [30, 50]]
dataloader = getData(dataroot, batch_size)
grid = getGrid(*testing)
scores = np.zeros((len(grid), len(grid[0])))
n = len(grid)

print("Started loop...")
for i in range(n) :
    print(f"Starting the {i+1}{'st' if (i+1)%10 == 1 else 'nd' if (i+1)%10 == 2 else 'rd' if (i+1)%10 == 3 else 'th'} out of {n} lines of the grid")
    t0 = time()
    for j, t in enumerate(grid[i]) :
        scores[i, j] = train(
            *initGenDis(*t, device, ngpu, nz, weights_init),
            lr, beta1, num_epochs, dataloader, device, nz, dataroot
            )
    print(f"execution time of the line : {time() - t0}")

fig = plt.figure()
ax = fig.add_subplot()
ax.contour(scores)
fig.savefig("grid_search/result")

np.save("grid_search/result", scores)

f = open("grid_search/result.txt", 'a')
f.write(f"\ntesting {testing}\n")
f.write(f"results : score of {scores.min()} for {grid[scores.argmin()]}\n")
print("Done !")