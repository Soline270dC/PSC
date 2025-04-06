import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
from grid_search.functions.usefulFunctions import *
import matplotlib.pyplot as plt
from time import time
from matplotlib import ticker

dataroot = "data"
name = "result4"
batch_size = 50
nz = 50
num_epochs = 20
lr = 0.0002
beta1 = 0.5
ngpu = 0
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Next thing to test : gen [[2, 3], [30, 40, 50]], disc [[4, 5], [30, 40, 50]]
# expected time : (3^2+3^3)*(3^4+3^5) -> 8h
# gen [[3], [40, 45, 50]], disc [[5], [30, 50]] -> 48 min
# testing = [[1, 3, 5], [30, 50]]
testing = [[2, 3], [30, 40, 50], [5], [30, 50]]
dataloader = getData(dataroot, batch_size)
grid = getGrid(*testing)
scores = np.zeros((len(grid), len(grid[0])))
n = len(grid)

print("Started loop...")
for i in range(n) :
    print(f"Starting the {i+1}{'st' if (i+1)%10 == 1 and (i+1)//100 != 11 else 'nd' if (i+1)%10 == 2 and (i+1)//100 != 12 else 'rd' if (i+1)%10 == 3 else 'th'} line out of {n}")
    t0 = time()
    for j, t in enumerate(grid[i]) :
        scores[i, j] = train(
            *initGenDis(*t, device, ngpu, nz, weights_init),
            lr, beta1, num_epochs, dataloader, device, nz, dataroot
            )
    print(f"execution time of the line : {time() - t0}")

# logScores = np.log(scores)
amin = amin = (scores.argmin()%len(grid[0]), scores.argmin()//len(grid[0]))
# fig = plt.figure()
# ax = fig.add_subplot()
# cont = ax.contourf(logScores, locator=ticker.LogLocator(subs = 'all'))
# ax.set_xlabel("Complexity of the Generator (number of layers)")
# ax.set_ylabel("Complexity of the Discriminator (number of layers)")
# ax.set_xticks([0, 2, 10], ['1', '3', '5'])
# ax.set_yticks([0, 2, 10], ['1', '3', '5'])
# plt.colorbar(cont)
# ax.scatter(*amin, c = 'red', marker= "x", label = "minimal $d_W$")
# ax.legend()
# fig.savefig("grid_search/results1"+name)

np.save("grid_search/"+name, scores)

f = open("grid_search/results1/result.txt", 'a')
f.write(f"\ntesting {testing}\n")
f.write(f"results : score of {scores.min()} for {grid[amin[1]][amin[0]]}\n")
print("Done !")