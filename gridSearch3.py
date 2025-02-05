import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
from matplotlib import ticker
from grid_search.usefulFunctions2 import *
from grid_search.usefulFunctions3 import *
from grid_search.usefulFunctions import getData
import matplotlib.pyplot as plt
from time import time

dataroot = "data"
name = "result3rd-2"
lr = 8e-5
beta1 = 0.5
nz = 40
ngpu = 0
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Next thing to test : gen [[2, 3], [30, 40, 50]], disc [[4, 5], [30, 40, 50]]
# expected time : (3^2+3^3)*(3^4+3^5) -> 8h
# gen [[3], [40, 45, 50]], disc [[5], [30, 50]] -> 48 min
# testing = [[1, 3, 5], [30, 50]]
'''
- lr : [8e-5;15e-5]
- beta1 : [0.2;0.8]
- num_epoch : [45]
- nz : [40]
'''
testing = [list(range(5, 26, 4)), list(range(30, 101, 10))]

grid = getGrid3(*testing)
scores = np.zeros((len(grid), len(grid[0])))
n = len(grid)
m = len(grid[0])

print("Started loop...")
for i in range(n) :
    print(f"Executing the {i+1}{'st' if (i+1)%10 == 1 and (i+1)//100 != 11 else 'nd' if (i+1)%10 == 2 and (i+1)//100 != 12 else 'rd' if (i+1)%10 == 3 else 'th'} line out of {n}")
    t0 = time()
    for j in range(m) :
        # print(f"\tExecuting the {j+1}{'st' if (j+1)%10 == 1 and (j+1)//100 != 11 else 'nd' if (j+1)%10 == 2 and (j+1)//100 != 12 else 'rd' if (j+1)%10 == 3 else 'th'} line out of {m}")
        t1 = time()
        batch_size, num_epoch = grid[i][j]
        dataloader = getData(dataroot, batch_size)
        scores[i, j] = train2(
            *initGenDis2(device, ngpu, nz, weights_init),
            lr, beta1, num_epoch, nz, dataloader, device, dataroot
            )
        # print(f"\t\texecution time of the subline : {time() - t1}")
    print(f"\texecution time of the line : {time() - t0}")

arg_min = np.unravel_index(scores.argmin(), scores.shape)

np.save("grid_search/"+name, scores)

f = open("grid_search/result3rd.txt", 'a')
f.write(f"\ntesting {testing}\n")
f.write(f"results : score of {scores.min()} for {grid[arg_min[0]][arg_min[1]]}\n")
print("Done !")


parameters = ['batch_size', 'num_epoch']

fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot()
cont = ax.contourf(scores, locator=ticker.LogLocator(subs = 'all'))
ax.set_xlabel(parameters[0])
ax.set_ylabel(parameters[1])
ax.set_xticks(range(len(testing[0])), testing[0])
ax.set_yticks(range(len(testing[1])), testing[1])
plt.colorbar(cont)
ax.scatter(arg_min[1], arg_min[0], c = 'red', marker= "x", label = "minimal $d_W$")
ax.legend()

fig.savefig("grid_search/"+name)