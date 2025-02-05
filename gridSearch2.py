import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
from matplotlib import ticker
from grid_search.usefulFunctions2 import *
from grid_search.usefulFunctions import getData
import matplotlib.pyplot as plt
from time import time

dataroot = "data"
name = "result2nd-3"
batch_size = 50
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
testing = [np.linspace(8e-5, 15e-4, 10), np.linspace(0.2, 0.8, 10), [45], [40]]
dataloader = getData(dataroot, batch_size)
grid = getGrid2(*testing)
scores = np.zeros((len(grid), len(grid[0]), len(grid[0][0]), len(grid[0][0][0])))
n = len(grid)
m = len(grid[0])
mm = len(grid[0][0])

print("Started loop...")
for i in range(n) :
    print(f"Executing the {i+1}{'st' if (i+1)%10 == 1 and (i+1)//100 != 11 else 'nd' if (i+1)%10 == 2 and (i+1)//100 != 12 else 'rd' if (i+1)%10 == 3 else 'th'} line out of {n}")
    t0 = time()
    for j in range(m) :
        print(f"\tExecuting the {j+1}{'st' if (j+1)%10 == 1 and (j+1)//100 != 11 else 'nd' if (j+1)%10 == 2 and (j+1)//100 != 12 else 'rd' if (j+1)%10 == 3 else 'th'} line out of {m}")
        t1 = time()
        for k in range(mm) :
            print(f"\t\tExecuting the {k+1}{'st' if (k+1)%10 == 1 and (k+1)//100 != 11 else 'nd' if (k+1)%10 == 2 and (k+1)//100 != 12 else 'rd' if (k+1)%10 == 3 else 'th'} line out of {mm}")
            for l, t in enumerate(grid[i][j][k]) :
                scores[i, j, k, l] = train2(
                    *initGenDis2(device, ngpu, t[3], weights_init),
                    *t, dataloader, device, dataroot
                    )
        print(f"\t\texecution time of the subline : {time() - t1}")
    print(f"\texecution time of the line : {time() - t0}")

arg_min = np.unravel_index(scores.argmin(), scores.shape)

np.save("grid_search/"+name, scores)

f = open("grid_search/result2nd.txt", 'a')
f.write(f"\ntesting {testing}\n")
f.write(f"results : score of {scores.min()} for {grid[arg_min[0]][arg_min[1]][arg_min[2]][arg_min[3]]}\n")
print("Done !")


parameters = ['lr', 'beta1', 'num_epoch', 'nz']

fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot()
cont = ax.contourf(scores[arg_min[3],arg_min[2],:,:], locator=ticker.LogLocator(subs = 'all'))
ax.set_xlabel(parameters[0])
ax.set_ylabel(parameters[1])
ax.set_xticks(range(len(testing[0])), testing[0].round(5))
ax.set_yticks(range(len(testing[1])), testing[1].round(3))
plt.colorbar(cont)
ax.scatter(arg_min[0], arg_min[1], c = 'red', marker= "x", label = "minimal $d_W$")
ax.legend()

fig.savefig("grid_search/"+name)

