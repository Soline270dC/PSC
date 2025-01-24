import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
from grid_search.usefulFunctions2 import *
from grid_search.usefulFunctions import getData
import matplotlib.pyplot as plt
from time import time

dataroot = "data"
name = "result2nd-2"
batch_size = 50
ngpu = 0
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Next thing to test : gen [[2, 3], [30, 40, 50]], disc [[4, 5], [30, 40, 50]]
# expected time : (3^2+3^3)*(3^4+3^5) -> 8h
# gen [[3], [40, 45, 50]], disc [[5], [30, 50]] -> 48 min
# testing = [[1, 3, 5], [30, 50]]
'''
- lr : [1e-4;1e-5]
- beta1 : [0;1]
- num_epoch : [10;50]
- nz : [1;50]
'''
testing = [np.linspace(1e-4, 1e-5, 5), np.linspace(0.01, 0.99, 5), list(range(10, 51, 10)), list(range(1, 51, 7))]
dataloader = getData(dataroot, batch_size)
grid = getGrid2(*testing)
scores = np.zeros((len(grid), len(grid[0]), len(grid[0][0]), len(grid[0][0][0])))
n = len(grid)
m = len(grid[0])

print("Started loop...")
for i in range(n) :
    print(f"Executing the {i+1}{'st' if (i+1)%10 == 1 and (i+1)//100 != 11 else 'nd' if (i+1)%10 == 2 and (i+1)//100 != 12 else 'rd' if (i+1)%10 == 3 else 'th'} line out of {n}")
    t0 = time()
    for j in range(m) :
        print(f"\tExecuting the {j+1}{'st' if (j+1)%10 == 1 and (j+1)//100 != 11 else 'nd' if (j+1)%10 == 2 and (j+1)//100 != 12 else 'rd' if (j+1)%10 == 3 else 'th'} line out of {m}")
        t1 = time()
        for k in range(len(grid[i][j])) :
            for l, t in enumerate(grid[i][j][k]) :
                scores[i, j, k, l] = train2(
                    *initGenDis2(device, ngpu, t[3], weights_init),
                    *t, dataloader, device, dataroot
                    )
        print(f"\t\texecution time of the subline : {time() - t1}")
    print(f"\texecution time of the line : {time() - t0}")

# logScores = np.log(scores)
arg_min = scores.argmin()
amin = (((arg_min%len(grid[0]))%len(grid[0][0]))%len(grid[0][0][0]), ((arg_min%len(grid[0]))%len(grid[0][0]))//len(grid[0][0][0]), (arg_min%len(grid[0]))//len(grid[0][0]), arg_min//len(grid[0]))
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
# fig.savefig("grid_search/"+name)

np.save("grid_search/"+name, scores)

f = open("grid_search/result2nd.txt", 'a')
f.write(f"\ntesting {testing}\n")
f.write(f"results : score of {scores.min()} for {grid[amin[3]][amin[2]][amin[1]][amin[0]]}\n")
print("Done !")