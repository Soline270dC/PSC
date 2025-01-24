import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from grid_search.usefulFunctions2 import getGrid2

name = "result2nd-1"
testing = [np.linspace(1e-4, 1e-5, 5), np.linspace(0.01, 0.99, 5), list(range(10, 51, 10)), list(range(1, 51, 7))]
parameters = ['lr', 'beta1', 'num_epoch', 'nz']
ticks = [(t, [str(x) for x in t]) for t in testing]
ntest = len(testing)
meanvalues = [len(t)//2 for t in testing]

scores = np.load("grid_search/"+name[:-1]+".npy")
grid = getGrid2(*testing)
logScores = np.log(scores)
arg_min = scores.argmin()
amin = (((arg_min%len(grid[0]))%len(grid[0][0]))%len(grid[0][0][0]), ((arg_min%len(grid[0]))%len(grid[0][0]))//len(grid[0][0][0]), (arg_min%len(grid[0]))//len(grid[0][0]), arg_min//len(grid[0]))

couples = [(s, t) for s in range(ntest) for t in range(ntest) if t > s]

fig = plt.figure()

for s, t in couples :
    ax = fig.add_subplot()

    toPlot = [()]
    cont = ax.contourf(toPlot, locator=ticker.LogLocator(subs = 'all'))
    ax.set_xlabel(parameters[s])
    ax.set_ylabel(parameters[t])
    ax.set_xticks(*ticks[s])
    ax.set_yticks(*ticks[t])
    plt.colorbar(cont)
    ax.scatter(amin[s], amin[t], c = 'red', marker= "x", label = "minimal $d_W$")
    ax.legend()
# fig.savefig("grid_search/"+name)
plt.show()