import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from grid_search.usefulFunctions import getGrid

name = "result3b"
testing = [[2, 3], [30, 40, 50], [5], [30, 50]]
xticks = ([0,7], ['2','3'])
yticks = ([100], ['5'])

scores = np.load("grid_search/"+name[:-1]+".npy")
grid = getGrid(*testing)
logScores = np.log(scores)
amin = (scores.argmin()%len(grid[0]), scores.argmin()//len(grid[0]))

fig = plt.figure()
ax = fig.add_subplot()
cont = ax.contourf(logScores, locator=ticker.LogLocator(subs = 'all'))
ax.set_xlabel("Complexity of the Generator (number of layers)")
ax.set_ylabel("Complexity of the Discriminator (number of layers)")
ax.set_xticks(*xticks)
ax.set_yticks(*yticks)
plt.colorbar(cont)
ax.scatter(*amin, c = 'red', marker= "x", label = "minimal $d_W$")
ax.legend()
# fig.savefig("grid_search/"+name)
plt.show()