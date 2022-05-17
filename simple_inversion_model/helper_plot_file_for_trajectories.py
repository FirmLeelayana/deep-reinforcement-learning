import matplotlib.pyplot as plt
import numpy as np

import matplotlib
font = {'family' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)

x = np.array([0, 1, 2, 3, 4, 5,6,7,8,9,10])
ypoints = [1, 3, 0, 0,0,0,0,0,0,0, 0]


plt.title(f"trajectory plots of a single test episode")
plt.xlabel("Time steps")
plt.ylabel(f"trajectory")
plt.plot(x, ypoints, "-b", label="a = 2, b = 1", color='blue')
plt.legend(loc="upper left")  # add a legend
plt.show()
