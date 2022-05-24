import matplotlib.pyplot as plt
import numpy as np

import matplotlib
font = {'family' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)

x = [1, 2, 3, 5, 10, 20]
trained_on = [2, 2, 3, 2, 0, 0]
unseen = [2, 3, 3, 2, 2, 2]


plt.title(f"Convergence against given number of time steps")
plt.xlabel("Given time steps")
plt.ylabel(f"Number of failure modes that converged")



plt.plot(x, trained_on, label="Trained-on failure mode")
plt.plot(x, unseen, label="Unseen failure mode")



plt.legend(loc="upper right")  # add a legend
plt.show()
