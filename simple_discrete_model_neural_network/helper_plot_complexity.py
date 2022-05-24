import matplotlib.pyplot as plt
import numpy as np

import matplotlib
font = {'family' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)

x = np.log([32, 142, 490, 42040])
trained_on = [74.5, 72.5, 58.7, 44.9]
unseen = [22.5, 5.9, 38.6, 39.5]


plt.title(f"Log complexity against average cost of unseen and trained-on failure modes")
plt.xlabel("Log complexity")
plt.ylabel(f"Average cost")



plt.plot(x, trained_on, label="Average cost for trained-on failure modes")
plt.plot(x, unseen, label="Average cost for unseen failure modes")



plt.legend(loc="upper right")  # add a legend
plt.show()
