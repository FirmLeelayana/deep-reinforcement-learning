import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

possible_a_vector = [1, 2, -1, -2]
possible_b_vector = [1, -1]
all_traj = []
index = 0
a_b_combo = []

for a in possible_a_vector:
    for b in possible_b_vector:

        traj = []
        for i in range(100):
            x_val = np.zeros(11)
            u_val = np.zeros(10)

            x_val[0] = random.randint(-3, 3)

            for k in range(10):
                u_val[k] = random.randint(-3, 3)

            x_val[1] = a * x_val[0] + b * u_val[0]
                
            for k in range(1, 3):
                x_val[k+1] = a * x_val[k] + b * u_val[k]

                traj.append(np.array([x_val[k+1], x_val[k], u_val[k]]))

        all_traj.append(traj)
        index += 1
        a_b_combo.append((a, b))

fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for three-dimensional scattered points
for i in range(len(all_traj)):
    xdata = [x[0] for x in all_traj[i]]
    ydata = [x[1] for x in all_traj[i]]
    zdata = [x[2] for x in all_traj[i]]

    a, b = a_b_combo[i]

    ax.scatter3D(xdata, ydata, zdata, cmap='Greens', label=f'a = {a}, b = {b}')

ax.set_xlabel('x[k+1]')
ax.set_ylabel('x[k]')
ax.set_zlabel('u[k]')
plt.title('Augmented state space for different a, b combinations')
plt.legend()
plt.show()
            

