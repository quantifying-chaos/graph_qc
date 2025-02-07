import numpy as np
import matplotlib.pyplot as plt

def logistic(l, x): 
    return 4 * l* x * (1 -x)

def logistic_con(c, x): 
    return np.exp(c * x) / (1 + np.exp(c * x))


def cal_y(l, x_0, n):
    y = []
    for i in range(n):
        x_1 = logistic(l, x_0)
        y.append(x_1)
        x_0 = x_1
    return y

# calculate the discrete logistic map
y = []
y_0 =  0.0003
for i in range(20):
    y_1 = logistic(0.5, y_0)
    y.append(y_1)
    y_0 = y_1


# create two subplots
fig, ax = plt.subplots(1,2, figsize=(14, 7))
# set common label 
frame = fig.add_subplot(111, frameon=False)
frame.set_xticks([])
frame.set_yticks([])
frame.set_ylabel('Population', fontsize=20, labelpad=40)


# mark each data point with a red circle and connecting with blue line
# remove x, y labels 
ax[0].plot(y, '-bo', markersize=10)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_xlabel('Time', fontsize=18, labelpad=20)
ax[0].set_title('Discrete Logistic Map', fontsize=20, y=-0.15)

# calculate the continuous logistic map
x = np.linspace(-10, 10, 500)
z = logistic_con(1, x)

ax[1].plot(x, z, label='logistic_con', linewidth=2)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_xlabel('Time', fontsize=18, labelpad=20)
ax[1].set_title('Continuous Logistic Map', fontsize=20, y=-0.15)

plt.tight_layout()
# plt.show()

plt.savefig('logistic_map.png')

