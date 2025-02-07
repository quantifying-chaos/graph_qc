import numpy as np
import matplotlib.pyplot as plt

def logistic(l, x): 
    return 4 * l* x * (1 -x)

lambda_vals = [1.2, 1, 0.5, 0.2]
x = np.linspace(-0.05, 1.05, 1000)
fig, ax = plt.subplots(figsize=(7, 7))
for l in lambda_vals:
    ax.plot(x, logistic(l, x), label=f'$\lambda$ =  {l}')

# graph the line y = 1 with dotted line 
ax.axhline(y=1, color='black', linestyle='dotted', label='y = 1') 
# graph y = x 
ax.plot(x, x, label='y = x', linestyle='dashdot')
ax.legend(fontsize=14)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
# set x, y ticks size 
ax.tick_params(axis='both', labelsize=14) 
# plt.show()
plt.savefig('logistic.png')
