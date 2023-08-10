"""
This shows the comparison between a normal distribution and a t-distribution
with DOF degrees of freedom
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

DOF = 3

x = np.linspace(-5, 5, 1000)

normal = stats.norm.pdf(x, 0, 1)
tdist = stats.t.pdf(x, DOF)

plt.plot(x, normal, 'b-', label='Normal Distribution')
plt.plot(x, tdist, 'r-', label='t-Distribution')
plt.legend()
plt.title('Normal Distribution vs. t-Distribution with DOF={}'.format(DOF))
plt.show()
