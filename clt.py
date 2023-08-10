import matplotlib
matplotlib.use('TkAgg')

import random
import matplotlib.pyplot as plt

N = 100
T = 10000

data = []
for _ in range(T):
    num_heads = 0
    for _ in range(N):
        num_heads += random.randint(0, 1)
    data.append(num_heads)

plt.title('Number of heads for {} samples of size {}'.format(T, N))
plt.hist(data, density=False)
plt.show()
