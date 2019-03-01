import numpy as np
import matplotlib.pyplot as plt
from KDEpy import FFTKDE
from scipy.stats import norm

# Generate a distribution and draw 2**6 data points
dist = norm(loc=0, scale=1)
data = dist.rvs(2**6)

# Compute kernel density estimate on a grid using Silverman's rule for bw
x, y1 = FFTKDE().fit(data)(2**10)

# Compute a weighted estimate on the same grid, using verbose API
weights = np.arange(len(data)) + 1
estimator = FFTKDE(kernel='biweight', bw='silverman')
y2 = estimator.fit(data, weights=weights).evaluate(x)

plt.plot(x, y1, label='KDE estimate with defaults')
plt.plot(x, y2, label='KDE estimate with verbose API')
plt.plot(x, dist.pdf(x), label='True distribution')
plt.grid(True, ls='--', zorder=-15); plt.legend()
plt.savefig('output.png', dpi=300)
