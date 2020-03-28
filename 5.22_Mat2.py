import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('classic')

rng = np.random.RandomState(0)
x = np.linspace(0, 10, 500)
y = np.cumsum(rng.randn(500, 8), 0)

sns.set()
data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])
# plot histograms and joint distributions of variables
for col in 'xy':
    plt.hist(data[col], normed=True, alpha=0.5)
# kernel density estimation
    sns.kdeplot(data[col], shade=True)
# kdeplot
    sns.distplot(data['x'])
    sns.distplot(data['y'])
    plt.show()
# joint distribution and the marginal distributions

    sns.kdeplot(data)

    with sns.axes_style('white'):
        sns.jointplot("x", "y", data, kind='kde');

plt.show()
