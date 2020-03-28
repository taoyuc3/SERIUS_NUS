# https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html#Exploring-Seaborn-Plots
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('classic')

sns.set()

tips = sns.load_dataset('tips')
tips.head()

# Faceted histograms
tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill']

grid = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
grid.map(plt.hist, "tip_pct", bins=np.linspace(0, 40, 15))

# Factor plots can be useful for this kind of visualization as well
with sns.axes_style(style='ticks'):
    g = sns.catplot("day", "total_bill", "sex", data=tips, kind="box")
    g.set_axis_labels("Day", "Total Bill")

# joint distribution between different data sets, along with the associated marginal distributions
with sns.axes_style('white'):
    sns.jointplot("total_bill", "tip", data=tips, kind='hex')

sns.jointplot("total_bill", "tip", data=tips, kind='reg')

# use the Planets data that we first saw in Aggregation and Grouping
planets = sns.load_dataset('planets')
planets.head()

with sns.axes_style('white'):
    g = sns.catplot("year", data=planets, aspect=2,
                       kind="count", color='blue')
    g.set_xticklabels(step=5)

# looking at the method of discovery of each of these planets
    with sns.axes_style('white'):
        g = sns.catplot("year", data=planets, aspect=4.0, kind='count',
                           hue='method', order=range(2001, 2015))
        g.set_ylabels('Number of Planets Discovered')

plt.show()

# # !curl -O https://raw.githubusercontent.com/jakevdp/marathon-data/master/marathon-data.csv
# data = pd.read_csv('marathon-data.csv')
# data.head()
#
# # By default, Pandas loaded the time columns as Python strings (type object);
# # we can see this by looking at the dtypes attribute of the DataFrame:
# data.dtypes
#
# # provide a converter for the times
#
#
# def convert_time(s):
#     h, m, s = map(int, s.split(':'))
#     return pd.datetools.timedelta(hours=h, minutes=m, seconds=s)
#
#
# data = pd.read_csv('marathon-data.csv',
#                    converters={'split':convert_time, 'final':convert_time})
# data.head()
# data.dtypes
# data['split_sec'] = data['split'].astype(int) / 1E9
# data['final_sec'] = data['final'].astype(int) / 1E9
# data.head()
#
# # To get an idea of what the data looks like, we can plot a jointplot over the data
# with sns.axes_style('white'):
#     g = sns.jointplot("split_sec", "final_sec", data, kind='hex')
#     g.ax_joint.plot(np.linspace(4000, 16000),
#                     np.linspace(8000, 32000), ':k')
