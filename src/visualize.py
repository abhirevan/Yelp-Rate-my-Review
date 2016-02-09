__author__ = 'priyanka'

import matplotlib.pyplot as plt
import pandas as pd
plt.interactive(True)

plt.show()
stars_vis = pd.read_csv('data\input\\results\wSubjectivity\yelp_academic_dataset_review_Shopping_senti.csv')
pruned_data = stars_vis[['stars','Polarity']]
#grpd_strs = pruned_data.groupby('stars')

plt.rcParams.update(pd.tools.plotting.mpl_stylesheet)
colors = pd.tools.plotting._get_standard_colors(len(pruned_data), color_type='random')

print pruned_data
pruned_data.plot(kind='scatter')
fig, ax = plt.subplots()
ax.set_color_cycle(colors)
#ax.margins(0.05)
#for name, group in enumerate(pruned_data.values):
ax.plot(pruned_data[1], pruned_data[2])
#ax.legend(numpoints=1, loc='upper left')
plt.interactive(False)
#plt.plot(grpd_strs[:1], grpd_strs[:2], kind='scatter')
plt.show()
#plt.savefig('foo.png')