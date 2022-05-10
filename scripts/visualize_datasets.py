from mcr.causality.scms.examples import scm_dict

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context('talk')
sns.set_style('white')

N = 10**3
savepath = '../experiments/visualize-data/'

for scm_name in scm_dict.keys():
    scm = scm_dict[scm_name]

    context = scm.sample_context(N)
    data = scm.compute()

    plt.figure()
    sns.pairplot(data, plot_kws={'alpha': 0.2, 'marker': '+'})
    plt.savefig(savepath + scm_name + '_pairplot.pdf')
