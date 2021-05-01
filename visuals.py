import numpy as np, json
import matplotlib.pyplot as plt
import seaborn as sns, pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KernelDensity

sns.set_theme()

def kernel_plot(ax, embeddings, attributes):
    sims = 1 - squareform(pdist(embeddings, metric='cosine'))
    sims = np.nan_to_num(sims)
    g1_sims = sims[attributes[:, 0] == 1].flatten()
    g2_sims = sims[attributes[:, 1] == 1].flatten()

    x = np.linspace(0, 1, 1000)
    kde = KernelDensity(bandwidth=0.2, kernel='gaussian')

    kde.fit(g1_sims[:, None])
    g1_logprob = kde.score_samples(x[:, None])

    kde.fit(g2_sims[:, None])
    g2_logprob = kde.score_samples(x[:, None])
    ax.fill_between(x, np.exp(g1_logprob) - np.exp(g2_logprob), alpha=0.5)
    #ax.set_ylim(-0.0002, 0.0008)
    ax.set_xlim(0, 1)

def main():

    with open('results/loss_history.json') as fp:
        history = json.load(fp)

    fig, axes = plt.subplots(1, 2)

    tl = []
    tl.append(history['base']['loss'])
    tl.append(history['targeted']['tl'])
    tl.append(history['community10']['tl'])
    tl.append(history['community100']['tl'])
    tl.append(history['reduction']['tl'])
    task_loss = pd.DataFrame(tl, index = ['GCN', 'GFO + GCN', 'CFO$_{10}$ + GCN', 'CFO$_{100}$ + GCN', 'FER + GCN'], columns = 1 + np.arange(300)).transpose()
    
    g = sns.lineplot(data=task_loss, legend = False, ax=axes[0])
    g.set_xlabel('Epoch')
    g.set_ylabel('Reconstruction Loss')
    axes[0].legend(['GCN', 'GFO + GCN', 'CFO$_{10}$ + GCN', 'CFO$_{100}$ + GCN', 'FER + GCN'], loc='upper right')
    g.set_xlim(1, 300)
    g.set_ylim(0.2, 0.5)

    fl = []
    fl.append(history['base']['dp_metric'])
    fl.append(history['targeted']['fl'])
    fl.append(history['community10']['fl'])
    fl.append(history['community100']['fl'])
    fl.append(history['reduction']['fl'])
    fair_loss = pd.DataFrame(fl, index = ['GCN', 'GFO + GCN', 'CFO$_{10}$ + GCN', 'CFO$_{100}$ + GCN', 'FER + GCN'], columns = 1 + np.arange(300)).transpose()

    g = sns.lineplot(data=fair_loss, legend = False, ax = axes[1])
    g.set_xlabel('Epoch')
    g.set_ylabel('Link Divergence')
    g.set_xlim(1, 300)
    plt.show()

    return

    attributes = np.load('results/attributes.npy')
    fig, axes = plt.subplots(2, 2)

    embeddings = np.load('results/base_nodes.npy')
    kernel_plot(axes[0, 0], embeddings, attributes)
    axes[0, 0].set_title('Base')

    embeddings = np.load('results/targeted_nodes.npy')
    kernel_plot(axes[0, 1], embeddings, attributes)
    axes[0, 1].set_title('Targeted')

    embeddings = np.load('results/community10_nodes.npy')
    kernel_plot(axes[1, 0], embeddings, attributes)
    axes[1, 0].set_title('Community')
    
    embeddings = np.load('results/reduction_nodes.npy')
    kernel_plot(axes[1, 1], embeddings, attributes)
    axes[1, 1].set_title('Edge Reduction')

    plt.show()

if __name__ == '__main__':
    main()