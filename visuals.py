import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KernelDensity

def kernel_plot(ax, embeddings, attributes):
    sims = 1 - squareform(pdist(embeddings, metric='cosine'))
    sims = np.nan_to_num(sims)
    g1_sims = sims[attributes[:, 0] == 1].flatten()
    g2_sims = sims[attributes[:, 1] == 1].flatten()

    x = np.linspace(0, 1, 1000)
    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')

    kde.fit(g1_sims[:, None])
    g1_logprob = kde.score_samples(x[:, None])

    kde.fit(g2_sims[:, None])
    g2_logprob = kde.score_samples(x[:, None])
    ax.fill_between(x, np.exp(g1_logprob) - np.exp(g2_logprob), alpha=0.5)
    ax.set_ylim(-0.0002, 0.0008)
    ax.set_xlim(0, 1)

def main():

    attributes = np.load('attributes.npy')
    fig, axes = plt.subplots(2, 2)

    embeddings = np.load('base_nodes.npy')
    kernel_plot(axes[0, 0], embeddings, attributes)
    axes[0, 0].set_title('Base')

    embeddings = np.load('targeted_nodes.npy')
    kernel_plot(axes[0, 1], embeddings, attributes)
    axes[0, 1].set_title('Targeted')

    embeddings = np.load('community_nodes.npy')
    kernel_plot(axes[1, 0], embeddings, attributes)
    axes[1, 0].set_title('Community')
    
    embeddings = np.load('reduction_nodes.npy')
    kernel_plot(axes[1, 1], embeddings, attributes)
    axes[1, 1].set_title('Edge Reduction')

    plt.show()

if __name__ == '__main__':
    main()