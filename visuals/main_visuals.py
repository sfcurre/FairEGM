import numpy as np, json, argparse
import matplotlib.pyplot as plt
import seaborn as sns, pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KernelDensity

sns.set_theme()
EPOCHS=100

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='facebook', help='The dataset to train models on, one of [cora, citeseer, facebook].')
    return parser.parse_args()

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

    args = parse_args()

    with open(f'results/{args.dataset}/results.json') as fp:
        results = json.load(fp)

    fig, axes = plt.subplots(1, 2, figsize=(10,5))

    tl = []
    tl.append(np.array([fold['history']['task loss'] for fold in results['base']]).mean(axis = 0))
    tl.append(np.array([fold['history']['task loss'] for fold in results['gfo']]).mean(axis = 0))
    tl.append(np.array([fold['history']['task loss'] for fold in results['cfo_10']]).mean(axis = 0))
    tl.append(np.array([fold['history']['task loss'] for fold in results['cfo_100']]).mean(axis = 0))
    tl.append(np.array([fold['history']['task loss'] for fold in results['few']]).mean(axis = 0))
    task_loss = pd.DataFrame(tl, index = ['GCN', 'GFO + GCN', 'CFO$_{10}$ + GCN', 'CFO$_{100}$ + GCN', 'FER + GCN'], columns = 1 + np.arange(EPOCHS)).transpose()
    
    g = sns.lineplot(data=task_loss, legend = False, ax=axes[0])
    g.set_xlabel('Epoch')
    g.set_ylabel('Reconstruction Loss')
    axes[0].legend(['GCN', 'GFO + GCN', 'CFO$_{10}$ + GCN', 'CFO$_{100}$ + GCN', 'FER + GCN'], loc='upper right')
    g.set_xlim(1, EPOCHS)

    fl = []
    fl.append(np.array([fold['history']['fair loss'] for fold in results['base']]).mean(axis = 0))
    fl.append(np.array([fold['history']['fair loss'] for fold in results['gfo']]).mean(axis = 0))
    fl.append(np.array([fold['history']['fair loss'] for fold in results['cfo_10']]).mean(axis = 0))
    fl.append(np.array([fold['history']['fair loss'] for fold in results['cfo_100']]).mean(axis = 0))
    fl.append(np.array([fold['history']['fair loss'] for fold in results['few']]).mean(axis = 0))
    fair_loss = pd.DataFrame(fl, index = ['GCN', 'GFO + GCN', 'CFO$_{10}$ + GCN', 'CFO$_{100}$ + GCN', 'FER + GCN'], columns = 1 + np.arange(EPOCHS)).transpose()

    g = sns.lineplot(data=fair_loss, legend = False, ax = axes[1])
    g.set_xlabel('Epoch')
    g.set_ylabel('Link Divergence')
    g.set_xlim(1, EPOCHS)

    plt.savefig(f'./visuals/images/{args.dataset}_main_training.png')
    plt.clf()

if __name__ == '__main__':
    main()