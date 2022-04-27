import numpy as np, json, argparse
import matplotlib.pyplot as plt
import seaborn as sns, pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KernelDensity

sns.set_theme()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='facebook', help='The dataset to train models on, one of [cora, citeseer, facebook].')
    parser.add_argument('--view', action='store_true', help='Whether or not to display the generated figure.')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to display.')
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

    with open(f'results/{args.dataset}/results_d-32_d2-16_i-glorot_normal_i2-glorot_normal.json') as fp:
        results = json.load(fp)

    fig, axes = plt.subplots(1, 2, figsize=(15,5))

    EPOCHS = args.epochs

    tl = []
    tl.append(np.array([results['base'][i]['history']['task loss'] for i in range(len(results['base']))]).mean(axis=0)[:EPOCHS])
    tl.append(np.array([results['gfo'][i]['history']['task loss'] for i in range(len(results['gfo']))]).mean(axis=0)[:EPOCHS])
    tl.append(np.array([results['cfo_10'][i]['history']['task loss'] for i in range(len(results['cfo_10']))]).mean(axis=0)[:EPOCHS])
    tl.append(np.array([results['cfo_100'][i]['history']['task loss'] for i in range(len(results['cfo_100']))]).mean(axis=0)[:EPOCHS])
    tl.append(np.array([results['few'][i]['history']['task loss'] for i in range(len(results['few']))]).mean(axis=0)[:EPOCHS])
    task_loss = pd.DataFrame(tl, index = ['GCN', 'GFO + GCN', 'CFO$_{10}$ + GCN', 'CFO$_{100}$ + GCN', 'FEW + GCN'], columns = 1 + np.arange(EPOCHS)).transpose()
    
    g = sns.lineplot(data=task_loss, legend = False, ax=axes[0])
    g.set_xlabel('Epoch', fontsize=16)
    g.set_ylabel('Reconstruction Loss', fontsize=16)
    axes[0].legend(['Base', 'GFO', 'CFO$_{10}$', 'CFO$_{100}$', 'FEW'], loc='upper right')
    g.set_xlim(1, EPOCHS)
    g.set_ylim(1.2, 1.6)

    fl = []
    fl.append(np.array([results['base'][i]['history']['fair loss'] for i in range(len(results['base']))]).mean(axis=0)[:EPOCHS])
    fl.append(np.array([results['gfo'][i]['history']['fair loss'] for i in range(len(results['gfo']))]).mean(axis=0)[:EPOCHS])
    fl.append(np.array([results['cfo_10'][i]['history']['fair loss'] for i in range(len(results['cfo_10']))]).mean(axis=0)[:EPOCHS])
    fl.append(np.array([results['cfo_100'][i]['history']['fair loss'] for i in range(len(results['cfo_100']))]).mean(axis=0)[:EPOCHS])
    fl.append(np.array([results['few'][i]['history']['fair loss'] for i in range(len(results['few']))]).mean(axis=0)[:EPOCHS])
    fair_loss = pd.DataFrame(fl, index = ['Base', 'GFO', 'CFO$_{10}$', 'CFO$_{100}$', 'FEW'], columns = 1 + np.arange(EPOCHS)).transpose()

    g = sns.lineplot(data=fair_loss, legend = False, ax = axes[1])
    g.set_xlabel('Epoch', fontsize=16)
    g.set_ylabel('Link Divergence', fontsize=16)
    g.set_xlim(1, EPOCHS)
    axes[1].ticklabel_format(scilimits=(-3,3))

    plt.savefig(f'./visuals/images/{args.dataset}_main_training.png')
    if args.view:
        plt.show()
    plt.clf()

if __name__ == '__main__':
    main()