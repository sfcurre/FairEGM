import numpy as np
import json, os, argparse
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='facebook', help='The dataset to train models on, one of [cora, citeseer, facebook].')
    return parser.parse_args()

def main():

    args = parse_args()

    with open(f'results/{args.dataset}/community_results.json') as fp:
        results = json.load(fp)

    x = [1, 2, 4, 8, 16] + list(range(30, 360, 10))
    labels = [('reconstruction loss', 'Reconstruction Loss'),
              ('link divergence', 'Link Divergence'),
              ('recall@40', 'Recall@40'),
              ('dp@40', 'DP@40')]
    fig, axes = plt.subplots(2, 2, figsize=(10,6))
    

    for i, (key, label) in enumerate(labels):
        values = []
        for n in x:
            val = np.array([fold[key] for fold in results[f'cfo_{n}']]).mean()
            values.append(val)
        ax = axes[i // 2, i % 2]
        ax.plot(x, values)
        ax.set_ylabel(label)
        ax.set_xlim(1, 340)
        ax.ticklabel_format(scilimits=(-3,3))

    axes[1, 0].set_xlabel('Number of Introduced Nodes')
    axes[1, 1].set_xlabel('Number of Introduced Nodes')
    plt.savefig(f'./visuals/images/{args.dataset}_cfo_metrics.png')
    plt.clf()

if __name__ == '__main__':
    main()