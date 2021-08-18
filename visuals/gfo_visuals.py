import numpy as np, pandas as pd
import json, os, argparse
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()
EPOCHS = 100

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='facebook', help='The dataset to train models on, one of [cora, citeseer, facebook].')
    return parser.parse_args()

def main():

    args = parse_args()

    with open(f'results/{args.dataset}/lambda_results.json') as fp:
        results = json.load(fp)

    x = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    labels = [('reconstruction loss', 'Reconstruction Loss'),
              ('link divergence', 'Link Divergence'),
              ('recall@20', 'Recall@20'),
              ('dp@20', 'DP@20')]
    fig, axes = plt.subplots(2, 2, figsize=(10,6))
    

    for i, (key, label) in enumerate(labels):
        values = []
        for n in x:
            val = np.array([fold[key] for fold in results[f'gfo_lambda{n}']]).mean()
            values.append(val)
        ax = axes[i // 2, i % 2]
        ax.plot(x, values)
        ax.set_ylabel(label)
        ax.set_xscale('log')

    axes[1, 0].set_xlabel('Lambda')
    axes[1, 1].set_xlabel('Lambda')

    plt.savefig(f'./visuals/images/{args.dataset}_gfo_lambda_metrics.png')
    plt.clf()

    tl, fl = [],[]
    for n in x:
        tl.append(np.array([fold['history']['task loss'] for fold in results[f'gfo_lambda{n}']]).mean(axis = 0))
        fl.append(np.array([fold['history']['fair loss'] for fold in results[f'gfo_lambda{n}']]).mean(axis = 0))
    
    task_loss = pd.DataFrame(tl, index = x, columns = 1 + np.arange(EPOCHS)).transpose()
    fair_loss = pd.DataFrame(fl, index = x, columns = 1 + np.arange(EPOCHS)).transpose()

    fig, axes = plt.subplots(1, 2, figsize=(10,5))

    g = sns.lineplot(data=task_loss, legend = False, ax=axes[0])
    g.set_xlabel('Epoch')
    g.set_ylabel('Reconstruction Loss')
    axes[0].legend(x, loc='upper right')
    g.set_xlim(1, EPOCHS)

    g = sns.lineplot(data=fair_loss, legend = False, ax = axes[1])
    g.set_xlabel('Epoch')
    g.set_ylabel('Link Divergence')
    g.set_xlim(1, EPOCHS)

    plt.savefig(f'./visuals/images/{args.dataset}_gfo_lambda_training.png')
    plt.clf()

if __name__ == '__main__':
    main()