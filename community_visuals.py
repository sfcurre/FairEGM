import numpy as np
import json, os
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

def main():

    with open('results/community_results.json') as fp:
        results = json.load(fp)

    x = [1, 2, 4, 8, 16] + list(range(30, 360, 10))
    labels = ['Reconstruction Loss', 'Link Divergence', 'Recall@10', 'DP@10']
    fig, axes = plt.subplots(2, 2)
    
    for i, label in enumerate(labels):
        values = []
        for n in x:
            if os.path.exists(f'results/community_{n}_nodes.npy'):
                os.remove(f'results/community_{n}_nodes.npy')
            values.append(results[f'community_{n}'][i])
        ax = axes[i // 2, i % 2]
        ax.plot(x, values)
        ax.set_ylabel(label)
        ax.set_xlim(1, 340)

    axes[1, 0].set_xlabel('Number of Introduced Nodes')
    axes[1, 1].set_xlabel('Number of Introduced Nodes')
    plt.show()

if __name__ == '__main__':
    main()