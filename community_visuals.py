import numpy as np
import json, os
import matplotlib.pyplot as plt

def main():

    with open('results/community_results.json') as fp:
        results = json.load(fp)

    x = [1, 2, 4, 8, 16, 32] + list(range(64, 772, 25))
    labels = ['Reconstruction Loss', 'DP Link Divergence', 'Recall@K=30', 'DP@K=30']
    fig, axes = plt.subplots(2, 2)
    
    for i, label in enumerate(labels):
        values = []
        for n in x:
            if os.path.exists(f'results/community_{n}_nodes.npy'):
                os.remove(f'results/community_{n}_nodes.npy')
            values.append(results[f'community_{n}'][i])
        ax = axes[i // 2, i % 2]
        ax.plot(x, values)
        ax.set_title(label)
    plt.show()

if __name__ == '__main__':
    main()