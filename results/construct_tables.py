import json, numpy as np

def make_main_from_file(filepath):
    with open(filepath) as fp:
        results = json.load(fp)
    metrics = ['reconstruction loss', 'link divergence', 'recall@k', 'dp@k']
    model_titles = ['GCN', 'GFO + GCN', 'CFO$_{10}$ + GCN', 'CFO$_{100}$ + GCN', 'FER + GCN']
    metric_titles = ['Reconstruction Loss', 'Link Divergence', 'Recall@20', 'DP@20']
    latex_str = ' & '.join(['model'] + metric_titles) + ' \\\\\n'
    for i, model in enumerate(['base', 'gfo', 'cfo_10', 'cfo_100', 'fer']): #'GCN', 'GFO + GCN', 'CFO$_{10}$ + GCN', 'CFO$_{100}$ + GCN', 'FER + GCN'
        addition = model_titles[i]
        for metric in metrics:
            try:
                addition += ' & ' + f'{float(f"{np.mean([fold[metric] for fold in results[model]]):.3g}"):g}'
            except:
                addition += ' & ' + '--'
        addition += ' \\\\\n'
        latex_str += addition
    return latex_str

def make_baselines_from_file(filepath):
    with open(filepath) as fp:
        results = json.load(fp)
    metrics = ['reconstruction loss', 'link divergence', 'recall@k', 'dp@k']
    latex_str = ' & '.join(['model'] + metrics) + ' \\\\\n'
    for model in ['gae', 'vgae', 'inform']:
        addition = model
        for metric in metrics:
            try:
                addition += ' & ' + f'{float(f"{np.mean([fold[metric] for fold in results[model]]):.3g}"):g}'
            except:
                addition += ' & ' + '---'
        addition += ' \\\\\n'
        latex_str += addition
    return latex_str

#=================================================================
def main():
    for dataset in ['citeseer', 'cora', 'facebook', 'pubmed']:
        main_table = make_main_from_file(f'./results/{dataset}/results.json')
        with open(f'./results/tables/main_{dataset}.txt', 'w') as fp:
            fp.write(main_table)
        if dataset != 'pubmed':
            base_table = make_baselines_from_file(f'./results/{dataset}/baseline_results.json')
            with open(f'./results/tables/baseline_{dataset}.txt', 'w') as fp:
                fp.write(base_table)

if __name__ == '__main__':
    main()