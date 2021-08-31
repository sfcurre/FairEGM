import json, numpy as np, os

def make_main_from_file(filepath, dataset):
    with open(filepath) as fp:
        results = json.load(fp)
    fix_json_typing(results)
    metrics = ['reconstruction loss', 'link divergence', 'recall@5', 'dp@5', 'recall@10', 'dp@10', 'recall@20', 'dp@20', 'recall@40', 'dp@40']
    model_titles = ['GCN', 'GFO + GCN', 'CFO$_{10}$ + GCN', 'CFO$_{100}$ + GCN', 'FEW + GCN']
    metric_titles = ['Reconstruction Loss', 'Link Divergence', 'Recall@5', 'DP@5', 'Recall@10', 'DP@10', 'Recall@20', 'DP@20', 'Recall@40', 'DP@40']
    latex_str = ' & '.join(['model'] + metric_titles) + ' \\\\\n'
    for i, model in enumerate(['base', 'gfo', 'cfo_10', 'cfo_100', 'few']):
        addition = '& ' + model_titles[i]
        if model == 'cfo_10':
            addition = dataset.capitalize() + ' ' + addition
        for metric in metrics:
            try:
                addition += ' & ' + f'{float(f"{np.mean([fold[metric] for fold in results[model]]):.3g}"):g}'
            except:
                raise
                addition += ' & ' + '--'
        addition += ' \\\\\n'
        latex_str += addition
    return latex_str

def make_baselines_from_file(filepath):
    with open(filepath) as fp:
        results = json.load(fp)
    metrics = ['reconstruction loss', 'link divergence', 'recall@5', 'dp@5', 'recall@10', 'dp@10', 'recall@20', 'dp@20', 'recall@40', 'dp@40']
    latex_str = ' & '.join(['model'] + metrics) + ' \\\\\n'
    for model in ['gae', 'vgae', 'inform', 'fairwalk']:
        addition = model
        for metric in metrics:
            try:
                addition += ' & ' + f'{float(f"{np.mean([fold[metric] for fold in results[model]]):.3g}"):g}'
            except:
                addition += ' & ' + '---'
        addition += ' \\\\\n'
        latex_str += addition
    return latex_str

def fix_json_typing(results):
    for model in ['base', 'gfo', 'cfo_10', 'cfo_100', 'few']:
        for fold in results[model]:
            for metric in ['reconstruction loss', 'link divergence']:
                if type(fold[metric]) is str:
                    fold[metric] = float(fold[metric].split(',')[0].split('(')[-1])

#=================================================================
def main():
    for dataset in ['citeseer', 'cora', 'facebook', 'pubmed', 'bail', 'german']:#, 'credit']:
        main_table = make_main_from_file(f'./results/{dataset}/results.json', dataset)
        with open(f'./results/tables/main_{dataset}.txt', 'w') as fp:
            fp.write(main_table)
        if os.path.exists(f'./results/{dataset}/baseline_results.json'):
            base_table = make_baselines_from_file(f'./results/{dataset}/baseline_results.json')
            with open(f'./results/tables/baseline_{dataset}.txt', 'w') as fp:
                fp.write(base_table)

if __name__ == '__main__':
    main()