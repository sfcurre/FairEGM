import json, numpy as np, os

def make_table_from_file(filepath, dataset, mod_list, metrics):
    with open(filepath) as fp:
        results = json.load(fp)
    latex_str = ' & '.join(['model'] + metrics) + ' \\\\\n'
    for model in mod_list:
        addition = '& ' + model.ljust(20)
        for metric in metrics:
            try:
                addition += ' & ' + f'{float(f"{np.mean([fold[metric] for fold in results[model]]):.3g}"):g}'.ljust(10)
            except:
                raise
                addition += ' & ' + '--'
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
    for dataset in ['citeseer', 'cora', 'facebook']:#, 'pubmed']:
    
        mod_list = ['base', 'gfo', 'cfo10', 'cfo100', 'few']
        specs = ['d-32_d2-16', 'd-32_d2-32', 'd-64_d2-64', 'd-128_d2-128', 'd-256_d2-256']
        mod_list = mod_list + [f'{m}_{s}' for m in mod_list for s in specs]
        mod_list += ['fairwalk']
    
        task_metrics = ['reconstruction_loss', 'link_divergence', 'test_auc', 'test_f1']
        fair_metrics = ['recall@10', 'recall@20', 'recall@40', 'dp@10', 'dp@20', 'dp@40', 'dyf10%', 'dyf20%']

        main_table = make_table_from_file(f'./results/{dataset}/results_all.json', dataset, mod_list, task_metrics)
        with open(f'./results/tables/task_metrics_{dataset}.txt', 'w') as fp:
            fp.write(main_table)
        main2_table = make_table_from_file(f'./results/{dataset}/results_all.json', dataset, mod_list, fair_metrics)
        with open(f'./results/tables/fair_metrics_{dataset}.txt', 'w') as fp:
            fp.write(main2_table)

if __name__ == '__main__':
    main()