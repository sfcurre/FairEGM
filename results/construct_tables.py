import json, numpy as np, os

def make_table_from_file(filepath, dataset, mod_list, metrics):
    with open(filepath) as fp:
        results = json.load(fp)
    latex_str = ' & '.join(['model'] + metrics) + ' \\\\\n'
    for model in mod_list:
        addition = '& ' + model.ljust(50)
        for metric in metrics:
            try:
                addition += ' & ' + f'{float(f"{np.mean([fold[metric] for fold in results[model]]):.3g}"):g}'.ljust(10)
            except:
                raise
                addition += ' & ' + '--'
        addition += ' \\\\\n'
        latex_str += addition
    return latex_str

def make_final_table_from_file(filepath, dataset, mod_list, mod_names, metrics):
    with open(filepath) as fp:
        results = json.load(fp)
    latex_str = ' & '.join(['model'] + metrics) + ' \\\\\n'
    for model, mname in zip(mod_list, mod_names):
        addition = '& ' + mname.ljust(10)
        for metric in metrics:
            try:
                addition += ' & ' + f'{float(f"{np.mean([fold[metric] for fold in results[model]]):.3g}"):g}'.ljust(10)
                addition += ' $\pm$ ' + f'{float(f"{np.std([fold[metric] for fold in results[model]]):.3g}"):g}'.ljust(10)
            except:
                raise
                addition += ' & ' + '--'
        addition += ' \\\\\n'
        latex_str += addition
    return latex_str

#=================================================================
def main():
    
    final = 'd-32_d2-16_i-glorot_normal_i2-glorot_normal'
    final_adj = {'citeseer': "tune_fairadj_final/lr-0.005_t2-10",
                 'cora': "tune_fairadj_final/lr-0.001_t2-10",
                 'facebook': "tune_fairadj_final/lr-0.01_t2-10"}
    final_walk = {'citeseer': "tune_fairwalk_final/dim-16_num-walk-20_walk-len-80_lr-0.01_epoch-1",
                  'cora': "tune_fairwalk_final/dim-16_num-walk-20_walk-len-80_lr-0.1_epoch-1",
                  'facebook': "tune_fairwalk_final/dim-16_num-walk-20_walk-len-80_lr-0.1_epoch-1"}
    final_names = ['FairWalk', 'FairAdj', 'Base', 'GFO', 'CFO$_{10}$', 'CFO$_{100}$', 'FEW']
    
    for dataset in ['citeseer', 'cora', 'facebook']:#, 'pubmed']:
    
        mod_list_base = ['base', 'gfo', 'cfo10', 'cfo100', 'few']
        specs = ['d_32_16', 'd_32_32', 'd_64_64', 'd_128_128', 'd_256_256']
        specs += ['d-32_d2-32_i-ones', 'd-32_d2-32_i-zeros', 'd-32_d2-32_i-glorot_normal', 'd-32_d2-32_i-glorot_uniform']
        specs += ['d-32_d2-32_i-glorot_normal_i2-ones', 'd-32_d2-32_i-random_normal_i2-ones', 'd-32_d2-32_i-random_normal_i2-glorot_normal', 'd-32_d2-32_i-random_normal_i2-glorot_uniform']
        specs += ['d-32_d2-32_i-glorot_normal_i2-glorot_normal_c-non_neg', 'd-32_d2-32_i-glorot_normal_i2-glorot_normal', 'd-32_d2-32_i-glorot_normal_i2-glorot_normal_Le-2', 'd-32_d2-32_i-glorot_normal_i2-glorot_normal_Le-3']
        mod_list = mod_list_base + [f'{m}_{s}' for m in mod_list_base for s in specs]
        mod_list_final = [final_walk[dataset], final_adj[dataset]] + [f'{m}_{final}' for m in mod_list_base]

        all_adj = os.listdir(f'./results/baselines/results/Results/{dataset}/tune_fairadj_final') 
        all_walk = os.listdir(f'./results/baselines/results/Results/{dataset}/tune_fairwalk_final')

        mod_list2 = set()
        for filelist, folder in [(all_adj, 'tune_fairadj_final'), (all_walk, 'tune_fairwalk_final')]:
            for fname in filelist:
                mod_list2.add(folder + '/' + '_'.join(fname.split('_')[:-1]))
        
        mod_list += sorted(mod_list2)

        task_metrics = ['reconstruction_loss', 'link_divergence', 'test_auc', 'test_f1']
        fair_metrics = ['recall@10', 'recall@20', 'recall@40', 'dp@10', 'dp@20', 'dp@40', 'dyf10%', 'dyf20%']

        main_table = make_table_from_file(f'./results/{dataset}/results_all.json', dataset, mod_list, task_metrics)
        with open(f'./results/tables/task_metrics_{dataset}.txt', 'w') as fp:
            fp.write(main_table)
        main2_table = make_table_from_file(f'./results/{dataset}/results_all.json', dataset, mod_list, fair_metrics)
        with open(f'./results/tables/fair_metrics_{dataset}.txt', 'w') as fp:
            fp.write(main2_table)
        
        final_table = make_final_table_from_file(f'./results/{dataset}/results_all.json', dataset, mod_list_final, final_names, task_metrics)
        with open(f'./results/tables/task_metrics_{dataset}_final.txt', 'w') as fp:
            fp.write(final_table)
        final2_table = make_final_table_from_file(f'./results/{dataset}/results_all.json', dataset, mod_list_final, final_names, fair_metrics[3:])
        with open(f'./results/tables/fair_metrics_{dataset}_final.txt', 'w') as fp:
            fp.write(final2_table)



if __name__ == '__main__':
    main()