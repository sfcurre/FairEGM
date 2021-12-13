import json, numpy as np, os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

#=================================================================
def main():
    for dataset in ['citeseer', 'cora', 'facebook']:#, 'pubmed']:
    
        mod_list = ['base', 'gfo', 'cfo10', 'cfo100', 'few']
        specs1 = ['d_32_16', 'd_32_32', 'd_64_64', 'd_128_128', 'd_256_256']
        specs2 = ['d-32_d2-32_i-ones', 'd-32_d2-32_i-zeros', 'd-32_d2-32_i-glorot_normal', 'd-32_d2-32_i-glorot_uniform']
        specs3 = ['d-32_d2-32_i-glorot_normal_i2-ones', 'd-32_d2-32_i-random_normal_i2-ones', 'd-32_d2-32_i-random_normal_i2-glorot_normal', 'd-32_d2-32_i-random_normal_i2-glorot_uniform']
        specs4 = ['d-32_d2-32_i-glorot_normal_i2-glorot_normal', 'd-32_d2-32_i-glorot_normal_i2-glorot_normal_Le-2', 'd-32_d2-32_i-glorot_normal_i2-glorot_normal_Le-3']
        
        task_metrics = ['reconstruction_loss', 'link_divergence', 'test_auc', 'test_f1']
        fair_metrics = ['recall@10', 'recall@20', 'recall@40', 'dp@10', 'dp@20', 'dp@40', 'dyf10%', 'dyf20%']

        metric = 'recall@20'
        #metric = 'dp@20'
        #metric = 'dyf20%'

        with open(f'./results/{dataset}/results_all.json') as fp:
            results = json.load(fp)

        for spec, specname in zip([specs1, specs2, specs3, specs4], ['d', 'i1', 'i2', 'Le']):
            fig = plt.figure(figsize=(16, 9))
            X = np.arange(len(spec))
            for i, mod in enumerate(mod_list):
                vals = []
                for s in spec:
                    vals.append(np.mean([fold[metric] for fold in results[f'{mod}_{s}']]))
                plt.bar(X + 0.15 * i, vals, width=0.15, label=mod)
            plt.title(dataset, fontsize=32)
            plt.ylabel(metric, fontsize=24)
            plt.xticks(X + 0.3, spec, fontsize=16)
            plt.legend(fontsize=16)
            plt.savefig(f"./results/bars/{dataset}_results_{metric}_bar_spec-{specname}.jpg")
            #plt.show()
                        
            
if __name__ == '__main__':
    main()