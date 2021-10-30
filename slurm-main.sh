#!/bin/bash
#SBATCH --job-name=FairGraphs
#SBATCH --time=23:59:59
#SBATCH --output="FairGraphs-%j.out"
#SBATCH --account=PAS0166
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=BEGIN,END,FAIL

#Code to run main.py for all datasets

set -x
set -e

source /usr/local/python/3.6-conda5.2/etc/profile.d/conda.sh
conda activate deepml4

cd ~/FairGraphs

python main.py --dataset citeseer -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
python main.py --dataset cora -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
python main.py --dataset facebook -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5

python main.py --dataset citeseer -d 64 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
python main.py --dataset cora -d 64 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
python main.py --dataset facebook -d 64 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5

python main.py --dataset citeseer -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
python main.py --dataset cora -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
python main.py --dataset facebook -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5

python main.py --dataset citeseer -d 256 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
python main.py --dataset cora -d 256 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
python main.py --dataset facebook -d 256 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5

python main.py --dataset citeseer -d 32 -d2 16 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
python main.py --dataset cora -d 32 -d2 16 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
python main.py --dataset facebook -d 32 -d2 16 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5

# python main.py --dataset citeseer -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
# python main.py --dataset cora -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
# python main.py --dataset facebook -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
# python main.py --dataset german -d 128 -lr 0.001 -e 1000 -k 5 10 20 40 -f 5

# python main.py --dataset bail -d 128 -lr 0.01 -e 1000 -k 5 10 20 40 -f 5
# python main.py --dataset credit -d 128 -lr 0.001 -e 100 -k 5 10 20 40 -f 5
# python main.py --dataset pubmed -d 128 -lr 0.0001 -e 200 -k 5 10 20 40 -f 5

python cfo.py --dataset citeseer -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
python cfo.py --dataset cora -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
python cfo.py --dataset facebook -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5

# python gfo.py --dataset citeseer -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
# python gfo.py --dataset cora -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
# python gfo.py --dataset facebook -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5

# python baselines.py --dataset citeseer -d 128 -lr 0.001 -e 100 -k 5 10 20 40 -f 5
# python baselines.py --dataset cora -d 128 -lr 0.001 -e 100 -k 5 10 20 40 -f 5
# python baselines.py --dataset facebook -d 128 -lr 0.001 -e 100 -k 5 10 20 40 -f 5
# python baselines.py --dataset german -d 128 -lr 0.001 -e 100 -k 5 10 20 40 -f 5

# python baselines.py --dataset bail -d 128 -lr 0.001 -e 100 -k 5 10 20 40 -f 5
# python baselines.py --dataset credit -d 128 -lr 0.001 -e 100 -k 5 10 20 40 -f 5
# python baselines.py --dataset pubmed -d 128 -lr 0.001 -e 100 -k 5 10 20 40 -f 5
