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

python main.py --dataset citeseer -d 32 -d2 16 -lr 0.0001 -e 300 -k 5 10 20 40 -r 20
python main.py --dataset cora -d 32 -d2 16 -lr 0.0001 -e 300 -k 5 10 20 40 -r 20
python main.py --dataset facebook -d 32 -d2 16 -lr 0.0001 -e 300 -k 5 10 20 40 -r 20
python main.py --dataset pubmed -d 32 -d2 16 -lr 0.0001 -e 300 -k 5 10 20 40 -r 20

python cfo.py --dataset citeseer -d 32 -d2 16 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
python cfo.py --dataset cora -d 32 -d2 16 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
