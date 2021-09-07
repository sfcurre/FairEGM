#!/bin/sh
#SBATCH -N 1
#SBATCH -p batch
#SBATCH --time 2-00:00:00

# Activate Conda Environment
source /home/he.1773/.bashrc
source activate FairGraph
echo $CONDA_PREFIX

# Show which python is used
which python

# Show used nodes
GPUS=$(srun hostname | tr '\n' ' ')
GPUS=${GPUS//".cluster"/""}
echo $GPUS

cd ~/workplace/FairGraphs/

# Commands for Experiments
#python main.py --dataset citeseer -d 128 -lr 0.001 -e 500 -k 5 10 20 40 -f 5
#python main.py --dataset cora -d 128 -lr 0.001 -e 500 -k 5 10 20 40 -f 5
#python main.py --dataset facebook -d 128 -lr 0.001 -e 500 -k 5 10 20 40 -f 5
#python main.py --dataset german -d 128 -lr 0.001 -e 500 -k 5 10 20 40 -f 5
#
#python main.py --dataset bail -d 128 -lr 0.001 -e 500 -k 5 10 20 40 -f 5
#python main.py --dataset credit -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5
#python main.py --dataset pubmed -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5
#
#python cfo.py --dataset citeseer -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5
#python cfo.py --dataset cora -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5
#python cfo.py --dataset facebook -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5
#python cfo.py --dataset german -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5
#
#python gfo.py --dataset citeseer -d 128 -e 300 -k 5 10 20 40 -f 5
#python gfo.py --dataset cora -d 128 -e 300 -k 5 10 20 40 -f 5
#python gfo.py --dataset facebook -d 128 -e 300 -k 5 10 20 40 -f 5
#python gfo.py --dataset german -d 128 -e 300 -k 5 10 20 40 -f 5

#python gfo.py --dataset bail -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5
#python gfo.py --dataset credit -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5
#python gfo.py --dataset pubmed -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5

#echo '*********baselines citeseer*********'
#python baselines_tune_inform.py --dataset citeseer -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -a 0.5
#echo '*********baselines cora*********'
#python baselines_tune_inform.py --dataset cora -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -a 0.5
echo '*********baselines facebook*********'
python baselines_tune_inform.py --dataset facebook -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -a 0.5
#echo '*********baselines german*********'
#python baselines_tune_inform.py --dataset german -d 128 -lr 0.001 -e 500 -k 5 10 20 40 -f 5 -a 0.5
#
#echo '*********baselines bail*********'
#python baselines_tune_inform.py --dataset bail -d 128 -lr 0.01 -e 700 -k 5 10 20 40 -f 5 -a 0.5
#echo '*********baselines credit*********'
#python baselines_tune_inform.py --dataset credit -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5 -a 0.5
#echo '*********baselines pubmed*********'
#python baselines_tune_inform.py --dataset pubmed -d 128 -lr 0.001 -e 100 -k 5 10 20 40 -f 5 -a 0.5