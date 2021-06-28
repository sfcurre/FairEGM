#Code to run main.py for all datasets
set -e

python main.py --dataset citeseer -d 128 -lr 0.001 -e 500 -k 5 10 20 40 -f 5
python main.py --dataset cora -d 128 -lr 0.001 -e 500 -k 5 10 20 40 -f 5
python main.py --dataset facebook -d 128 -lr 0.001 -e 500 -k 5 10 20 40 -f 5
python main.py --dataset german -d 128 -lr 0.001 -e 500 -k 5 10 20 40 -f 5

#python main.py --dataset bail -d 128 -lr 0.001 -e 500 -k 5 10 20 40 -f 5
#python main.py --dataset credit -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5
#python main.py --dataset pubmed -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5

#python cfo.py --dataset citeseer -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5
#python cfo.py --dataset cora -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5
#python cfo.py --dataset facebook -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5
#python cfo.py --dataset german -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5

#python gfo.py --dataset citeseer -d 128 -e 300 -k 5 10 20 40 -f 5
#python gfo.py --dataset cora -d 128 -e 300 -k 5 10 20 40 -f 5
#python gfo.py --dataset facebook -d 128 -e 300 -k 5 10 20 40 -f 5
#python gfo.py --dataset german -d 128 -e 300 -k 5 10 20 40 -f 5

#python gfo.py --dataset bail -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5
#python gfo.py --dataset credit -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5
#python gfo.py --dataset pubmed -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5

#python baselines.py --dataset citeseer -d 128 -lr 0.001 -e 500 -k 5 10 20 40 -f 5
#python baselines.py --dataset cora -d 128 -lr 0.001 -e 500 -k 5 10 20 40 -f 5
#python baselines.py --dataset facebook -d 128 -lr 0.001 -e 500 -k 5 10 20 40 -f 5
#python baselines.py --dataset german -d 128 -lr 0.001 -e 500 -k 5 10 20 40 -f 5

#python baselines.py --dataset bail -d 128 -lr 0.001 -e 500 -k 5 10 20 40 -f 5
#python baselines.py --dataset credit -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5
#python baselines.py --dataset pubmed -d 128 -lr 0.001 -e 300 -k 5 10 20 40 -f 5
