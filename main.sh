#Code to run main.py for all datasets
set -e

python main.py --dataset citeseer -d 128 -lr 0.001 -e 500 -k 20 -f 5
python main.py --dataset cora -d 128 -lr 0.001 -e 500 -k 20 -f 5
python main.py --dataset facebook -d 128 -lr 0.001 -e 500 -k 20 -f 5

#python main.py --dataset credit -d 128 -lr 0.001 -e 300 -k 20 -f 5
#python main.py --dataset pubmed -d 128 -lr 0.001 -e 300 -k 20 -f 5

#python cfo.py --dataset citeseer -d 128 -lr 0.001 -e 300 -k 20 -f 5
#python cfo.py --dataset cora -d 128 -lr 0.001 -e 300 -k 20 -f 5
#python cfo.py --dataset facebook -d 128 -lr 0.001 -e 300 -k 20 -f 5

#python cfo.py --dataset credit -d 128 -lr 0.001 -e 300 -k 20 -f 5
#python cfo.py --dataset pubmed -d 128 -lr 0.001 -e 300 -k 20 -f 5

#python gfo.py --dataset citeseer -d 128 -e 300 -k 20 -f 5
#python gfo.py --dataset cora -d 128 -e 300 -k 20 -f 5
#python gfo.py --dataset facebook -d 128 -e 300 -k 20 -f 5