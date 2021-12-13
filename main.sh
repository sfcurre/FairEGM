#Code to run main.py for all datasets
set -e

# python main.py --dataset citeseer -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
# python main.py --dataset cora -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
# python main.py --dataset facebook -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5

# python main.py --dataset citeseer -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 --vgae
# python main.py --dataset cora -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 --vgae
# python main.py --dataset facebook -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 --vgae

# python main.py --dataset citeseer -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -Le 2
# python main.py --dataset cora -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -Le 2
# python main.py --dataset facebook -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -Le 2 

# python main.py --dataset citeseer -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -Le 3
# python main.py --dataset cora -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -Le 3
# python main.py --dataset facebook -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -Le 3

# python main.py --dataset citeseer -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -c non_neg
python main.py --dataset cora -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -c non_neg
# python main.py --dataset facebook -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -c non_neg

# python main.py --dataset citeseer -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i zeros
# python main.py --dataset cora -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i zeros
# python main.py --dataset facebook -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i zeros

# python main.py --dataset citeseer -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i ones
# python main.py --dataset cora -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i ones
# python main.py --dataset facebook -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i ones

# python main.py --dataset citeseer -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i glorot_normal
# python main.py --dataset cora -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i glorot_normal
# python main.py --dataset facebook -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i glorot_normal

# python main.py --dataset citeseer -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i glorot_uniform
# python main.py --dataset cora -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i glorot_uniform
# python main.py --dataset facebook -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i glorot_uniform

# python main.py --dataset citeseer -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i2 ones
# python main.py --dataset cora -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i2 ones
# python main.py --dataset facebook -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i2 ones

# python main.py --dataset citeseer -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i2 ones -i glorot_normal
# python main.py --dataset cora -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i2 ones -i glorot_normal
# python main.py --dataset facebook -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i2 ones -i glorot_normal

# python main.py --dataset citeseer -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i2 glorot_normal
# python main.py --dataset cora -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i2 glorot_normal
# python main.py --dataset facebook -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i2 glorot_normal

# python main.py --dataset citeseer -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i2 glorot_uniform
# python main.py --dataset cora -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i2 glorot_uniform
# python main.py --dataset facebook -d 32 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5 -i2 glorot_uniform

# python main.py --dataset citeseer -d 64 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
# python main.py --dataset cora -d 64 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
# python main.py --dataset facebook -d 64 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5

# python main.py --dataset citeseer -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
# python main.py --dataset cora -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
# python main.py --dataset facebook -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5

# python main.py --dataset citeseer -d 256 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
# python main.py --dataset cora -d 256 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
# python main.py --dataset facebook -d 256 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5

# python main.py --dataset citeseer -d 32 -d2 16 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
# python main.py --dataset cora -d 32 -d2 16 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
# python main.py --dataset facebook -d 32 -d2 16 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5

# python cfo.py --dataset citeseer -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
# python cfo.py --dataset cora -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5
# python cfo.py --dataset facebook -d 128 -lr 0.0001 -e 300 -k 5 10 20 40 -f 5

# python run_metrics.py --dataset citeseer -f 5
# python run_metrics.py --dataset cora -f 5
# python run_metrics.py --dataset facebook -f 5
