#code to generate visuals for all datasets
set -e

python visuals/main_visuals.py --dataset cora --epochs 300
python visuals/main_visuals.py --dataset citeseer --epochs 300
python visuals/main_visuals.py --dataset facebook --epochs 300
python visuals/main_visuals.py --dataset pubmed --epochs 200
# python visuals/main_visuals.py --dataset german --epochs 100
python visuals/main_visuals.py --dataset bail --epochs 1000

python visuals/cfo_visuals.py --dataset cora
python visuals/cfo_visuals.py --dataset citeseer
python visuals/cfo_visuals.py --dataset facebook
# python visuals/cfo_visuals.py --dataset german

# python visuals/gfo_visuals.py --dataset cora
# python visuals/gfo_visuals.py --dataset citeseer
# python visuals/gfo_visuals.py --dataset facebook
# python visuals/gfo_visuals.py --dataset german