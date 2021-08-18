#code to generate visuals for all datasets
set -e

python visuals/main_visuals.py --dataset cora
python visuals/main_visuals.py --dataset citeseer
python visuals/main_visuals.py --dataset facebook
python visuals/main_visuals.py --dataset pubmed
python visuals/main_visuals.py --dataset german
python visuals/main_visuals.py --dataset bail

python visuals/cfo_visuals.py --dataset cora
python visuals/cfo_visuals.py --dataset citeseer
python visuals/cfo_visuals.py --dataset facebook
python visuals/cfo_visuals.py --dataset german

python visuals/gfo_visuals.py --dataset cora
python visuals/gfo_visuals.py --dataset citeseer
python visuals/gfo_visuals.py --dataset facebook
python visuals/gfo_visuals.py --dataset german