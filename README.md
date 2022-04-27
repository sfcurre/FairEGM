# FairEGM: Fair Link Prediction and Reccomendation via Emulated Graph Modification
A repository for implementing and testing code from the paper FairEGM: Fair Link Prediction and Reccomendation via Emulated Graph Modification.

Sean Current, Yuntian He, Saket Gurukar, and Srinivasan Parthasarathy. "[FairEGM: Fair Link Prediction and Reccomendation via Emulated Graph Modification.](https://arxiv.org/abs/2201.11596)" Arxiv preprint.

## Setup
FairEGM uses Tensorflow >2.5.0 and Scikit-Learn >1.0.0.

## Experiments
All datasets are stored in `./data`. To run FairEGM experiments, use:

    python main.py --dataset cora -d 32 -d2 16 -lr 0.0001 -e 300 -r 20
    
Where `--dataset` can be used to change the experimental dataset (must be one of `citeseer`, `cora`, `facebook`, or `pubmed`). The parameters `-d` and `-d2` can be used to change the dimensions of the first and second GCN layer, respectively. The learning rate and number of training epochs can be changed with `-lr` and `-e`. Finally, `-r` specifies the number of random runs. For a single run, set `-r` to 1.

## Code

Code for FairEGM models are in the `./layers` and `./models` directories. The GFO, CFO, and FEW approaches are implemented as tf.keras.layers.Layer subclasses. These must be used in tandem with a FairModel object from `./models/fair_model.py`, which stacks the GFO, CFO, and FEW convolutions with a learning layer and secondary GCN layer. 
