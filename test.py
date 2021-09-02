import numpy as np
from models.metrics import *
from models.losses import *

atr = np.load('negative_ld_loss_attributes.npy')
emb = np.load('negative_ld_loss_embedding.npy')
dot = emb @ emb.T

print(tf.reduce_mean(dp_link_divergence_loss(atr, dot)))
print(dp_link_divergence(atr, dot))

print(dot)