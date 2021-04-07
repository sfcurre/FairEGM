# egonets-Facebook dataset

## Basic information

n = 4039, m = 88234

Downloaded from:	https://snap.stanford.edu/data/egonets-Facebook.html

## Files

Unzip the data in the same directory with `preprocess.py`

.
+-- facebook:				contains 10 ego networks, each has 5 files
+-- facebook_combined.txt:	all edges in the 10 ego networks with consecutive indices starting from 0
+-- readme-Ego.txt:			description for ego network files


## How to use

Run with the following command:

` python preprocess.py
`

It will create 3 new files: `fb_adjacency.txt`, `fb_features.txt`, and `fb_featnames.txt`.
