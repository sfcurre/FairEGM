import os
from collections import defaultdict
import numpy as np

class Indexer:
	def __init__(self):
		self.d = dict()
		self.v = list()
		self.num = 0

	def add_object(self, obj):
		self.d[obj] = self.num
		self.v.append(obj)
		self.num += 1

	def get_id_add(self, obj):
		if obj not in self.d:
			self.add_object(obj)
		return self.d[obj]

def get_filelist(dir, subfix=None):
	filelist = os.listdir(dir)
	if subfix is None:
		return filelist
	elig_list = list()
	l_s = len(subfix)
	for file in filelist:
		if file[-l_s:] == subfix:
			elig_list.append(file)
	return elig_list

def get_ego_networks(dir):
	ego_ids = [int(ele[:-5]) for ele in get_filelist(dir, subfix='.feat')]
	return ego_ids

def create_feature_matrix(dir, n):
	ego_ids = get_ego_networks(dir)	# ego_id for opening a group of ego files
	feat2fid = Indexer()  # d: feature name -> id,  v: id -> feature name,  num: number of features
	id2fid = defaultdict(list) # each node's features
	for ego_id in ego_ids:
		print("Start ego %d"%(ego_id))
		# update feature dictionary, build map from ego feat id to global id
		file_fname = dir + str(ego_id) + ".featnames"
		infile = open(file_fname)
		ego2global = dict()  # mapping from features in ego to features in global network
		for line in infile:
			pos_space = line.find(" ")
			feat_name = line[pos_space+1:]
			feat_id_global = feat2fid.get_id_add(feat_name)
			feat_id_ego = int(line[:pos_space])  # feature id in ego network, each ego network has only a subset of features
			ego2global[feat_id_ego] = feat_id_global

		# read node features
		file_feat = dir + str(ego_id) + ".feat"
		infile = open(file_feat)
		for line in infile:
			eles = line.strip().split()
			node_id = int(eles[0])
			# assert len(eles) - 1 == len(ego2global.keys()), "line %s \n, ego %d, node_id %d, len(eles)-1 %d, # ego %d"%(line, ego_id, node_id, len(eles) - 1, len(ego2global.keys()))
			for i in range(1, len(eles)):
				if eles[i] == "1":
					id2fid[node_id].append(ego2global[i - 1])

		# read ego node features
		file_egofeat = dir + str(ego_id) + ".egofeat"
		infile = open(file_egofeat)
		for line in infile:
			eles = line.strip().split()
			# assert len(eles) == len(ego2global.keys()), "line %s \n, ego %d, node_id %d, len(eles)-1 %d, # ego %d"%(line, ego_id, node_id, len(eles) - 1, len(ego2global.keys()))
			for i in range(len(eles)):
				if eles[i] == "1":
					id2fid[ego_id].append(ego2global[i])

	"""
	Before this step, id2fid[i] may contain duplicate elements. Writing to the matrix can remove it.
	"""
	num_feats = feat2fid.num
	feat_matrix = np.zeros((n, num_feats), dtype=np.int32)
	for i in range(n):
		for j in id2fid[i]:
			feat_matrix[i, j] = 1

	print("Number of features: %d"%(num_feats))
	return feat_matrix, feat2fid

def create_adjacency_matrix(dir, n):
	adj = np.zeros((n, n), dtype=np.int32)
	file_edge = dir + "facebook_combined.txt"

	# read edges
	infile = open(file_edge)
	for line in infile:
		eles = line.strip().split()
		# assert len(eles) == 2, "line %s len(eles) %d"%(line, len(eles))
		n0, n1 = [int(ele) for ele in eles]
		# assert n0 <= n1, "line %s n0 %d n1 %d"%(line, n0, n1)
		adj[n0, n1] = adj[n1, n0] = 1

	return adj


if __name__ == "__main__":
	FEAT_DIR = "facebook/"
	EDGE_DIR = "./"
	N = 4039

	# create feature matrix
	feat, feat2fid = create_feature_matrix(FEAT_DIR, N)
	adj = create_adjacency_matrix(EDGE_DIR, N)

	# output
	np.savetxt("fb_features.txt", feat, fmt='%1d')
	np.savetxt("fb_adjacency.txt", adj, fmt='%1d')

	file_id2feat = "fb_featnames.txt"
	f = open(file_id2feat, "w")
	num_feats = feat2fid.num
	for i in range(num_feats):
		f.write("%d %s"%(i, feat2fid.v[i]))
	f.close()
