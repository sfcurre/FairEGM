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

def create_feature_matrix_singleEgo(dir, n, ego_id, old2new, contain_ego_id):
	feat2fid = Indexer()  # d: feature name -> id,  v: id -> feature name,  num: number of features
	id2fid = defaultdict(list) # each node's features
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
		if int(eles[0]) not in old2new.d.keys():
			continue
		node_id = old2new.d[int(eles[0])]
		# assert len(eles) - 1 == len(ego2global.keys()), "line %s \n, ego %d, node_id %d, len(eles)-1 %d, # ego %d"%(line, ego_id, node_id, len(eles) - 1, len(ego2global.keys()))
		for i in range(1, len(eles)):
			if eles[i] == "1":
				id2fid[node_id].append(ego2global[i - 1])

	if contain_ego_id:
		# read ego node features
		file_egofeat = dir + str(ego_id) + ".egofeat"
		infile = open(file_egofeat)
		for line in infile:
			eles = line.strip().split()
			# assert len(eles) == len(ego2global.keys()), "line %s \n, ego %d, node_id %d, len(eles)-1 %d, # ego %d"%(line, ego_id, node_id, len(eles) - 1, len(ego2global.keys()))
			for i in range(len(eles)):
				if eles[i] == "1":
					id2fid[0].append(ego2global[i])

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

def create_adjacency_matrix_singleEgo(dir, ego_id, contain_ego_id):
	file_edge = dir + str(ego_id) + ".edges"
	old2new = Indexer()
	if contain_ego_id:
		old2new.add_object(ego_id)
	edgelist = list()

	# read edges
	infile = open(file_edge)
	for line in infile:
		eles = line.strip().split()
		# assert len(eles) == 2, "line %s len(eles) %d"%(line, len(eles))
		n0, n1 = [int(ele) for ele in eles]
		new_n0, new_n1 = old2new.get_id_add(n0), old2new.get_id_add(n1)
		# assert n0 <= n1, "line %s n0 %d n1 %d"%(line, n0, n1)
		edgelist.append([new_n0, new_n1])

	n = old2new.num
	adj = np.zeros((n, n), dtype=np.int32)

	for edge in edgelist:
		adj[edge[0], edge[1]] = adj[edge[1], edge[0]] = 1

	if contain_ego_id:
		# ego_center is connected to every node in the ego network
		for node in range(1, old2new.num):
			adj[0, node] = adj[node, 0] = 1

	print("ego_id %d, node_num %d, edge_num in edgelist %d"%(ego_id, n, len(edgelist)))
	return adj, n, old2new

def write_mapping(filename, rgn, mapping, nl=False):
	f = open(filename, "w")
	for i in rgn:
		f.write("%d %s%s"%(i, mapping[i], '\n' if nl else ''))
	f.close()

if __name__ == "__main__":
	EGO_DIR = "facebook/"
	ego_id = 1684
	contain_ego_id = False  # ICLR2021 set it to false, but it should be true

	# create feature matrix
	adj, N, old2new = create_adjacency_matrix_singleEgo(EGO_DIR, ego_id, contain_ego_id)
	feat, feat2fid = create_feature_matrix_singleEgo(EGO_DIR, N, ego_id, old2new, contain_ego_id)

	# output
	np.savetxt("fb_features_ego_%d.txt"%(ego_id), feat, fmt='%1d')
	np.savetxt("fb_adjacency_%d.txt"%(ego_id), adj, fmt='%1d')

	# write mappings
	write_mapping("fb_featnames_%d.txt"%(ego_id), range(feat2fid.num), feat2fid.v)  # fid in feature matrix -> meaning
	write_mapping("fb_id_old2new_%d.txt"%(ego_id), old2new.d.keys(), old2new.d, nl=True)
	write_mapping("fb_id_new2old_%d.txt"%(ego_id), range(N), old2new.v, nl=True)
