'''
Construct and visualise a discourse-based graph from the collected data.
'''

import torch
import dgl 
from constants import *
import matplotlib.pyplot as plt
import networkx as nx
import json
import pprint

pp = pprint.PrettyPrinter(indent=4)


# Get graph data
with open("resources/eLife_split/train_disc_graphs.jsonl", "r") as f:
  lines = f.readlines()
  graph_data = json.loads(lines[0])
  print(graph_data['id'])


with open("resources/eLife_split/train.json", "r") as f:
  data_text = json.loads(f.read())
  data_text = [x for x in data_text if x['id'] == "elife-35500-v1"][0]
  pp.pprint(" ".join(data_text['sections'][1]))


# Relation types
relation_types = set()
rel_map = {}
with open(UMLS_RELTYPES_FILE, 'r') as f:
  for line in f:
    relation_types.add(line.strip().split('|')[1])
    rel_map[line.strip().split('|')[1]] = line.strip().split('|')[0]


# Semantic types
node_map = {}
with open(UMLS_SEMTYPES_FILE, 'r') as f:
  for line in f:
    comps = line.strip().split('|')
    node_map[comps[0]] = comps[2]

for rel in ['contains', 'has_title' 'has_keyword', 'was_published_in']:
  relation_types.add(rel)
  rel_map[rel] = rel

print("---")


# Construct graph
u, v = [], []
edges = graph_data["edges"]

# Add missing edges!!
for n in graph_data["nodes"]:
  if graph_data['id'] + "_S" in n:
    edges.append((graph_data['id'], "contains" , n))

central_node = graph_data['id'] + "_Sec1" #"C3245479" #graph_data['id'] + "_Sec1" #graph_data['id'] + "_Abs" 

edges = [e for e in edges if (e[0] == central_node or e[2] == central_node)]
pp.pprint(edges)
# print(edges)


# mapping dicts
#nodes = graph_data["nodes"]
nodes = list(set([e[0] for e in edges] + [e[2] for e in edges]))
node_dict = {n: i for i, n in enumerate(nodes)}

for n1, r, n2 in edges:
  u.append(node_dict[n1])
  v.append(node_dict[n2])

# print(node_map)
# print("---")
# print(rel_map)

g = dgl.graph((u, v))
options = {
    'node_color': 'black',
    'node_size': 20,
    'width': 1,
}

G = dgl.to_networkx(g)
plt.figure(figsize=[15,7])
nx.draw(G, **options)
plt.savefig("Graph.png", format="PNG")
