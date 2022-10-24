import dgl
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import re

from utils import *
from constants import *
from models.base import *
from models.helpers import *
from external_knowledge import *
from models.graph.rgcn import RGCNModel
from models.graph.gcn import GraphConvolution

from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

# TEXT2GRAPH = pickle.loads(open(UMLS_TEXT2GRAPH_FILE, 'rb').read().replace(b'\r\n', b'\n'))

TEXT2GRAPH = pickle.loads(open("./resources/eLife_split/disc_graphs.pkl", 'rb').read())#.replace(b'\r\n', b'\n'))

with open(UMLS_SEMTYPES_FILE, "r") as in_file:
    lines = in_file.readlines()
    semtypes = [line.strip().split("|")[0] for line in lines]


def is_concept_node(node_id):
    return re.match(r'^[C][0-9]{7}', node_id)

def get_edges_with_n(node_id, edges):
    return [edge for edge in edges if node_id in edge]

def get_edges_with_n1(node_id, edges):
    return [edge for edge in edges if node_id == edge[0]]

def get_edges_with_n2(node_id, edges):
    return [edge for edge in edges if node_id == edge[2]]

def get_nodes(edges):
    nodes = set()
    for n1, edge_type, n2 in edges:
        nodes.add(n1)
        nodes.add(n2)
    return list(nodes)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_sentence_embeddings(sents):
    sents = [s for s in sents if s]
    encoded_input = tokenizer(sents, padding='max_length', truncation=True, return_tensors='pt', max_length=100).to(device)
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, mean pooling
    pool = mean_pooling(model_output, encoded_input['attention_mask'])

    m = nn.Linear(768, 50).to(device)

    return(m(pool))

def get_graph(nodes, edges):

    # Build DGL graph
    graph_data = {}

    # Process edges
    edgetype2tensor1, edgetype2tensor2, edge_types = {}, {}, set()
    for n1, edge_type, n2 in edges:
        node1_index = nodes.index(n1)
        node2_index = nodes.index(n2)
        if not edge_type in edgetype2tensor1: edgetype2tensor1[edge_type] = []
        if not edge_type in edgetype2tensor2: edgetype2tensor2[edge_type] = []
        edgetype2tensor1[edge_type].append(node1_index)
        edgetype2tensor2[edge_type].append(node2_index)
        edge_types.add(edge_type)
        
    for edge_type in edge_types:
        graph_data[(NODE, edge_type, NODE)] = (torch.tensor(edgetype2tensor1[edge_type]),
                                               torch.tensor(edgetype2tensor2[edge_type]))

    # Finalize the graph
    G = dgl.heterograph(graph_data)
    assert(G.number_of_nodes() == len(nodes))

    return G, nodes


# class MultiHeadGATLayer(nn.Module):
#     def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
#         super(MultiHeadGATLayer, self).__init__()
#         self.heads = nn.ModuleList()
#         for i in range(num_heads):
#             self.heads.append(GATLayer(g, in_dim, out_dim))
#         self.merge = merge

#     def forward(self, h):
#         head_outs = [attn_head(h) for attn_head in self.heads]
#         if self.merge == 'cat':
#             # concat on the output feature dimension (dim=1)
#             return torch.cat(head_outs, dim=1)
#         else:
#             # merge using average
#             return torch.mean(torch.stack(head_outs))


# class GATModel(nn.Module):
#     def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads=4):
#         super(GAT, self).__init__()
#         self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
#         # Be aware that the input dimension is hidden_dim*num_heads since
#         # multiple head outputs are concatenated together. Also, only
#         # one attention head in the output layer.
#         self.layer2 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)

#         self.layer3 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

#     def forward(self, h):
#         h = self.layer1(h)
#         h = F.elu(h)
#         h = self.layer2(h)
#         h = F.elu(h)
#         # h = self.layer3(h)
#         return h

class GATModel(nn.Module):
    def __init__(self, in_size, hid_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # three-layer GAT
        self.gat_layers.append(dglnn.GATConv(in_size, hid_size, heads[0], activation=F.elu))
        self.gat_layers.append(dglnn.GATConv(hid_size*heads[0], hid_size, heads[1], residual=True, activation=F.elu))
        # self.gat_layers.append(dglnn.GATConv(hid_size*heads[1], out_size, heads[2], residual=True, activation=None))

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            # if i == 2:  # last layer 
            #     h = h.mean(1)
            # else:       # other layer(s)
            h = h.flatten(1)
        return h

class KnowledgeEnhancerModule(nn.Module):
    def __init__(self, configs):
        super(KnowledgeEnhancerModule, self).__init__()
        self.configs = configs
        self.cuid2embs = pickle.load(open(UMLS_EMBS, 'rb'))
        print('Size of cuid2embs: {}'.format(len(self.cuid2embs)))

        # Edge types of external knowledge graphs
        self.ekg_etypes = set()
        with open(UMLS_RELTYPES_FILE, 'r') as f:
            for line in f:
                self.ekg_etypes.add(line.strip().split('|')[1])
        
        for rel in ['contains', 'has_title' 'has_keyword', 'was_published_in']:
          self.ekg_etypes.add(rel)
        
        self.ekg_etypes = list(self.ekg_etypes)
        self.ekg_etypes.sort()

        # # RGCNModel for external knowledge graphs
        # self.ekg_gnn_model = RGCNModel(self.ekg_etypes, h_dim=UMLS_EMBS_SIZE,
        #                               num_bases=configs['ekg_gnn_num_bases'],
        #                               num_hidden_layers=configs['ekg_gnn_hidden_layers'],
        #                               dropout=configs['ekg_gnn_dropout'], use_self_loop=True)
        # self.ekg_out_linear = nn.Linear(UMLS_EMBS_SIZE, configs['span_emb_size'])

        # self.gat_model = GATModel(g, in_dim=UMLS_EMBS_SIZE, hidden_dim=1024, \
        #                           out_dim=UMLS_EMBS_SIZE, num_heads=1)
        # self.relu = nn.ReLU()



        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

    def get_initial_embeddings(self, nodes, central_node_id, section_title):
        ret_nodes, ret_embs = [], []
        for n in nodes:
            if is_concept_node(n):           ## Concept nodes ##
                if n in self.cuid2embs:      # has defintion embedding
                    emb = torch.tensor(self.cuid2embs[n]).to(self.device)
                    ret_embs.append(emb)
                    ret_nodes.append(n)
                else:                        # no definition embedding
                    print(f"### {n}")

            else:                            ## Non-concept nodes ##
                if n in semtypes:            # Semantic type node
                    print(f"--- {n}" )
                else:                        # titles, keywords, and section text
                    if n == central_node_id: # sections text
                        #ret_embs.append(get_sentence_embeddings(node_sents))
                        ret_embs.append(get_sentence_embeddings([section_title])[0])
                        ret_nodes.append(n)
                    else:                    # titles and keywords
                        embs = get_sentence_embeddings([n])
                        ret_embs.append(embs[0]) 
                        ret_nodes.append(n)                        
        
        return ret_nodes, ret_embs

    def forward(self, text, aid, section_index):

        is_abstract = (section_index == -1)

        node_id = aid + '_Sec' + str(section_index) if not is_abstract else aid + '_Abs'

        # Extract external kg and apply RGCN on it
        graph_info = TEXT2GRAPH[aid]

        nodes = graph_info['nodes']
        edges = graph_info['edges']

        # if is_abstract:
        #     nodes.append("Abstract")
        #     edges.append((node_id, "has_title", "Abstract"))

        if not is_abstract:
            for edge in edges:
                e1, r, e2 = edge
                if r == "has_title":
                    edges.remove(edge)
                    nodes.remove(e2)
                    sec_title = e2
                    break
        else:
            sec_title = "Abstract"

        used_edges = get_edges_with_n1(node_id, edges)

        # for (n1, edge_type, n2) in used_edges:
        #     if is_concept_node(n2):
        #         used_edges = used_edges + get_edges_with_n1(n2, edges)

        used_nodes = get_nodes(used_edges)
    
        final_nodes, embeddings = self.get_initial_embeddings(used_nodes, node_id, sec_title)
        final_edges = [e for e in used_edges if (e[0] in final_nodes and e[2] in final_nodes)] 

        # print("---")
        # print(len(final_nodes))
        # print(final_nodes)


        # print("---")
        print(len(embeddings))
        # print([e.shape for e in embeddings])

        embeddings = torch.stack(embeddings)
        print(embeddings.shape)

        G = get_graph(final_nodes, final_edges)

        return embeddings, G
        
        # ekg_out_h = self.ekg_gnn_model(ekg_graph, ekg_in_h)[NODE]
        # ekg_out_h = self.relu(self.ekg_out_linear(ekg_out_h))
        # print(ekg_out_h.shape)
