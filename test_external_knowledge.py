import torch

from models.my_external_knowledge import *
from dgl.dataloading import GraphDataLoader


is_LaySumm = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test module
if __name__ == '__main__':
    configs = {
        'ekg_gnn_num_bases': -1,
        'ekg_gnn_hidden_layers': 3,
        'ekg_gnn_dropout': 0.1,
        'span_emb_size': 512,
    }
    kem = KnowledgeEnhancerModule(configs)

    with open("./resources/eLife_split/train.json", "r") as in_file:
      train = json.loads(in_file.read())
    
    GATModel = GATModel(50, 256, heads=[4,4]).to(device)

    for inst in train[:1]:
        aid = inst['id']
        
        print("---")

        # Get graph features 
        if is_LaySumm:
            text = inst['abstract']
            text = " ".join(text).strip()
            graph_features, G = kem(text, aid, -1)

        for i, text in enumerate(inst['sections']):
            text = " ".join(text).strip()
            graph_features, G = kem(text, aid, i)

        

        # logits = model(batched_graph, features)