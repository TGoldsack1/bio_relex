import pprint
from external_knowledge import umls_search_concepts

ISA_RELATION = 'T186'

# My variable
# data_source = "./resources/eLife_split/train.json"

with open("./resources/umls_rels.txt", "r") as in_file:
  lines = in_file.readlines()
  relation_triples = [tuple(l.replace("\n", "").split('|')) for l in lines]

def get_bg_graph(text):
  nodes = set()
  edges = set()
  
  # Get concepts using metamap
  kg_concepts = umls_search_concepts([text])[0][0]['concepts']
  
  # Add concept nodes and concept-type relations
  for c in kg_concepts:
    nodes.add(c['cui'])
    for stype in c['semtypes']:
      nodes.add(stype)
      edges.add((c['cui'], ISA_RELATION, stype))

  # add type-type relations from UMLS 
  for rel in relation_triples:
    if (rel[0] in nodes) and (rel[2] in nodes): 
      edges.add(rel)

  return { 'edges': edges, "nodes": nodes }

# text = 'An oligonucleotide with both GAGAG548 and GAGA558 mutated (55/56 M3) does not bind GAF (lane 5).'
# text = 'Because  CtBP binds to the  MLL repression domain, we wished to determine  whether  HPC2 also binds to the  MLL repression domain.'
out = get_bg_graph(text)

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(out)
