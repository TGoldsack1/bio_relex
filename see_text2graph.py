import json
from constants import *
import pickle
from external_knowledge import umls_search_concepts
import pprint

pp = pprint.PrettyPrinter(indent=4)

TEXT2GRAPH = pickle.loads(open(UMLS_TEXT2GRAPH_FILE, 'rb').read().replace(b'\r\n', b'\n'))

print(type(TEXT2GRAPH))

print(len(TEXT2GRAPH.keys()))

# text = 'An oligonucleotide with both GAGAG548 and GAGA558 mutated (55/56 M3) does not bind GAF (lane 5).'
text = 'Because  CtBP binds to the  MLL repression domain, we wished to determine  whether  HPC2 also binds to the  MLL repression domain.'

print("-"*30)
print("text2graph")
print("-"*30)
pp.pprint(TEXT2GRAPH[text])

print("\n")
print("-"*30)
print("Metamap")
print("-"*30)

kg_concepts = umls_search_concepts([text])#[0][0]['concepts']

print("\n")
print("-"*30)
print("UMLS search")
print("-"*30)
pp.pprint(kg_concepts)