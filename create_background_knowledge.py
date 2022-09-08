import pprint
import logging
import json
from os import path
from datetime import datetime
from external_knowledge import umls_search_concepts


logging.basicConfig(
  filename="./logs/create_graphs.log.{}".format(datetime.timestamp(datetime.now())), 
  level=logging.INFO,
  format = '%(asctime)s | %(levelname)s | %(message)s'
)

ISA_RELATION = 'T186'

# My variable
data_files = [
  "./resources/eLife_split/train.json",
  "./resources/eLife_split/val.json",
  "./resources/eLife_split/test.json",
  "./resources/PLOS_split/train.json",
  "./resources/PLOS_split/val.json",
  "./resources/PLOS_split/test.json",
  "./resources/pubmed-dataset/train.txt",
  "./resources/pubmed-dataset/val.txt",
  "./resources/pubmed-dataset/test.txt"
]

def load_datafile(fp):
  with open(fp, "r") as in_file:
    if "pubmed" in fp:
      data = in_file.readlines()
      data = [json.loads(l) for l in data]
      sentences = [l['sections'][0] for l in data]
      ids = [l['article_id'] for l in data]
    else:
      data = json.loads(in_file.read())
      sentences = [[x["abstract"] + x["sections"][0]] for x in data]
      sentences = [item for sublist in sentences for item in sublist]
      ids = [x["id"] for x in data]

    return ids, sentences


with open("./resources/umls_rels.txt", "r") as in_file:
  lines = in_file.readlines()
  relation_triples = [tuple(l.replace("\n", "").split('|')) for l in lines]

def get_bg_graph(text, prune=False):
  nodes = set()
  edges = set()
  
  # Get concepts using metamap
  kg_concepts = umls_search_concepts([text], prune)[0][0]['concepts']
  
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

  return { 'edges': list(edges), "nodes": list(nodes) }

for fp in data_files:

  ds = fp.split("/")[2] + "/" + fp.split("/")[3]
  ids, sentences = load_datafile(fp)
  out_path = fp.replace(".txt", "_graphs.jsonl") if "pubmed" in fp else fp.replace(".json", "_graphs.jsonl")

  is_existing = path.exists(out_path)
  o_type = "r+" if is_existing else "w"

  with open(out_path, o_type) as out_file:
    i = len(out_file.readlines()) if is_existing else 0

    data = zip(ids[i:], sentences[i:])
    print(i)

    for aid, sents in data:
      logging.info(f'data_file={ds}, idx={i}, id={aid}')
      i += 1
      sents = [s if len(s) < 1000 else s[:1000] for s in sents]

      try:
        text = " ".join(sents).strip()
        out_dict = get_bg_graph(text)
      
      except IndexError:
        logging.info(f'Index error occurred')
        edges = set()
        nodes = set()
        for s in sents:
          sent_dict = get_bg_graph(s, True)
          
          for e in sent_dict['edges']:
            edges.add(e)

          for n in  sent_dict['nodes']:
            nodes.add(n)
        
        out_dict = {
          "edges": list(edges),
          "nodes": list(nodes)
        }

      out_dict['id'] = aid
      out_file.write(json.dumps(out_dict))
      out_file.write("\n")