import json
import logging
from os import path
from datetime import datetime
from enum import Enum
import pprint
from external_knowledge import umls_search_concepts
import chardet    


pp = pprint.PrettyPrinter(indent=4)

logging.basicConfig(
  filename="./logs/create_discourse_graphs.log.{}".format(datetime.timestamp(datetime.now())), 
  level=logging.INFO,
  format = '%(asctime)s | %(levelname)s | %(message)s'
)

class Discourse_Relations(Enum):
  CONTAINS = "contains"
  # IS_CONTAINED_IN = "is_contained_in"
  HAS_TITLE = "has_title"
  # IS_TITLE_OF = "is_title_of"
  HAS_KEYWORD = "has_keyword"
  # IS_KEYWORD_OF = "is_keyword_of"
  WAS_PUBLISHED_IN = "was_published_in" # ref. year 
  # IS_THE_YEAR_OF_PUBLICATION_FOR = "is_the_year_of_publication_for"
  
ISA_RELATION = 'T186'

# My variable
data_files = [
  # "./resources/eLife_split/train.json",
  # "./resources/eLife_split/val.json",
  # "./resources/eLife_split/test.json",
  # "./resources/PLOS_split/train.json",
  # "./resources/PLOS_split/val.json",
  # "./resources/PLOS_split/test.json",
  "./resources/pubmed-dataset/train.txt"
  # "./resources/pubmed-dataset/val.jsonl",
  # "./resources/pubmed-dataset/test.jsonl"
]

# with open("./resources/umls_rels.txt", "r") as in_file:
#   lines = in_file.readlines()
#   relation_triples = [tuple(l.replace("\n", "").split('|')) for l in lines]

def load_datafile(fp):
  with open(fp, "r") as in_file:
    if "pubmed" in fp:
      data = in_file.readlines()
      data = [l for l in data if l]
      data = [json.loads(l) for l in data]
      sections = [x['sections'] for x in data]
      section_names = [x['section_names'] for x in data]
      abstracts = ["" for x in data]
      titles = ["" for x in data]
      keywords = [[] for x in data]
      years = ["" for x in data]
      ids = [l['article_id'] for l in data]
    else:
      data = json.loads(in_file.read())
      sections = [x["sections"] for x in data]
      section_names = [x['headings'] for x in data]
      abstracts = [x['abstract'] for x in data]
      titles = [x["title"] for x in data]
      keywords = [x["keywords"] for x in data]
      ids = [x["id"] for x in data]
      years = [x["year"] for x in data]
    
    return ids, sections, section_names, abstracts, titles, keywords, years


def get_discourse_graph(document_dict):
  nodes = set()
  edges = set()
  
  nodes.add(document_dict['id']) # document node

  # Title nodes / relations
  if document_dict['title'] != "":
    nodes.add(document_dict['title']) # title node
    edges.add((document_dict['id'], Discourse_Relations.HAS_TITLE.value, document_dict['title']))

  # Year nodes / relations
  if document_dict['year'] != "":
    nodes.add(document_dict['year'])
    edges.add((document_dict['id'], Discourse_Relations.WAS_PUBLISHED_IN.value, document_dict['year']))

  # Abstract nodes / relations
  if document_dict['abstract'] != "":
    abstract_node = document_dict['id'] + "_Abs"
    print(abstract_node)

    nodes.add(abstract_node)
    edges.add((document_dict['id'], Discourse_Relations.CONTAINS.value, abstract_node))

    abstract = " ".join(document_dict['abstract']).strip()

    # Abstract sentence nodes / relations
    kg_concepts = umls_search_concepts([abstract])[0][0]['concepts']

    for c in kg_concepts:
      nodes.add(c['cui'])
      edges.add((abstract_node, Discourse_Relations.CONTAINS.value, c['cui']))

      for stype in c['semtypes']:
        nodes.add(stype)
        edges.add((c['cui'], ISA_RELATION, stype))
      
  print("abstract_completed")

  # Keyword nodes / relations
  for kw in document_dict['keywords']:
    nodes.add(kw)
    edges.add((document_dict['id'], Discourse_Relations.HAS_KEYWORD.value, kw))

  # Section nodes / relations
  for i, section in enumerate(document_dict['sections']):

    section_heading = document_dict['section_names'][i]
    
    # Section node
    sec_node = document_dict['id'] + "_Sec" + str(i)
    nodes.add(sec_node)
    
    edges.add((document_dict['id'], Discourse_Relations.CONTAINS.value, sec_node))


    print(sec_node)

    # Section heading node
    nodes.add(section_heading)
    edges.add((sec_node, Discourse_Relations.HAS_TITLE.value, section_heading))

    section = section if len(section) < 100 else section[:100]
    section = [s if len(s) < 1000 else s[:1000] for s in section] 

    try:
      section_text = " ".join(section).strip()
      kg_concepts = umls_search_concepts([section_text])[0][0]['concepts']
    except IndexError:
      print("IndexError")
      success = False
      i = 2
      
      while not success:
        split_point = int(len(section) / i)

        try:
          kg_concepts = []
          for j in range(i):
            section_text = " ".join(section[split_point*j:split_point*(j+1)]).strip()
            kg_concepts = kg_concepts + umls_search_concepts([section_text])[0][0]['concepts']
          
          success = True
        
        except IndexError:
          print(f"IndexError{i}")
          i += 2
          if i > 9:
            kg_concepts = umls_search_concepts([section[0]])[0][0]['concepts']
            success = True
          elif i > 7:
            section = [s if len(s) < 125 else s[:125] for s in section]
          elif i > 5:
            section = [s if len(s) < 250 else s[:250] for s in section]
          elif i > 3:
            section = [s if len(s) < 500 else s[:500] for s in section]


    for c in kg_concepts:
      nodes.add(c['cui'])
      edges.add((sec_node, Discourse_Relations.CONTAINS.value, c['cui']))

      for stype in c['semtypes']:
        nodes.add(stype)
        edges.add((c['cui'], ISA_RELATION, stype))

  return { 'edges': list(edges), "nodes": list(nodes) }


for fp in data_files:
  print(fp)

  ds = fp.split("/")[2] + "/" + fp.split("/")[3]
  ids, sections, section_names, abstracts, titles, keywords, years = load_datafile(fp)
  out_path = fp.replace(".txt", "_disc_graphs.jsonl") if "pubmed" in fp else fp.replace(".json", "_disc_graphs.jsonl")

  is_existing = path.exists(out_path)
  o_type = "r+" if is_existing else "w"

  print(out_path)

  with open(out_path, o_type) as out_file:
    
    if "pubmed" in fp and "train" in fp:
      i = 49380
    else:
      i = len(out_file.readlines()) if is_existing else 0
    
    print(len(ids))
    # data = zip(ids[i:], sections[i:], section_names[i:], abstracts[i:], titles[i:], keywords[i:], years[i:])

    for ind in range(i, len(ids)):
      logging.info(f'data_file={ds}, idx={ind}, id={ids[ind]}')

      data_dict = {
        "id": ids[ind],
        "sections": sections[ind],
        "section_names": section_names[ind],
        "abstract": abstracts[ind],
        "title": titles[ind],
        "keywords": keywords[ind],
        "year": years[ind],
      }

      out_dict = get_discourse_graph(data_dict)
        
      out_dict['id'] = ids[ind]
      out_file.write(json.dumps(out_dict))
      out_file.write("\n")
