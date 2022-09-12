import json
import logging
from os import path
from datetime import datetime
from enum import Enum
import pprint

from external_knowledge import umls_search_concepts

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

# with open("./resources/umls_rels.txt", "r") as in_file:
#   lines = in_file.readlines()
#   relation_triples = [tuple(l.replace("\n", "").split('|')) for l in lines]

def load_datafile(fp):
  with open(fp, "r") as in_file:
    if "pubmed" in fp:
      data = in_file.readlines()
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


def get_discourse_graph(document_dict, prune=False):
  nodes = set()
  edges = set()
  
  nodes.add(document_dict['id']) # document node

  # Title nodes / relations
  if document_dict['title'] != "":
    nodes.add(document_dict['title']) # title node
    edges.add((document_dict['id'], Discourse_Relations.HAS_TITLE.value, document_dict['title']))
    # edges.add((document_dict['title'], Discourse_Relations.IS_TITLE_OF.value, document_dict['id']))

  # Year nodes / relations
  if document_dict['year'] != "":
    nodes.add(document_dict['year'])
    edges.add((document_dict['id'], Discourse_Relations.WAS_PUBLISHED_IN.value, document_dict['year']))
    # edges.add((document_dict['year'], Discourse_Relations.IS_THE_YEAR_OF_PUBLICATION_FOR.value, document_dict['id']))

  # Abstract nodes / relations
  if document_dict['abstract'] != "":
    abstract_node = document_dict['id'] + "_Abs"
    nodes.add(abstract_node)
    edges.add((document_dict['id'], Discourse_Relations.CONTAINS.value, abstract_node))
    # edges.add((abstract_node, Discourse_Relations.IS_CONTAINED_IN.value, document_dict['id']))

    # Abstract sentence nodes / relations
    for i, sentence in enumerate(document_dict['abstract']):
      sentence_node = abstract_node + "_sent" + str(i)
      nodes.add(sentence_node)
      edges.add((abstract_node, Discourse_Relations.CONTAINS.value, sentence_node))
      # edges.add((sentence_node, Discourse_Relations.IS_CONTAINED_IN.value, abstract_node))

  # Keyword nodes / relations
  for kw in document_dict['keywords']:
    nodes.add(kw)
    edges.add((document_dict['id'], Discourse_Relations.HAS_KEYWORD.value, kw))
    # edges.add((kw, Discourse_Relations.IS_KEYWORD_OF.value, document_dict['id']))

  # Section nodes / relations
  for i, section in enumerate(document_dict['sections']):

    section_heading = document_dict['section_names'][i]
    
    # Section node
    sec_node = document_dict['id'] + "_Sec" + str(i)
    nodes.add(sec_node)

    # Section heading node
    nodes.add(section_heading)
    edges.add((sec_node, Discourse_Relations.HAS_TITLE.value, section_heading))
    # edges.add((section_heading, Discourse_Relations.IS_TITLE_OF.value, sec_node))

    # Section sentence nodes / relations
    for j, sent in enumerate(section):
      sent = sent if len(sent) < 1000 else sent[:1000]
      sent_node = sec_node + "_sent" + str(j)
      nodes.add(sent_node)
      edges.add((sec_node, Discourse_Relations.CONTAINS.value, sent_node))
      # edges.add((sent_node, Discourse_Relations.IS_CONTAINED_IN.value, sec_node))

      # Get concepts using metamap
      kg_concepts = umls_search_concepts([sent], prune)[0][0]['concepts']
      
      # Add concept nodes and concept-type relations
      for c in kg_concepts:
        nodes.add(c['cui'])
        edges.add((sent_node, Discourse_Relations.CONTAINS.value, c['cui']))
        # edges.add((c['cui'], Discourse_Relations.IS_CONTAINED_IN.value, sent_node))

        for stype in c['semtypes']:
          nodes.add(stype)
          edges.add((c['cui'], ISA_RELATION, stype))

  # # add type-type relations from UMLS 
  # for rel in relation_triples:
  #   if (rel[0] in nodes) and (rel[2] in nodes): 
  #     edges.add(rel)

  return { 'edges': list(edges), "nodes": list(nodes) }


for fp in data_files:

  ds = fp.split("/")[2] + "/" + fp.split("/")[3]
  ids, sections, section_names, abstracts, titles, keywords, years = load_datafile(fp)
  out_path = fp.replace(".txt", "_disc_graphs.jsonl") if "pubmed" in fp else fp.replace(".json", "_disc_graphs.jsonl")

  is_existing = path.exists(out_path)
  o_type = "r+" if is_existing else "w"

  with open(out_path, o_type) as out_file:
    i = len(out_file.readlines()) if is_existing else 0
    # data = zip(ids[i:], sections[i:], section_names[i:], abstracts[i:], titles[i:], keywords[i:], years[i:])

    for ind in range(i, len(ids)):
      logging.info(f'data_file={ds}, idx={i}, id={ids[ind]}')
      i += 1

      data_dict = {
        "id": ids[ind],
        "sections": sections[ind],
        "section_names": section_names[ind],
        "abstract": abstracts[ind],
        "title": titles[ind],
        "keywords": keywords[ind],
        "year": years[ind],
      }

      out_dict = get_discourse_graph(data_dict, True)
        
      out_dict['id'] = ids[ind]
      out_file.write(json.dumps(out_dict))
      out_file.write("\n")




