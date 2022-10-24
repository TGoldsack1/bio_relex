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

      for i, line in enumerate(data):
        print(i)
        line = json.loads(line)        

      # data = [json.loads(l) for l in data]
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
      
      split_point = int(len(section) / 2)

      try:
        section_text1 = " ".join(section[:split_point]).strip()
        section_text2 = " ".join(section[split_point:]).strip()
        kg_concepts1 = umls_search_concepts([section_text1])[0][0]['concepts']
        kg_concepts2 = umls_search_concepts([section_text2])[0][0]['concepts']
        kg_concepts = kg_concepts1 + kg_concepts2
      except IndexError:
        print("IndexError2")

        if sec_node not in ["PMC3238543_Sec4", "PMC2571050_Sec7", "PMC4927224_Sec0",
         "PMC4686718_Sec3", "PMC5080807_Sec1", "PMC4361202_Sec0", "PMC4898103_Sec4", "PMC4641989_Sec2",
         "PMC3338224_Sec12", "PMC4738510_Sec2"]:
          section = [s if len(s) < 500 else s[:500] for s in section]

          split_point = int(len(section) / 4)

          section_text1 = " ".join(section[:split_point]).strip()
          section_text2 = " ".join(section[split_point:2*split_point]).strip()
          section_text3 = " ".join(section[2*split_point:3*split_point]).strip()
          section_text4 = " ".join(section[3*split_point:]).strip()
  
          kg_concepts1 = umls_search_concepts([section_text1])[0][0]['concepts']
          print("KG1")
          kg_concepts2 = umls_search_concepts([section_text2])[0][0]['concepts']
          print("KG2")

          if not document_dict['id'] in ["PMC4799521", "PMC3238543", "PMC5047016", "PMC3690703", 
            "PMC2571050", "PMC4927224", "PMC4905508", "PMC4686718", "PMC4360456", 'PMC5080807', 'PMC4361202',
            "PMC4898103", "PMC4641989", "PMC4738510"]:
            kg_concepts3 = umls_search_concepts([section_text3])[0][0]['concepts']
            print("KG3")
            kg_concepts4 = umls_search_concepts([section_text4])[0][0]['concepts']
            print("KG4")
          else:
            kg_concepts3 = []
            kg_concepts4 = []
        else:
          # section_text1 = " ".join(section[:1]).strip()
          kg_concepts1 = umls_search_concepts([section[0]])[0][0]['concepts']
          kg_concepts2 = []
          kg_concepts3 = []
          kg_concepts4 = []

        kg_concepts = kg_concepts1 + kg_concepts2 + kg_concepts3 + kg_concepts4

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
  out_path = fp.replace(".jsonl", "_disc_graphs.jsonl") if "pubmed" in fp else fp.replace(".json", "_disc_graphs.jsonl")

  is_existing = path.exists(out_path)
  o_type = "r+" if is_existing else "w"

  with open(out_path, o_type) as out_file:
    
    # if "pubmed" in fp and "train" in fp:
    #   i = 33019
    # else:
    #   i = len(out_file.readlines()) if is_existing else 0
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
