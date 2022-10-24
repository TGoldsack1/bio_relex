import pickle, json

ds = "eLife"


GRAPH_DATA = {}
for split in ["train", "val", "test"]: 
    with open(f"./resources/{ds}_split/{split}_disc_graphs.jsonl", "r") as in_file:
        for line in in_file.readlines():
            line_dict = json.loads(line)
            GRAPH_DATA[line_dict['id']] = line_dict


with open(f"./resources/{ds}_split/disc_graphs.pkl", "wb") as out_file:
    pickle.dump(GRAPH_DATA, out_file)