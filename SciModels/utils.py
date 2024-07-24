import json


def read_json(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.rstrip("\n")))
    f.close()
    return data
