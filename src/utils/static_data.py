import json

with open("data/set16/static/item.json") as f:
    ITEMS = json.load(f)

with open("data/set16/static/unit.json") as f:
    UNITS = json.load(f)

with open("data/set16/static/trait.json") as f:
    TRAITS = json.load(f)
