import json

from static_data import ITEMS, TRAITS, UNITS


def generate_vocab(output_dir: str) -> None:
    """Generate a vocab from the static data and save it to a json file.

    Args:
        output_dir (str): The directory where the vocab will be saved.
    """
    unit_vocab = {unit: i + 1 for i, unit in enumerate(UNITS)}
    item_vocab = {item: i + 1 for i, item in enumerate(ITEMS)}
    trait_list = []
    for trait, bps in TRAITS.items():
        for bp in sorted(bps):
            trait_list.append(f"{trait}_{bp}")

    trait_vocab = {t: i + 1 for i, t in enumerate(trait_list)}
    with open(f"{output_dir}/unit_vocab.json", "w") as f:
        json.dump(unit_vocab, f, indent=4)
    with open(f"{output_dir}/trait_vocab.json", "w") as f:
        json.dump(trait_vocab, f, indent=4)
    with open(f"{output_dir}/item_vocab.json", "w") as f:
        json.dump(item_vocab, f, indent=4)


if __name__ == "__main__":
    generate_vocab("data/set16/static/vocabulary")
