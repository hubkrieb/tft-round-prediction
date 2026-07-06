import json


def load_vocab(vocab_path: str) -> dict[str, int]:
    """
    Load a vocab from a json file.

    Args:
        vocab_path (str): The path to the vocab file.

    Returns:
        dict[str, int]: The vocab dictionary.
    """
    with open(vocab_path) as f:
        return json.load(f)
