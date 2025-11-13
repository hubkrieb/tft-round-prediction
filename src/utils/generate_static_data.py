import json

import requests


def generate_static_data(set_number: int, patch: str = "latest") -> tuple[dict, dict]:
    """
    Extracts formatted units and traits static data from Community Dragon.

    Args:
        set_number (int): Set number.
        patch (str): Patch number. Default to "latest".

    Returns:
        tuple[dict, dict]: Tuple containing both units and trait data.
    """
    cdragon_url = f"https://raw.communitydragon.org/{patch}/cdragon/tft/en_us.json"
    response = requests.get(cdragon_url)

    if response.status_code != 200:
        raise Exception("Could not open Community Dragon")
    else:
        data = response.json()

        set_data = data["sets"][set_number]
        unit_info = {}
        for unit in set_data["champions"]:
            unit_name = unit["apiName"]
            traits = unit.get("traits", [])
            cost = unit.get("cost", None)
            unit_info[unit_name] = {"traits": traits, "cost": cost}

        trait_breakpoints = {}
        for trait in set_data["traits"]:
            trait_name = trait["name"]
            breakpoints = trait.get("effects", [])
            trait_breakpoints[trait_name] = [
                bp["minUnits"] for bp in breakpoints if "minUnits" in bp
            ]

        with open("static_data/unit.json", "w") as f:
            json.dump(unit_info, f, indent=4)
        with open("static_data/trait.json", "w") as f:
            json.dump(trait_breakpoints, f, indent=4)
        return unit_info, trait_breakpoints


if __name__ == "__main__":
    generate_static_data("15")
